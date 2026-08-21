---
name: bfb
description: Use when benchmarking Qdrant with bfb — authoring or editing bfb experiment YAML configs (upload / search / scroll), running upload or search benchmarks, measuring recall on real datasets (dbpedia, glove, H&M, cohere-wiki, LAION), A/B-comparing Qdrant builds, interpreting bfb --json results, or debugging dataset download / BFB_DATASETS_DIR problems.
---

# bfb — Qdrant benchmark tool

## Overview

bfb (qdrant/bfb) is a Rust CLI that stress-tests **Qdrant only**, over gRPC
(default `--uri http://localhost:6334`). Build with `cargo build --release` →
`./target/release/bfb`. Auth via `export QDRANT_API_KEY=...`.

**Core model — this is the thing everyone gets wrong:** a YAML config describes
only the *shape* (collection layout, data/query generation); the CLI flags
control the *how* (counts, concurrency, limits, target URI). There is no
`uri:`, `batch_size:`, `parallel:`, or `search_limit:` key in any YAML —
runtime knobs live on the command line, and **unknown YAML fields are hard
parse errors** (every config struct is `deny_unknown_fields`). Do not invent
keys from memory: run `bfb schema` — it prints the complete annotated
upload-config schema (types, defaults, allowed values) as valid YAML you can
copy as a template.

## Commands

| Command | Does |
|---|---|
| `bfb upload --file cfg.yaml [flags]` | (Re)create collection from YAML shape, upload, wait for index. **Deletes an existing collection of the same name unless `--create-if-missing` (no-op if it exists) or `--skip-create` is passed.** |
| `bfb search --file cfg.yaml [flags]` | Run search templates from YAML against an existing collection. Never touches the collection. Collection name comes from the YAML (overrides `--collection-name`). |
| `bfb scroll --file cfg.yaml [flags]` | Scroll/sample workload against an existing collection. |
| `bfb schema` | Print the annotated upload-config schema. Ground truth for upload YAML. |
| `bfb [flags]` (no subcommand) | Legacy flag-driven pipeline: create + upload random/fbin data (+ `--search` / `--scroll`). To only search an existing collection: `bfb --skip-setup --search …`. |

Integer flags accept suffixes `k/M/G/T/ki/Mi/Gi/Ti` and underscores:
`-n 100k`, `--offset 1_000_000`.

## Runtime CLI flags (the "how")

Workload: `-n` points to upload / **queries** to run (scroll: requests; omit on
upload with dataset sources ⇒ full dataset) · `--offset` start row/id (resumes interrupted
dataset uploads; `-n` capped by what remains) · `-m/--max-id` random upserts in
[offset, max_id).

Concurrency: `-p` parallel in-flight requests (closed loop) · `--rps` fixed
request rate, open loop, overrides `-p` · `-t` worker threads · `-c` client
connections per URI (pool = c × #URIs) · `-b` upload batch size (points) ·
`--search-batch-size` queries per search request (default 1; requests issued =
n / batch) · `--throttle` batches/searches per second · `--delay` ms between
requests.

Search: `--search-limit` (default 10) · `--search-hnsw-ef` · `--search-exact` ·
`--search-with-payload` / `--search-with-vectors` · `--prefetch N` (wrap query
in a prefetch of limit N; rescore is disabled inside the prefetch stage) ·
`--indexed-only` · `--timeout` (client deadline + server-side timeout).

Quantized search: bfb **always sends explicit** quantization search params:
rescore = value of `--quantization-rescore`, **false when the flag is absent**;
`--quantization-oversampling` only sent when given. So the default measures the
pure-quantized path; pass `--quantization-rescore true` to measure rescoring.
Campaign scripts still pass `--quantization-rescore false` explicitly so the
run's meta records intent.

Phases: `--skip-create` · `--create-if-missing` · `--skip-upload` ·
`--skip-wait-index` · `--skip-setup` (implies all three) · `--wait-on-upsert` ·
`--set-payload` (payload via separate SetPayload request).

Reliability: each request tries the client pool in random order (failing over
across URIs); `--retry N` adds N more rounds over the pool with
`--retry-interval s` between rounds · `--ignore-errors` keeps going on search
errors. Multiple `--uri` values are allowed.

Distributed: `--shards`, `--replication-factor`, `--write-consistency-factor`,
`--shard-key`, `--write-ordering`, `--read-consistency`.

Output: `--json path` (results document, see below) · `--jsonl-updates` /
`--jsonl-searches` / `--jsonl-rps` (per-request time series; visualize with
qdrant/mri) · `--absolute-time` · `--p9 N` digits in p99… · `--timing-threshold
s` logs slow requests.

Legacy-mode-only shape flags (`-d` dim, `--distance`, `--datatype`,
`--quantization`, `--memory-*` placements, `-k/--keywords` and the other
payload generators, `--sparse-vectors`, `--hnsw-*`, `--segments`, …) mirror the
YAML knobs; see `bfb --help`. Prefer the YAML subcommands for anything
non-trivial.

## Upload config (collection shape)

Single top-level key `collection:`. Full skeleton (defaults in comments; every
section except `vectors`/`sparse_vectors` — at least one vector — is optional):

```yaml
collection:
  name: benchmark             # default "benchmark"
  id: integer                 # integer | uuid — integer ⇒ point id = dataset row (needed for recall)
  on_disk_payload: true       # default true
  shard_number: null
  replication_factor: 1
  write_consistency_factor: 1
  sharding: { method: custom, key: my_field }   # custom shard key (only `custom`)

  hnsw:
    m: 16
    payload_m: null
    ef_construct: 100
    full_scan_threshold: null
    on_disk: false
    inline_storage: false
    memory: null              # cold | cached | pinned — supersedes on_disk (Qdrant 1.19+)

  optimizers:
    default_segment_number: 2
    indexing_threshold: null
    memmap_threshold: null
    max_segment_size: null    # bigger segments search faster, index slower
    deleted_threshold: null         # vacuum trigger fraction (server default 0.2)
    vacuum_min_vector_number: null  # smallest segment vacuum considers (server default 1000)
    prevent_unoptimized: false

  quantization:               # collection-wide; also settable per dense vector
    type: binary              # none | scalar | binary | binary-2bit | binary-1.5bit |
                              # turbo-1bit | turbo-1.5bit | turbo-2bit | turbo-4bit |
                              # product-x4 | -x8 | -x16 | -x32 | -x64
    always_ram: true
    memory: null              # cold | cached | pinned — supersedes always_ram

  vectors:                    # at most ONE entry may omit `name` (the default vector);
    - name: image             # with 2+ entries every one must be named & unique
      size: 1024              # required
      distance: cosine        # cosine | dot | euclid | manhattan
      datatype: float32       # float32 | float16 | uint8 | turbo4 (1.19+)
      on_disk: true
      memory: null            # cold | cached (dense storage can't be pinned)
      multivector: { comparator: max_sim, count: 4 }
      quantization: null      # same shape as collection.quantization
      source: random          # see Sources

  sparse_vectors:
    - name: bm25              # required; names unique across ALL vectors
      datatype: float32       # float32 | float16 | uint8
      on_disk: false
      memory: null            # cold | cached | pinned (inverted index)
      modifier: none          # none | idf — idf required for BM25 scoring & search idf_corpus
      source: { type: random, vocab_size: 100000, length: 100, distribution: zipf }

  payload:                    # payload-wide settings
    memory: null              # cold | cached — supersedes on_disk_payload (1.19+)
    source: null              # whole-payload dataset source, see below

  fields:                     # payload fields: value generation and/or index declaration
    - name: color
      type: keyword           # keyword | integer | float | bool | uuid | geo | text | datetime
      index: true             # false ⇒ unindexed filler payload
      on_disk: false
      memory: null            # cold | cached | pinned — field index placement
      is_tenant: false        # keyword/uuid
      is_principal: false     # integer/float/datetime
      range_index: true       # integer only
      prefix: false           # keyword only; required for search match_prefix filters
      tokenizer: null         # text only: word | whitespace | prefix | multilingual
      source: { type: random, cardinality: 100 }
```

`memory:` everywhere means RAM placement (Qdrant 1.19+): `cold` (load on
demand), `cached` (pre-warmed disk cache, evictable), `pinned` (never evicted).
It supersedes the `on_disk`/`always_ram` booleans; bfb sends both so configs
work on older servers.

### Vector sources

```yaml
source: random                                    # default; random vectors
source: { type: file, path: ./vectors.fbin, strategy: random-sample }  # or from-start;
                                                  # path may be an http(s):// URL (cached)
source:                                           # dense dataset: fields INLINE (flattened)
  type: dataset
  name: glove-25-angular
  format: h5                                      # h5 | tar | sparse | npy | parquet
  path: glove-25-angular/glove-25-angular.hdf5    # relative to datasets dir
  link: http://ann-benchmarks.com/glove-25-angular.hdf5
```

**Asymmetry to remember:** the dense vector dataset source is *flattened*
(dataset fields directly in the source map). Sparse-vector, per-field payload,
and whole-payload dataset sources nest them under a `dataset:` key instead:

```yaml
sparse_vectors:
  - name: bm25
    source:
      type: dataset
      dataset: { name: my-sparse, format: sparse, path: my-sparse/my-sparse, link: https://…tgz }
```

### Payload value generation (`fields[].source`)

Kinds: `random` (default) · `random-clusters` (geo) · `now` (datetime) ·
`dataset` (needs nested `dataset:` **and** `field:` naming the dataset column).
Params by type — irrelevant keys are ignored:

| type | params |
|---|---|
| keyword | `cardinality`, `values_per_point`, `length_multiplier` |
| integer / float | `min`, `max` (+ `values_per_point` for integer) |
| bool | `true_ratio` |
| geo | `clusters` (with `random-clusters`) |
| text | `vocab_size`, `min_length`, `max_length`, `distribution: uniform\|zipf` |
| datetime | `now`, or `min`/`max` range |

### Whole-payload source + index-only fields

For datasets that ship real payload objects (`tar` bundles' `payloads.jsonl`,
`parquet` rows), load the entire object per point and declare only the indexes:

```yaml
payload:
  source:
    type: dataset
    dataset: { name: laion-small-clip, format: tar, path: laion-small-clip/laion-small-clip, link: https://…tgz }
fields:
  - name: similarity          # index-only: no `source` — value comes from payload.source
    type: float
```

Fields present in the object but not listed are uploaded, just unindexed.
Parquet sources take three extra keys: `columns:` (keep only), `exclude:`
(drop; also skips decoding), `fill_null:` (substitute for null/NaN/±inf;
omitted ⇒ field absent).

### Validation rules that bite

At least one dense or sparse vector · one unnamed dense vector max, and only
alone · vector/sparse names unique across both lists · `payload.source` must be
`type: dataset` and must NOT set `field` · per-field dataset source MUST set
`field` · `type: file` local paths must exist (http(s) ok, s3:// rejected) ·
sparse `length <= vocab_size`.

## Datasets (real data)

The real datasets come from Qdrant's **vector-db-benchmark** project: its
`datasets.json` catalog defines each dataset (name, format, path, download
link — bfb's README links the full catalog), and bfb accepts the same fields
spelled inline in a `source:` block, so no local copy of that project is
needed. Formats:

| `format` | Contents | Queries + ground truth (auto-detected) |
|---|---|---|
| `h5` | ann-benchmarks HDF5: `train` (+`test`/`neighbors`) | `test` / `neighbors` |
| `tar` | `.tgz` of `vectors.npy` + `payloads.jsonl` + `tests.jsonl` | `tests.jsonl` `query` / `closest_ids` |
| `sparse` | CSR: `data.csr` (+`queries.csr`/`results.gt`) | `queries.csr` / `results.gt` |
| `npy` | one 2-D float `.npy` — dense vectors only | — |
| `parquet` | payload rows only | — |

h5/tar/sparse are *bundles* (one artifact yields vectors+payloads+queries);
npy/parquet are *components* paired row-by-row in one config (row *i* of each
source lands on point *i*; upload stops at the smallest source, like `zip`).

Downloaded on first use into `./datasets/` — override with **`BFB_DATASETS_DIR`**.
An optional `datasets.json` registry there enables name-only shorthand, but
**bfb eagerly parses the whole registry file and dies on entries it doesn't
model** (e.g. vector-db-benchmark's `jsonl` datasets). Point
`BFB_DATASETS_DIR` at a registry-free directory of symlinks and spell out
`name`/`format`/`path`/`link` inline in the YAML.

Known-good datasets (all cosine):

| name | format | corpus | queries | link |
|---|---|---|---|---|
| dbpedia-openai-100K-1536-angular | tar | 100k × 1536 | 5,000, GT depth 10 | `https://storage.googleapis.com/ann-filtered-benchmark/datasets/dbpedia_openai_100K.tgz` — path `dbpedia-openai-100K-1536-angular/dbpedia_openai_100K` |
| cohere-wiki-1m | tar | 1M × 768 | 999, GT depth 100 | `…/ann-filtered-benchmark/datasets/cohere-wiki-1m.tgz` — path `cohere-wiki-1m/cohere_wiki_1m` |
| h-and-m-2048-angular(-filters) | tar | 105,100 × 2048 + 24 payload fields | (use no-filters set) | `…/ann-filtered-benchmark/datasets/hnm.tgz` — path `h-and-m-2048-angular/hnm` |
| h-and-m-2048-angular-no-filters | tar | same vectors | 10,000 unfiltered, GT depth 10 | `…/ann-filtered-benchmark/datasets/hnm_no_filters.tgz` — path `h-and-m-2048-angular-no-filters/hnm_no_filters` |
| glove-25-angular | h5 | 25-d corpus | h5 `test`/`neighbors` | `http://ann-benchmarks.com/glove-25-angular.hdf5` (any ann-benchmarks .hdf5 works the same way) |
| laion-small-clip | tar | 512-d + payloads | — | `…/ann-filtered-benchmark/datasets/laion-small-clip.tgz` |
| LAION-400M | npy + parquet, 410 parts | ~407M × 512 | — | see `examples/upload-laion-400m.yaml` |

bfb ships no dataset catalog of its own — these rows are collected from the
dataset `source:` blocks in `examples/*.yaml` and the repo-root `bench_*.yaml`
configs, which are the places to copy working definitions from.

### Sharded corpora (`parts:`) and streaming

npy/parquet sources published as numbered parts are read as one global row
space (`{i}` substituted per part; ids stay global):

```yaml
source:
  type: dataset
  name: laion-400m-img-emb
  format: npy
  parts: { count: 410, start: 0, path: "laion/img_emb_{i}.npy", link: "https://…/img_emb_{i}.npy" }
                          # NB: quote {i} templates inside flow mappings — bare {i}
                          # breaks flow-style YAML (block style needs no quotes)
  cache: evict            # keep (default) | evict — evict streams: prefetch next part,
                          # delete passed parts (only ones bfb downloaded itself)
```

Part row counts are measured (one ranged HTTP request per part, cached in
`datasets/.parts-index/`), never configured. `--offset N` resumes an
interrupted upload (skips N rows and ids). `parts` is mutually exclusive with
`path`/`link` and only valid for npy/parquet; `cache: evict` only with `parts`.

## Search config

```yaml
collection:
  name: benchmark             # must match the uploaded collection

requests:                     # templates; ONE picked at random per batch
  - kind: dense
    using: image              # omit for the unnamed default vector
    size: 1024                # required for random source; dataset/file sources define the dimension
    datatype: float32
    source: random            # random | {type: file, path: q.fbin} | dataset (flattened, recall mode)
    filters:                  # ANDed payload conditions, fresh random values per request
      - name: color
        type: keyword
        source: { type: random, cardinality: 100 }
        match_any: null       # keyword: match any of N random values
        match_prefix: null    # keyword: prefix of N chars; needs index built with
                              # prefix: true; takes precedence over match_any
  - kind: sparse
    using: bm25               # required
    source: { type: random, vocab_size: 100000, length: 1000, distribution: zipf }
    filters: []
    idf_corpus: []            # Qdrant 1.19+: compute IDF stats over points matching these
                              # conditions (multi-tenant); needs modifier: idf on the vector
```

Integer filter conditions are one-sided — `field >= x` with x drawn from
[min, max) per request; widen the range to make them more selective. Filter
`source`/`type` reuse the payload vocabulary above.

## Measuring recall on real data

Give a request a dataset query source; bfb then reads the dataset's *query
set* (preloaded into memory at startup, off the timed path), issues each query
via an advancing cursor (wraps modulo the set size), and scores returned ids
against the ground truth. Recall = `|found ∩ expected[:limit]| / limit`,
reported on stdout under `--- Precision ---` (label says `precision@10` but it
is recall at `--search-limit`) and in `results.search.precision` ({avg, p50}).

Requirements & caveats:

- Corpus must be uploaded with `id: integer` (default) so point id == dataset
  row index. `id: uuid` silently gives recall 0.
- Make `-n` a multiple of the query-set size (dbpedia 5,000; H&M 10,000;
  cohere-wiki-1m 999) so every rep issues each query equally often and recall
  is comparable.
- Recall is only meaningful up to the GT depth: dbpedia and H&M ship 10 GT ids
  per query, so at `--search-limit 100` recall caps at 0.10 by construction
  (timings still valid). cohere-wiki-1m ships 100.
- No dataset GT? `--search-quality` re-runs each query with `exact: true` and
  scores the approximate result against it (works in any mode; forces the main
  query approximate even with `--search-exact`).

## Scroll config

```yaml
collection: { name: benchmark }
mode: scroll                  # scroll: first page per request, always from the top
                              # sequential: cursor walk, opens at a random point
                              # sample: vector-less query with sample: random
requests:
  - filters: []               # one template picked at random per request
  - filters: [ { name: color, type: keyword, source: { type: random, cardinality: 100 } } ]
```

CLI still controls `-n`, `-p`, `--search-limit` (page size),
`--search-with-payload`. Results land in `results.scroll`.

## Results

`--json out.json` writes one document per run: `config` (bfb version,
collection, counts, parallelism…) + `results.{upload,index,search,scroll}` —
only phases that ran. Search/scroll phases contain `duration_secs`,
`server_timings[]`, `full_timings[]`, `rps[]`, `qps[]` (qps counts queries,
rps requests; equal at batch 1), precomputed `server_time` / `request_time`
summaries `{min, avg, p50, p95, max}`, and `precision {avg, p50}` when
measured. Upload: `{duration_secs, num_points, points_per_sec}`; index:
`{wait_secs}`. (Top-level `server_timings`/`rps`/`full_timings` mirrors are
deprecated back-compat.) Typical extraction:

```bash
jq '.results.search | {qps: .qps, server_avg: .server_time.avg, p95: .server_time.p95, recall: .precision.avg}' out.json
```

`server_time` is Qdrant's reported per-request time; `request_time` includes
client+network. Under contention compare **avg server_time** across builds;
sanity-check saturation with `qps × request_p50 ≈ -p`.

## Recipe: real-data benchmark end to end

```bash
# 0. Serve Qdrant (build under test) on 127.0.0.1:6334 — use the IP, not
#    `localhost` (which may resolve to ::1).
# 1. Datasets dir (registry-free!):
export BFB_DATASETS_DIR=$PWD/bench_datasets

# 2. Upload once — id: integer, --create-if-missing so reruns never clobber:
./target/release/bfb upload --file bench_dbpedia_bq_upload.yaml \
  --uri http://127.0.0.1:6334 --create-if-missing -b 128 -p 8 -t 8

# 3. Warmup rep (discard), then measured reps:
./target/release/bfb search --file bench_dbpedia_bq.yaml --uri http://127.0.0.1:6334 \
  -n 5000 -p 1 -t 1 --search-limit 10 --search-hnsw-ef 100 \
  --quantization-rescore false --json rep1.json
```

The repo root has working upload/search YAML pairs to copy:
`bench_dbpedia_bq*.yaml`, `bench_hnm_bq*.yaml`, `cohere_wiki_1m_*.yaml`
(pattern: BQ `always_ram: true`, f32 originals `on_disk: true`, rescore off ⇒
pure 1-bit scoring path).

## Recipe: A/B-compare two Qdrant builds

Use `bench_dbpedia_ab.sh` (repo root) as the template — one invocation per
side against whichever build currently serves 6334:

```bash
bash bench_dbpedia_ab.sh A          # while build A serves
# restart server with build B
bash bench_dbpedia_ab.sh B
python3 bench_dbpedia_report.py     # compare
```

Its methodology, if reimplementing: cells = {light: limit 10 / ef 100, heavy:
limit 100 / ef 512} × {single: `-p 1 -t 1`, contended: `-p 32 -t 12`}; one
discarded warmup per cell, 3 reps single / 5 contended; `-n` a multiple of the
query-set size; record server version+commit, binary path+md5, collection
info (curl :6333) into a meta file per side; pull `median qps` and
`server_time.avg/p50/p95` + `precision.avg` per rep from `--json`. Override
via env: `CONFIG=`, `COLLECTION=`, `OUT_DIR=`, `EXTRA_ARGS=`, `N_*=`,
`SKIP_WARMUP=1`.

## Common mistakes

| Mistake | Reality |
|---|---|
| Runtime knobs (`uri`, `parallel`, `batch_size`, `search_limit`…) in YAML | Hard parse error. Shape in YAML, how on CLI. |
| Inventing schema keys (`hnsw_config:`, `quantization_config:`, top-level `dataset:`) | `deny_unknown_fields` rejects them. Run `bfb schema`, copy `examples/*.yaml`. |
| `bfb upload` on an existing collection without `--create-if-missing` | Deletes and recreates it. |
| `BFB_DATASETS_DIR` → dir with a foreign `datasets.json` | bfb dies parsing the registry even when it isn't needed. Registry-free dir. |
| Expecting a `report_recall`-style switch | Recall is automatic when the request `source` is `type: dataset` (and upload used `id: integer`). |
| Recall = 0 | `id: uuid` upload, wrong collection name, or corpus/query-set mismatch. |
| `match_prefix` filter fails | Keyword index must be created with `prefix: true`. |
| `idf_corpus` has no effect | Sparse vector needs `modifier: idf`. |
| Assuming rescore is on by default | bfb sends rescore=false unless `--quantization-rescore true`. |
| Comparing contended qps on old dataset-query runs | Pre-2b824ba runs are client-capped (~5650 qps); compare server_time. |
| `-n` not a multiple of the query-set size | Reps see different query mixes; recall not comparable. |
| `--uri http://localhost:6334` | May resolve to ::1; use `http://127.0.0.1:6334`. |
| Judging recall at limit > GT depth | Capped by construction (dbpedia & H&M GT@10 ⇒ ≤0.10 at limit 100). |

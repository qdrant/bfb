---
name: bfb
description: Use when benchmarking Qdrant with bfb — authoring or editing bfb YAML configs (upload / search / scroll), running upload, search, scroll or serverless benchmarks, measuring recall on real datasets (dbpedia, glove, H&M, cohere-wiki, LAION), interpreting bfb --json results, or debugging dataset download / BFB_DATASETS_DIR problems.
---

# bfb — Qdrant benchmark tool

## Overview

bfb is a Rust CLI that stress-tests **Qdrant only**, over gRPC (default
`--uri http://localhost:6334`; auth via `export QDRANT_API_KEY=...`). Build with
`cargo build --release` → `./target/release/bfb`; the README also covers
prebuilt binaries and Docker.

**Core model — the thing everyone gets wrong:** a YAML config describes only
the *shape* (collection layout, how data and queries are generated); CLI flags
control the *how* (counts, concurrency, limits, target URI). There is no `uri:`,
`batch_size:`, `parallel:` or `search_limit:` key in any YAML, and **unknown
YAML keys are hard parse errors** (every config struct is
`deny_unknown_fields`). Never invent keys from memory:

- `bfb schema` prints the complete annotated upload-config schema (types,
  defaults, allowed values) as valid YAML. Ground truth for upload YAML.
- `bfb examples` lists the built-in configs (the tracked
  [`examples/*.yaml`](../../../examples/), compiled into the binary).
  `bfb examples <name> > my.yaml` dumps one to edit; `--example <name>` runs
  one directly in place of `--file`.

## Commands

| Command | Does |
|---|---|
| `bfb upload --file cfg.yaml` (or `--example NAME`) | (Re)create the collection from the YAML shape, upload, wait for the index. **Deletes an existing collection of the same name** unless `--create-if-missing` (no-op when it exists) or `--skip-create` is passed. |
| `bfb search --file cfg.yaml` | Run search templates against an existing collection. Never creates or deletes. Collection name comes from the YAML. |
| `bfb scroll --file cfg.yaml` | Scroll / sample workload against an existing collection. |
| `bfb serverless {upload,query,list,clear}` | Collection-per-tenant mode for Qdrant Serverless (see below). |
| `bfb schema` · `bfb examples [NAME]` | Upload-schema reference · built-in config catalog. |
| `bfb [flags]` (no subcommand) | Legacy flag-driven pipeline: create + upload random/fbin data, optionally `--search` / `--scroll`. Search-only against an existing collection: `bfb --skip-setup --search …`. |

`bfb self-update` and `bfb completions <shell>` also exist (README).

## Runtime CLI flags (the "how")

Integers accept `k/M/G/T` (and `ki/Mi/Gi/Ti`) suffixes and `_` separators:
`-n 5k`, `--offset 1_000_000`. Numbers inside YAML do not.

Workload: `-n` points to upload / **queries** to run (scroll: requests; default
100k). With dataset sources, omit `-n` on upload to load the whole dataset ·
`--offset` start row/id — resumes an interrupted dataset upload, `-n` is capped
by what remains · `-m/--max-id` random upserts of ids in [offset, max_id).

Concurrency: `-p` in-flight requests, closed loop (default 2) · `--rps` fixed
request rate, open loop, replaces `-p` · `-t` worker threads (default 2) · `-c`
connections per URI (pool = c × #URIs; default 1) · `-b` upload batch size in
points (default 100) · `--search-batch-size` queries per search request
(default 1; requests issued = n / batch) · `-T/--throttle` batches or searches
per second · `--delay` ms between requests.

Search: `--search-limit` (default 10) · `--search-hnsw-ef` · `--search-exact` ·
`--search-with-payload` / `--search-with-vectors` · `--prefetch N` (wraps the
query in a prefetch stage of limit N; that stage always has rescore off and
gets the same hnsw_ef) · `--indexed-only true|false` · `--timeout s` (server-side
request timeout; the client channel deadline is s + 5).

Quantized search: bfb **always sends explicit** quantization params — rescore =
the value of `--quantization-rescore true|false`, **false when the flag is
absent**; `--quantization-oversampling` only when given. The default therefore
measures the pure quantized path; pass `--quantization-rescore true` to measure
rescoring. Spelling out `false` is still good practice so the recorded command
line states the intent.

Phases: `--skip-create` · `--create-if-missing` · `--skip-upload` ·
`--skip-wait-index` · `--skip-setup` (implies the three skips) ·
`--skip-field-indices` · `--wait-on-upsert` · `--set-payload` (payload via a
separate SetPayload request).

Reliability: each request tries every client in the pool in random order
(failing over across URIs); `--retry N` repeats that N more rounds with
`--retry-interval s` between rounds · `--ignore-errors` keeps a query phase
going after errors. Multiple `--uri` values are allowed.

Consistency: `--write-ordering`, `--read-consistency` (valid in YAML modes too).

Output: `--json path` (results document, see Results) · `--jsonl-updates` /
`--jsonl-searches` / `--jsonl-rps` (per-request time series; visualize with
qdrant/mri) · `--absolute-time true` · `--p9 N` digits in p99… ·
`--timing-threshold s` logs requests slower than s (default 0.1).

**Legacy-only flags** — rejected after a subcommand, or silently ignored by it:
`--collection-name`, `-d/--dim`, `--distance`, `--datatype`, `--quantization`,
`--quantization-in-ram`, `--memory-*`, `--shards`, `--replication-factor`,
`--write-consistency-factor`, `--shard-key`, `-k/--keywords` and the other
payload generators, `--sparse-vectors`, `--hnsw-*`, `--segments`, `--uuids`,
`--search`, `--scroll`. In YAML mode those live in the config (`name`,
`shard_number`, `replication_factor`, `sharding`, `hnsw`, …). `bfb --help`
lists them all.

## Upload config (collection shape)

Single top-level key `collection:`. Every section except at least one of
`vectors` / `sparse_vectors` is optional. Values below are examples, not
defaults — `bfb schema` has the defaults; the comments here add what it does
not spell out:

```yaml
collection:
  name: benchmark
  id: integer                 # integer (default) | uuid — integer ⇒ point id = dataset row (needed for recall)
  on_disk_payload: true
  shard_number: null
  replication_factor: 1
  write_consistency_factor: 1
  sharding: { method: custom, key: tenant-a }   # creates shard key `tenant-a`; all upserts go to it

  hnsw: { m: 16, payload_m: null, ef_construct: 100, full_scan_threshold: null,
          on_disk: false, inline_storage: false, memory: null }
  optimizers: { default_segment_number: 2, indexing_threshold: null, memmap_threshold: null,
                max_segment_size: null,            # bigger segments search faster, index slower
                deleted_threshold: null, vacuum_min_vector_number: null, prevent_unoptimized: false }

  quantization:               # collection-wide; also settable per dense vector
    type: binary              # none | scalar | binary | binary-2bit | binary-1.5bit |
                              # turbo-1bit | turbo-1.5bit | turbo-2bit | turbo-4bit |
                              # product-x4 | -x8 | -x16 | -x32 | -x64
    always_ram: true          # default false
    memory: null              # cold | cached | pinned — supersedes always_ram

  vectors:                    # at most ONE entry may omit `name`; with 2+ entries all must be named
    - name: image
      size: 1024              # required; for dataset sources it must equal the dataset dimension (bfb does not cross-check)
      distance: cosine        # cosine | dot | euclid | manhattan
      datatype: float32       # float32 | float16 | uint8 | turbo4 (Qdrant 1.19+)
      on_disk: true
      memory: null            # cold | cached (dense storage cannot be pinned)
      multivector: { comparator: max_sim, count: 4 }
      quantization: null      # same shape as collection.quantization
      source: random          # see Sources

  sparse_vectors:
    - name: bm25              # required; names unique across dense AND sparse
      datatype: float32       # float32 | float16 | uint8
      on_disk: false
      memory: null            # cold | cached | pinned (inverted index)
      modifier: none          # none | idf — idf required for BM25 scoring and for search idf_corpus
      source: { type: random, vocab_size: 100000, length: 100, distribution: zipf }

  payload:
    memory: null              # cold | cached — supersedes on_disk_payload
    source: null              # whole-payload dataset source, see below

  fields:                     # payload fields: value generation and/or index declaration
    - name: color
      type: keyword           # keyword | integer | float | bool | uuid | geo | text | datetime
      index: true             # default true; false ⇒ unindexed filler payload
      on_disk: false
      memory: null            # field-index placement
      is_tenant: false        # keyword/uuid
      is_principal: false     # integer/float/datetime
      range_index: true       # integer only
      prefix: false           # keyword only; required for search match_prefix filters
      tokenizer: null         # text only: word (default) | whitespace | prefix | multilingual
      source: { type: random, cardinality: 100 }
```

`memory:` everywhere is RAM placement (Qdrant 1.19+): `cold` load on demand,
`cached` pre-warmed disk cache (evictable), `pinned` never evicted. It
supersedes the `on_disk` / `always_ram` booleans; bfb sends both so configs
work on older servers.

### Vector sources

```yaml
source: random                                    # default
source: { type: file, path: ./vectors.fbin, strategy: random-sample }  # or from-start;
                                                  # path may be an http(s):// URL (downloaded once, cached)
source:                                           # dense dataset: fields INLINE (flattened)
  type: dataset
  name: glove-25-angular
  format: h5                                      # h5 | tar | sparse | npy | parquet | multivector
  path: glove-25-angular/glove-25-angular.hdf5    # relative to the datasets dir
  link: http://ann-benchmarks.com/glove-25-angular.hdf5
```

**Asymmetry to remember:** the dense-vector dataset source is *flattened*
(dataset fields directly in the source map). Sparse-vector, per-field payload
and whole-payload dataset sources nest them under a `dataset:` key:

```yaml
sparse_vectors:
  - name: bm25
    source:
      type: dataset
      dataset: { name: my-sparse, format: sparse, path: my-sparse/my-sparse, link: https://…tgz }
```

### Payload value generation (`fields[].source`)

Kinds: `random` (default) · `random-clusters` (geo) · `now` (datetime) ·
`dataset` (nested `dataset:` **plus** `field:` naming the dataset column). Keys
that do not apply to the field type are ignored:

| type | keys (defaults) |
|---|---|
| keyword | `cardinality` (100), `values_per_point` (1), `length_multiplier` (1); values look like `keyword_17` |
| integer | `min` / `max` (0 / 100, half-open), `values_per_point` |
| float | `min` / `max` (-1 / 1) |
| bool | `true_ratio` (0.5) |
| geo | `clusters` (10, with `random-clusters`); points near Berlin ±1° |
| text | `vocab_size`, `min_length` (16 words), `max_length`, `distribution: uniform\|zipf` |
| datetime | `now`, else a random instant within the past year (`min` / `max` are accepted but currently unused) |

### Whole-payload source + index-only fields

For datasets that ship payload objects (`tar` bundles' `payloads.jsonl`,
`parquet` rows), load the whole object per point and declare only the indexes:

```yaml
payload:
  source:
    type: dataset
    dataset: { name: laion-small-clip, format: tar, path: laion-small-clip/laion-small-clip, link: https://…tgz }
fields:
  - name: similarity          # index-only: no `source` — the value comes from payload.source
    type: float
```

Fields present in the object but not listed are uploaded, just unindexed. For a
`tar` bundle the vector source and `payload.source` name the same dataset.
Parquet sources take `columns:` (keep only), `exclude:` (drop; also skips
decoding), `fill_null:` (substitute for null/NaN/±inf; omitted ⇒ field absent).

### Validation rules that bite

At least one dense or sparse vector · at most one unnamed dense vector, and
only when it is the sole dense vector · vector/sparse names unique across both
lists · `payload.source` must be `type: dataset` and must NOT set `field` · a
per-field dataset source MUST set `field` · `type: file` local paths must exist
at parse time (http(s) OK, `s3://` rejected) · sparse `length <= vocab_size` ·
`sharding.method` only `custom`.

## Datasets (real data)

Dataset definitions use the field names of Qdrant's vector-db-benchmark
`datasets.json` catalog (`name`, `format` — `type` is accepted as an alias in
nested `dataset:` maps — `path`, `link`), spelled inline in a `source:` block.
bfb ships no catalog of its own; working blocks to copy live in
`examples/*.yaml`.

| `format` | Contents | Query set + ground truth (auto-detected) |
|---|---|---|
| `h5` | ann-benchmarks HDF5: `train` (+ `test` / `neighbors`) | `test` / `neighbors` |
| `tar` | `.tgz` of `vectors.npy` + optional `payloads.jsonl` / `tests.jsonl` | `tests.jsonl` `query` / `closest_ids` (+ per-query `conditions`) |
| `sparse` | CSR: `data.csr` (+ `queries.csr` / `results.gt`) | `queries.csr` / `results.gt` |
| `npy` | one 2-D float `.npy` — dense vectors only | — |
| `multivector` | directory of `vectors.npy` (flat sub-vectors) + `offsets.npy` — ColBERT-style; the vector needs a `multivector:` block, whose `count` is then ignored | — |
| `parquet` | payload rows only | — |

h5/tar/sparse are *bundles* (one artifact yields vectors + payloads + queries);
npy/multivector/parquet are *components* paired row by row in one config (row
*i* of each source lands on point *i*; upload stops at the smallest source,
like `zip`). Directory formats must be linked as `.tgz` / `.tar.gz`.

Downloads land in `./datasets/` relative to the cwd — override with
**`BFB_DATASETS_DIR`**. An optional `datasets.json` in that directory enables
name-only shorthand, but bfb parses the **whole** file eagerly every time it
opens a dataset and fails on any entry it does not model (vector-db-benchmark's
own `datasets.json` contains `jsonl` datasets, for instance). So never point
`BFB_DATASETS_DIR` at a vector-db-benchmark checkout: use a directory without a
registry and spell out `name` / `format` / `path` / `link` inline.

Known-good `tar` datasets, all cosine, all under
`https://storage.googleapis.com/ann-filtered-benchmark/datasets/` (`path` is
what the archive extracts to, relative to the datasets dir):

| `name` | link file · `path` | corpus | queries / GT depth | notes |
|---|---|---|---|---|
| dbpedia-openai-100K-1536-angular | `dbpedia_openai_100K.tgz` · `dbpedia-openai-100K-1536-angular/dbpedia_openai_100K` | 100,000 × 1536 | 5,000 / 10 | unfiltered |
| dbpedia-openai-1M-1536-angular | `dbpedia_openai_1M.tgz` · `dbpedia-openai-1M-1536-angular/dbpedia_openai_1M` | 975,000 × 1536 | 5,000 / 10 | unfiltered |
| cohere-wiki-1m | `cohere-wiki-1m.tgz` · `cohere-wiki-1m/cohere_wiki_1m` | 999,999 × 768 | 999 / 100 | unfiltered; 9 payload fields |
| h-and-m-2048-angular | `hnm.tgz` · `h-and-m-2048-angular/hnm` | 105,100 × 2048 | 10,000 / 25 | **every query filtered**: one keyword `match` on one of ten `*_name` fields (below); 24 payload fields in total |
| h-and-m-2048-angular-no-filters | `hnm_no_filters.tgz` · `h-and-m-2048-angular-no-filters/hnm_no_filters` | same vectors | 10,000 / 10 | unfiltered; no payloads |
| laion-small-clip | `laion-small-clip.tgz` · `laion-small-clip/laion-small-clip` | 512-d + payloads | 5,000, half carry a `range` condition on `similarity` | `examples/upload-laion-small-clip.yaml` |

For h-and-m-2048-angular, upload with `payload.source` and index these as
`type: keyword`: colour_group_name, department_name, garment_group_name,
graphical_appearance_name, index_group_name, perceived_colour_master_name,
perceived_colour_value_name, product_group_name, product_type_name,
section_name. To find the filtered fields of any other `tar` set, inspect the
`conditions` in its `tests.jsonl`.

Other shapes: `glove-25-angular` (h5, `http://ann-benchmarks.com/glove-25-angular.hdf5`,
any ann-benchmarks `.hdf5` works the same way; `examples/upload-dataset-config.yaml`)
and LAION-400M (410 npy + parquet parts, ~407M × 512; `examples/upload-laion-400m.yaml`).

### Sharded corpora (`parts:`) and streaming

npy/parquet sources published as numbered parts are read as one global row
space (`{i}` substituted per part; ids stay global):

```yaml
source:
  type: dataset
  name: laion-400m-img-emb
  format: npy
  parts: { count: 410, start: 0, path: "laion/img_emb_{i}.npy", link: "https://…/img_emb_{i}.npy" }
                          # NB: quote {i} templates inside flow mappings — a bare {i}
                          # breaks flow-style YAML (block style needs no quotes)
  cache: evict            # keep (default) | evict — evict streams: prefetch the next part,
                          # delete passed parts (only ones bfb downloaded itself)
```

Part row counts are measured (one ranged HTTP request per part, cached in
`datasets/.parts-index/`), never configured; the host must honour `Range`
requests. `--offset N` resumes an interrupted upload (skips N rows and ids).
`parts` is mutually exclusive with `path` / `link` and only valid for
npy/parquet; `cache: evict` only with `parts`.

## Search config

```yaml
collection:
  name: benchmark             # must match the uploaded collection

requests:                     # templates; ONE picked at random per request (batch)
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
        match_prefix: null    # keyword: prefix of N chars; needs an index built with
                              # prefix: true; takes precedence over match_any
  - kind: sparse
    using: bm25               # required
    source: { type: random, vocab_size: 100000, length: 1000, distribution: zipf }
    filters: []
    idf_corpus: []            # Qdrant 1.19+: compute IDF stats over points matching these
                              # conditions (multi-tenant); needs modifier: idf on the vector
```

Filter `source` / `type` reuse the payload vocabulary above, but the generated
condition depends on the type: keyword → `match` one random value (`match_any`,
`match_prefix` as above) · integer → one-sided `field >= x`, x uniform in
[min, max) — widen the range to make it more selective · float → always
`field >= 0` (`min` / `max` ignored) · bool → random with `true_ratio` · geo →
radius 1–50 km around a random point · text → full-text match of a few random
words · uuid → a fresh random UUID (matches nothing) · datetime → no condition
at all.

## Measuring recall on real data

Give a request a dataset query source. bfb loads the dataset's *query set*
into memory at startup (off the timed path), hands out queries through an
advancing cursor (wrapping modulo the set size), and scores the returned ids
against the ground truth. Reported on stdout under `--- Precision ---` (the
label always reads `precision@10`, whatever the limit) and in
`results.search.precision` (`{avg, p50}`).

Recall = `|returned ∩ GT[:k]| / k` with **k = min(--search-limit, GT depth)**:

- At `--search-limit ≤ GT depth` this is recall@limit, comparable with
  vector-db-benchmark.
- Above the GT depth the denominator stays at the GT depth, so the number
  becomes "how many of the true top-k appear anywhere in the returned `limit`"
  — inflated, not recall@limit. Keep the limit at or below the GT depth (10 for
  dbpedia and H&M-no-filters, 25 for H&M, 100 for cohere-wiki-1m) whenever the
  recall figure matters; timings stay valid either way.

Requirements and caveats:

- The corpus must be uploaded with `id: integer` (default) so point id ==
  dataset row. `id: uuid` silently gives recall 0.
- Make `-n` a multiple of the query-set size so every rep issues each query
  equally often and recall is comparable across reps.
- **Filtered query sets** (`tests.jsonl` rows with non-empty `conditions`, e.g.
  h-and-m-2048-angular, laion-small-clip): bfb applies each query's own
  conditions (`and` → must, `or` → should, over `match` / `range` / `geo`)
  because the ground truth assumes them, and ignores any `filters:` block on
  that request. The corpus must then be uploaded with `payload.source` and the
  filtered fields indexed under `fields:`, otherwise recall collapses. bfb
  prints how many queries carry conditions when it opens the set.
- Dataset without ground truth? `--search-quality` re-runs each query with
  `exact: true` and scores the approximate result against it (works in any
  mode; forces the main query approximate even with `--search-exact`).

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

The CLI still controls `-n`, `-p`, `--search-limit` (page size),
`--search-with-payload`. Results land in `results.scroll`.

## Serverless mode

`bfb serverless` spreads traffic over `--collections-count` collections named
`<--collection-prefix>0…` (`--distribution uniform|zipf`), created lazily from
an upload YAML — only vectors and payload indexes are applied; HNSW,
quantization and on-disk knobs are ignored. A `--uri` without a port defaults
to 443; auth via `QDRANT_API_KEY`. `upload` needs `--file` / `--example`
(upload schema) and `--total-points` (falls back to `-n`); `query` takes an
optional search YAML and otherwise derives random dense/sparse queries from
the first collection's config; `list` and `clear` act on the prefix. Point ids
are contiguous across collections, so a dataset source gives every collection
a different slice. Details: README, `bfb serverless <cmd> --help`.

## Results

`--json out.json` writes one document per run: `config` (bfb version,
collection, `-n`, `-b`, `-p`, `-t`, `--rps`, config file) + `results.{upload,
index, search, scroll}` — only phases that ran. Search/scroll phases contain
`duration_secs`, `server_timings[]`, `full_timings[]`, `rps[]`, `qps[]` (qps
counts queries, rps requests; equal at batch 1), precomputed `server_time` /
`request_time` summaries `{min, avg, p50, p95, max}`, and `precision {avg,
p50}` when measured. Upload: `{duration_secs, num_points, points_per_sec}`;
index: `{wait_secs}`. Top-level `server_timings` / `rps` / `full_timings`
mirrors are deprecated back-compat. Typical extraction:

```bash
jq '.results.search | {qps: .qps, server_avg: .server_time.avg, p95: .server_time.p95, recall: .precision.avg}' out.json
```

`server_time` is Qdrant's reported per-request time; `request_time` includes
client + network. Under contention compare **avg server_time** across builds;
sanity-check saturation with `qps × request_p50 ≈ -p`.

## Recipe: real-data benchmark end to end

A complete, copyable pair for dbpedia-100K with binary quantization —
quantized vectors in RAM (`always_ram: true`), f32 originals on disk, rescore
off at search time ⇒ pure 1-bit scoring. `upload.yaml`:

```yaml
collection:
  name: dbpedia-100k-bq
  id: integer                 # point id = dataset row — required for recall
  quantization:
    type: binary
    always_ram: true
  vectors:
    - size: 1536
      distance: cosine
      on_disk: true
      source:
        type: dataset
        name: dbpedia-openai-100K-1536-angular
        format: tar
        path: dbpedia-openai-100K-1536-angular/dbpedia_openai_100K
        link: https://storage.googleapis.com/ann-filtered-benchmark/datasets/dbpedia_openai_100K.tgz
```

`search.yaml`: `collection.name: dbpedia-100k-bq` plus one `kind: dense` request
whose `source` is the identical dataset block (flattened, no `filters`) — that
makes the dataset the query source, so recall is measured.

```bash
# 1. Upload once — --create-if-missing so reruns never clobber the collection
#    (the dataset downloads into ./datasets on first use; omit -n for the full corpus):
bfb upload --file upload.yaml --create-if-missing -b 128 -p 8 -t 8

# 2. One warmup rep (discard), then measured reps.
#    -n 5k = dbpedia's 5,000-query set, once; --search-limit 10 = its GT depth:
bfb search --file search.yaml -n 5k -p 1 -t 1 --search-limit 10 --search-hnsw-ef 100 \
  --quantization-rescore false --json rep1.json
```

The built-in examples show the same flow for other shapes:
`upload-dataset-config` + `search-dataset-accuracy` (glove recall pair),
`upload-laion-small-clip` (tar with payload + filtered query set),
`upload-laion-part` / `upload-laion-400m` (npy + parquet, sharded),
`simple-hybrid` (dense + sparse).

## Common mistakes

| Mistake | Reality |
|---|---|
| Runtime knobs (`uri`, `parallel`, `batch_size`, `search_limit`…) in YAML | Hard parse error. Shape in YAML, how on the CLI. |
| Inventing schema keys (`hnsw_config:`, `quantization_config:`, top-level `dataset:`) | `deny_unknown_fields` rejects them. Run `bfb schema`, copy from `bfb examples`. |
| `bfb upload` on an existing collection without `--create-if-missing` | Deletes and recreates it. |
| Passing `--collection-name`, `--shards`, `--quantization` … to `bfb upload/search` | Legacy-only; rejected or ignored. Put them in the YAML. |
| `--indexed-only`, `--quantization-rescore`, `--absolute-time` without a value | They take an explicit `true` / `false`. |
| `BFB_DATASETS_DIR` → a directory with a foreign `datasets.json` | bfb fails parsing the registry even when it isn't needed. Use a registry-free dir. |
| Expecting a `report_recall`-style switch | Recall is automatic when the request `source` is `type: dataset` (and the upload used `id: integer`). |
| Recall = 0 | `id: uuid` upload, wrong collection name, or corpus / query-set mismatch. |
| Low recall on a filtered dataset (H&M, laion-small-clip) | The queries' own conditions are applied; the payload must be uploaded (`payload.source`) and the filtered fields indexed. |
| Reading recall at `--search-limit` > GT depth as recall@limit | Denominator is the GT depth, so the number is inflated. Keep the limit ≤ GT depth. |
| `-n` not a multiple of the query-set size | Reps see different query mixes; recall not comparable. |
| Assuming rescore is on by default | bfb sends rescore = false unless `--quantization-rescore true`. |
| `match_prefix` filter fails | The keyword index must be created with `prefix: true`. |
| `idf_corpus` has no effect | The sparse vector needs `modifier: idf`. |
| Connection refused on `http://localhost:6334` while Qdrant is up | Qdrant binds IPv4 by default and `localhost` may resolve to `::1`; use `http://127.0.0.1:6334`. |

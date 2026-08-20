# BFB

Benchmarking tool for the [Qdrant](https://github.com/qdrant/qdrant) project

## Installation

### Prebuilt binaries

Every [GitHub release](https://github.com/qdrant/bfb/releases) ships a
single-file `bfb` binary for:

| Platform              | Asset                            |
|-----------------------|----------------------------------|
| Linux x86_64 (static) | `bfb-x86_64-unknown-linux-musl`  |
| Linux aarch64 (static)| `bfb-aarch64-unknown-linux-musl` |
| macOS Apple Silicon   | `bfb-aarch64-apple-darwin`       |
| macOS Intel           | `bfb-x86_64-apple-darwin`        |

Each asset is the `bfb` executable itself. Download the one for your platform,
put it somewhere on your `PATH` and make it executable:

```bash
curl -sSfL -o ~/.local/bin/bfb https://github.com/qdrant/bfb/releases/latest/download/bfb-x86_64-unknown-linux-musl
chmod +x ~/.local/bin/bfb
bfb --version
```

The Linux binaries are statically linked against musl, so they run on any
distribution without extra dependencies. Pick a specific version by replacing
`latest/download` with `download/v<version>`; each asset has a matching
`.sha256` file.

### Upgrading

An installed binary can upgrade itself to the latest release:

```bash
bfb self-update           # install the latest release
bfb self-update --check   # only report whether a newer release exists
bfb self-update --tag v0.2.0   # install (or roll back to) a specific release
```

It downloads the asset for the current platform, verifies its checksum and
atomically replaces the running executable (re-run with `sudo` if the binary
sits somewhere you cannot write).

### Shell completions

`bfb completions <shell>` prints a completion script to stdout — redirect it to
where your shell looks for one:

```bash
# bash
mkdir -p ~/.local/share/bash-completion/completions
bfb completions bash > ~/.local/share/bash-completion/completions/bfb

# zsh (any directory on your $fpath works)
mkdir -p ~/.zsh/completions && echo 'fpath=(~/.zsh/completions $fpath)' >> ~/.zshrc
bfb completions zsh > ~/.zsh/completions/_bfb

# fish
bfb completions fish > ~/.config/fish/completions/bfb.fish
```

Open a new shell to pick them up. `elvish` and `powershell` are supported too.

### Docker

```bash
docker run --rm --network=host qdrant/bfb:dev /bfb --help
docker run --rm --network=host qdrant/bfb:dev /bfb upload --example upload-config -n 1M
```

### From source

```bash
cargo install --git https://github.com/qdrant/bfb --branch dev bfb
```

## Usage

### `upload` — YAML-driven collection shape

For collection shapes that are awkward to express as flat flags (per-vector
datatype/distance, multiple named vectors with different sources, mixed payload
types, unindexed filler payload, …), use a YAML config file:

```bash
bfb upload --file config.yaml -n 1M -b 256 -p 16 -t 8 --uri http://localhost:6334
```

The YAML file describes only the *shape* of the data (collection params + how
each field is generated). The runtime flags (`-n`, `-b`, `-p`, `-t`, `--uri`,
`--rps`, `--offset`, …) still control *how* it is uploaded. See
[`examples/upload-config.yaml`](examples/upload-config.yaml) for the full schema.

Shipped examples are compiled into the binary, so they work without a checkout
or a Docker volume:

```bash
bfb examples                                 # list names
bfb upload --example upload-config -n 1M -b 256 -p 16 -t 8 --uri http://localhost:6334
bfb examples upload-config > my.yaml         # dump one to customize
bfb upload --file my.yaml ...                # custom file still works
```

#### vector-db-benchmark datasets

Upload configs can source dense vectors, sparse vectors, and payloads from
inline dataset definitions (same fields as
[vector-db-benchmark `datasets.json`](https://github.com/qdrant/vector-db-benchmark/blob/master/datasets/datasets.json)).

| `format` | Contents |
|----------|----------|
| `h5` | ann-benchmarks HDF5 bundle — `train`, optional `test`/`neighbors`. Pure-Rust reader, no system libraries |
| `tar` | `.tgz` of `vectors.npy` + optional `payloads.jsonl` / `tests.jsonl` |
| `sparse` | CSR matrices (`data.csr`, optional `queries.csr` / `results.gt`) |
| `npy` | One 2-D float `.npy` — dense vectors only |
| `parquet` | One parquet file — payload rows only |

The first three are *bundles*: vectors, payloads, and queries all come out of a
single artifact. `npy` and `parquet` are *components*, so a config pairs them —
one source per slot, row *i* of each landing on point *i*:

```yaml
collection:
  vectors:
    - size: 512
      source: { type: dataset, name: emb, format: npy, path: emb.npy }
  payload:
    source:
      type: dataset
      dataset: { name: meta, format: parquet, path: meta.parquet, exclude: [exif] }
```

Parquet sources accept three extra keys: `columns` (keep only these), `exclude`
(drop these), and `fill_null` (a value substituted for nulls and for NaN/±inf
floats, which have no JSON form — by default such fields are simply absent).
See [`examples/upload-laion-part.yaml`](examples/upload-laion-part.yaml).

#### Sharded datasets

Corpora published as numbered parts are read as one row space with a `parts:`
block, so point ids stay global across the whole set. `npy` and `parquet`
sources support it; `{i}` is substituted with each part's number:

```yaml
source:
  type: dataset
  name: laion-400m-img-emb
  format: npy
  parts:
    count: 410                 # parts 0..409; `start:` moves the first index
    path: laion/img_emb_{i}.npy
    link: https://deploy.laion.ai/.../img_emb_{i}.npy
```

Part row counts are **measured, never configured**. Both formats keep their
shape at a known end of the file — the `.npy` header at the front, the parquet
footer at the back — so bfb sizes every part with one ranged HTTP request each
and downloads none of them (820 LAION parts in ~1.5 minutes). The result is
cached in `datasets/.parts-index/<name>.json`, keyed on the parts spec, so later
runs issue no requests at all.

There is deliberately no "rows per part" setting. LAION-400M turns out to have
seven distinct part sizes — 404 parts of 1,000,448 rows, one of 1,000,501, and
five short ones (parts 8, 107, 220, 319 and 409, from 189,159 to 642,675 rows)
— so a fixed guess would go wrong at part 8 and silently pair payloads with the
wrong vectors across ~98% of the corpus. The host must support ranged requests;
one that answers `200` to a `Range:` request is reported rather than silently
downloaded.

Because a point's id *is* its dataset row, `--offset` resumes an interrupted
upload — it skips that many rows as well as ids, and `-n` is capped by what
remains. See [`examples/upload-laion-400m.yaml`](examples/upload-laion-400m.yaml)
for the full 410-part, ~409.7M-point corpus.

##### Streaming a corpus larger than the disk

Parts are downloaded as they are reached, and by default they accumulate.
`cache: evict` streams instead — the next part is fetched in the background
while the current one uploads, and parts already passed are deleted:

```yaml
source:
  type: dataset
  name: laion-400m-img-emb
  format: npy
  parts: { count: 410, path: laion/img_emb_{i}.npy, link: "https://…/img_emb_{i}.npy" }
  cache: evict               # keep | evict (default: keep)
```

That holds peak disk to the few parts in flight (~4 GB for LAION) instead of
the ~600 GB the whole corpus occupies. Eviction only ever removes parts bfb
downloaded itself — a file staged in the datasets dir by hand is never deleted,
and a part still being read is left until nothing references it.

Use `format` for the dataset storage type in upload configs (`type` is reserved
for the source kind). An optional local `datasets/datasets.json` registry is
still supported for name-only shorthand.

Datasets are downloaded on first use into `./datasets/` (override with
`BFB_DATASETS_DIR`). Omit `-n` to upload the full dataset; when multiple dataset
sources are configured, upload stops at the smallest source size (like Python's
`zip`).

```bash
bfb upload --example upload-dataset-config -b 256 -p 16 --uri http://localhost:6334
```

Example vector source in YAML:

```yaml
source:
  type: dataset
  name: glove-25-angular
  format: h5
  path: glove-25-angular/glove-25-angular.hdf5
  link: http://ann-benchmarks.com/glove-25-angular.hdf5
```

See [`examples/upload-dataset-config.yaml`](examples/upload-dataset-config.yaml).

### `search` — YAML-driven search requests

For search workloads that mirror a YAML-uploaded collection (named vectors,
sparse vectors, payload filters with the same field names), use a search config
file:

```bash
bfb upload --example upload-config -n 1M --uri http://localhost:6334
bfb search --example search-config -n 50k -p 8 --uri http://localhost:6334
```

The YAML file describes only the *shape* of search requests (which vectors to
query, optional payload filters). The runtime flags (`-n`, `--parallel`,
`--search-batch-size`, `--uri`, `--rps`, …) still control *how* the benchmark
runs. See [`examples/search-config.yaml`](examples/search-config.yaml).

Each entry under `requests:` is a search template; one is picked at random per
batch. Supported kinds:

| `kind` | Fields | Notes |
|--------|--------|-------|
| `dense` | `size`, optional `using`, `source`, `filters` | Query a dense vector; `source: { type: dataset }` measures recall (see below) |
| `sparse` | `using`, `source`, `filters` | Query a named sparse vector; `source: { type: dataset }` measures recall (see below) |

Filter entries reuse the same payload `type` / `source` vocabulary as the
upload config.

#### Measuring accuracy against a reference dataset

A `dense` or `sparse` request can draw its queries from a vector-db-benchmark
dataset instead of generating random vectors, using `source: { type: dataset }`.
When a dataset query source is used, bfb reads the dataset's *query set* and,
for each query, compares the returned point ids against the dataset's
ground-truth nearest neighbors. Recall (`|found ∩ expected[:k]| / k`, the same
metric as vector-db-benchmark) is reported under `--- Precision ---`.

```bash
bfb upload --example upload-dataset-config --uri http://localhost:6334
bfb search --example search-dataset-accuracy --search-limit 10 -p 8 --uri http://localhost:6334
```

Query set + ground truth are auto-detected from the dataset files:

| `format` | Query vectors | Ground truth |
|----------|---------------|--------------|
| `h5` (ann-benchmarks) | `test` dataset | `neighbors` dataset |
| `tar` (ann-filtering-benchmark-datasets) | `tests.jsonl` `query` | `tests.jsonl` `closest_ids` |
| `sparse` | `queries.csr` | `results.gt` |

Accuracy only lines up when the corpus was uploaded with the default integer id
scheme (point id == dataset row index), which is what `bfb upload` does for
`id: integer` collections. See
[`examples/search-dataset-accuracy.yaml`](examples/search-dataset-accuracy.yaml).

### `scroll` — run a scroll workload as its own phase

The same workload as the legacy `--scroll` flag, runnable on an existing
collection so it can be timed and reported independently:

```bash
bfb scroll --example scroll-config -n 50k -p 8 --json scroll.json
```

The YAML describes the *shape* of scroll requests: how to traverse the
collection, and the payload filters to scroll by. One template is picked at
random per request, with fresh random filter values — so a single run can
compare filtered against unfiltered:

```yaml
collection:
  name: benchmark

mode: scroll             # scroll | sequential | sample

requests:
  - filters: []          # unfiltered
  - filters:
      - name: color
        type: keyword
        source: { type: random, cardinality: 100 }
```

`mode` picks the traversal, and is orthogonal to `filters`:

| mode | request |
| --- | --- |
| `scroll` | first page matching the filter; every request starts at the top (default) |
| `sequential` | cursor walk — each request resumes from the previous page. Walks open at a random point, so concurrent ones cover different stretches instead of all re-reading the first page |
| `sample` | a vector-less `query` with `sample: random` |

Filter entries use the same payload `type` / `source` vocabulary as the upload
and search configs. The CLI still controls *how* the benchmark runs (`-n`, `-p`,
`--search-limit`, `--search-with-payload`, …). Results land under
`results.scroll`. See [`examples/scroll-config.yaml`](examples/scroll-config.yaml).

The flag-driven path (`bfb --scroll --keywords 100 …`) is still available when
you prefer flat CLI flags over a YAML file.

### `schema` — print the upload-config file schema

Print an annotated YAML reference enumerating every option accepted by an
upload-config file, with its type, default, and allowed values:

```bash
bfb schema
```

The output is itself valid YAML, so it doubles as a copy-paste starting
template for your own config.

### `examples` — list or dump built-in YAML configs

The files under [`examples/`](examples/) are compiled into the binary:

```bash
bfb examples                      # names + one-line descriptions
bfb examples search-config        # print YAML to stdout
```

Use a name with `--example` on `upload` / `search` / `scroll`. `--file` still
accepts any custom YAML.

#### Searching a YAML-built collection

Use the `search` subcommand with a matching search config — no need to align
legacy `--search` flags with the upload YAML:

```bash
bfb upload --file config.yaml -n 1M --uri http://localhost:6334
bfb search --file search-config.yaml -n 50k -p 8 --uri http://localhost:6334
```

The legacy flag-driven search path (`bfb --skip-create --skip-upload --search …`)
is still available when you prefer flat CLI flags over a YAML file.

### Results document (`--json`)

`--json <path>` writes one document describing the whole run — the parameters it
ran with, plus a section per phase that actually executed. It works on every
mode (`bfb upload`, `bfb search`, and legacy flag-driven runs), so phase timings
no longer have to be scraped from stdout or timed by the calling shell.

```bash
bfb -n 1M --search --json results.json
```

```json
{
  "config": {
    "bfb_version": "0.1.1",
    "collection_name": "benchmark",
    "num_vectors": 1000000,
    "batch_size": 100,
    "parallel": 2,
    "threads": 2
  },
  "results": {
    "upload": { "duration_secs": 57.8, "num_points": 1000000, "points_per_sec": 17301.0 },
    "index":  { "wait_secs": 12.4 },
    "search": {
      "duration_secs": 30.1,
      "server_timings": [0.000095, 0.000103],
      "full_timings":   [0.001019, 0.001185],
      "rps": [989.1], "qps": [989.1],
      "server_time":  { "min": 0.000095, "avg": 0.000196, "p50": 0.00017, "p95": 0.00023, "max": 0.0124 },
      "request_time": { "min": 0.001019, "avg": 0.001885, "p50": 0.00118, "p95": 0.00137, "max": 0.0379 },
      "precision": { "avg": 0.98, "p50": 1.0 }
    }
  }
}
```

Phases that did not run are omitted: `bfb search --example …` yields only
`results.search`, and `precision` appears only when accuracy was measured
(`--search-quality`, or a dataset query source with ground truth). Timings are
in seconds. The per-request `--jsonl-*` time series are unchanged.

**Backward compatibility.** The three arrays `--json` used to emit at the top
level — `server_timings`, `rps`, `full_timings` — are still written there, so
existing `jq '.rps'` consumers keep working. As before, they mirror the last
query phase that ran (scroll takes precedence over search), and they are absent
when no query phase ran. They are **deprecated**: prefer `results.search` /
`results.scroll`, which also expose `qps`, per-phase `duration_secs`, and the
precomputed `server_time` / `request_time` summaries.

### Legacy flag-driven mode

The main command runs without any subcommands, just options:

```bash
Usage: bfb [OPTIONS]

Options:
      --uri <URI>
          Qdrant service URI [default: http://localhost:6334]
      --fbin <FBIN>
          Source of data to upload - fbin file. Random if not specified
  -n, --num-vectors <NUM_VECTORS>
          Number of points to upload [default: 100_000]
      --vectors-per-point <VECTORS_PER_POINT>
          Number of named dense vectors per point [default: 1]
  -o, --offset <OFFSET>
          [default: 0]
  -m, --max-id <MAX_ID>
          If set, will randomly upsert/override vector ids within range [offset, max_id)
  -d, --dim <DIM>
          Number of dimensions in each dense vector or max dimension for sparse vectors [default: 128]
  -t, --threads <THREADS>
          Number of worker threads to use [default: 2]
  -p, --parallel <PARALLEL>
          Number of parallel requests to send (ignored when --rps is set) [default: 2]
      --rps <RATE>
          Target requests per second. When set, requests are sent at a fixed rate regardless of how many are currently in-flight (simulates real user traffic). This overrides --parallel for concurrency control
  -c, --connections <CONNECTIONS>
          Number of connections to open from the client to the server [default: 1]
  -b, --batch-size <POINTS>
          Batch size for updates, in number of points. [default=100] [default: 100]
      --search-batch-size <QUERIES>
          Batch size for searches, in number of queries per batch [default: 1]
  -T, --throttle <RPS>
          Throttle updates and searches, in batches/searches per second. [default=no throttling]
      --skip-create
          Skip creating a collection
      --create-if-missing
          Create if not exists. Avoid re-creating collection
      --skip-wait-index
          Skip wait until collection is indexed after upload
      --skip-upload
          Skip uploading new points
      --skip-setup
          Skip setting up collections. Implies --skip-create --skip-upload --skip-wait-index
      --search
          Perform search
      --search-exact
          Perform search without approximation
      --prefetch <PREFETCH>
          Prefetch search
      --scroll
          Perform scroll
      --search-limit <SEARCH_LIMIT>
          Search limit [default: 10]
      --json <JSON>
          Write the benchmark results document (config + every phase) to this path
      --p9 <P9>
          Number of 9 digits to show in p99* results [default: 2]
      --collection-name <COLLECTION_NAME>
          Name of the collection to use [default: benchmark]
      --distance <DISTANCE>
          Distance function used for comparing vectors [default: Cosine]
      --datatype <DATATYPE>
          Vector datatypes (Uint8, Float16, Float32, Turbo4)
      --mmap-threshold <MMAP_THRESHOLD>
          Store vectors on disk
      --indexing-threshold <INDEXING_THRESHOLD>
          Index vectors on disk
      --segments <SEGMENTS>
          Number of segments
      --max-segment-size <MAX_SEGMENT_SIZE>
          Do not create segments larger this size (in kilobytes)
      --on-disk-payload <ON_DISK_PAYLOAD>
          On disk payload [default: true] [possible values: true, false]
      --on-disk-payload-index
          On disk payload
      --on-disk-index <ON_DISK_INDEX>
          On disk index [possible values: true, false]
      --on-disk-vectors <ON_DISK_VECTORS>
          On disk vectors [possible values: true, false]
      --memory-payload <PLACEMENT>
          Memory placement of the payload storage (supersedes --on-disk-payload) [possible values: cold, cached, pinned]
      --memory-payload-index <PLACEMENT>
          Memory placement of the payload field indices (supersedes --on-disk-payload-index) [possible values: cold, cached, pinned]
      --memory-index <PLACEMENT>
          Memory placement of the HNSW graph and sparse index (supersedes --on-disk-index) [possible values: cold, cached, pinned]
      --memory-vectors <PLACEMENT>
          Memory placement of the vector storage (supersedes --on-disk-vectors) [possible values: cold, cached, pinned]
      --memory-quantization <PLACEMENT>
          Memory placement of the quantized vectors (supersedes --quantization-in-ram) [possible values: cold, cached, pinned]
      --timing-threshold <TIMING_THRESHOLD>
          Log requests if the take longer than this [default: 0.1]
      --uuids
          Use UUIDs instead of sequential ids
      --skip-field-indices
          Skip field indices creation if payloads are not empty
  -k, --keywords <KEYWORDS>
          Use keyword payloads. Defines how many different keywords there are in the payload
      --keywords-length-multiplier <KEYWORDS_LENGTH_MULTIPLIER>
          Multiplies the length of keyword payload values by a given factor. Can be used to test larger keyword payloads. Note: This must be set for both upsertions and searches (in case they're running in parallel) to prevent empty results due to different keywords being used. [default: 1]
      --max-keywords <MAX_KEYWORDS>
          Maximum number of keywords per point [default: 1]
      --float-payloads <FLOAT_PAYLOADS>
          Use float payloads [possible values: true, false]
      --match-any <MATCH_ANY>
          Match any count
      --int-payloads <INT_PAYLOADS>
          Use integer payloads
      --int-payloads-range
          Whether to enable the range index for the integer payloads
      --max-int-payloads <MAX_INT_PAYLOADS>
          Maximum number of integer payloads per point [default: 1]
      --uuid-payloads

      --bool-payloads
          Generate true/false payloads
      --geo-payloads
          Use geo payloads
      --text-payloads
          generate text-like payloads
      --text-payload-length <TEXT_PAYLOAD_LENGTH>
          Length of the text-like payloads
      --text-payload-vocabulary <TEXT_PAYLOAD_VOCABULARY>
          Vocabulary size for text-like payloads
      --timestamp-payload
          Add payload with the current timestamp to all points
      --set-payload
          Use separate request to set payload on just upserted points
      --hnsw-ef-construct <HNSW_EF_CONSTRUCT>
          `hnsw_ef_construct` parameter used during index
      --hnsw-m <HNSW_M>
          `hnsw_m` parameter used during index
      --hnsw-payload-m <HNSW_PAYLOAD_M>
          `hnsw_payload_m` parameter used during index
      --search-hnsw-ef <SEARCH_HNSW_EF>
          `hnsw_ef` parameter used during search
      --search-with-payload
          Whether to request payload in search results
      --search-with-vectors
          Whether to request vectors in search results
      --wait-on-upsert
          Wait on upsert
      --replication-factor <REPLICATION_FACTOR>
          Replication factor [default: 1]
      --shards <SHARDS>
          Number of shards in the collection
      --write-consistency-factor <WRITE_CONSISTENCY_FACTOR>
          Write consistency factor to use for collection creation [default: 1]
      --write-ordering <WRITE_ORDERING>
          Write ordering parameter to use for all write requests
      --read-consistency <READ_CONSISTENCY>
          Read consistency parameter to use for all read requests
      --timeout <TIMEOUT>
          Timeout for requests in seconds
      --retry <RETRIES>
          Number of retries for each URI on error, 0 for no retries [default: 0]
      --retry-interval <SECONDS>
          Number of seconds between each retry [default: 0]
      --ignore-errors
          Keep going on search error
      --quantization <QUANTIZATION>
          [possible values: none, binary, binary2bit, binary1p5bit, turbo1bit, turbo1p5bit, turbo2bit, turbo4bit, scalar, product-x4, product-x8, product-x16, product-x32, product-x64]
      --quantization-in-ram <QUANTIZATION_IN_RAM>
          Keep quantized vectors in memory [possible values: true, false]
      --quantization-rescore <QUANTIZATION_RESCORE>
          Enable quantization re-score during search [possible values: true, false]
      --quantization-oversampling <QUANTIZATION_OVERSAMPLING>
          Quantization oversampling factor
      --delay <DELAY>
          Delay between requests in milliseconds
      --indexed-only <INDEXED_ONLY>
          Skip un-indexed segments during search [possible values: true, false]
      --sparse-vectors <SPARSITY>
          Whether to use sparse vectors and with how much sparsity
      --sparse-vectors-per-point <SPARSE_VECTORS_PER_POINT>
          Number of named sparse vectors per point [default: 1]
      --multivector-size <MULTIVECTOR_SIZE>
          Whether to set dense vectors as multivectors
      --sparse-dim <SPARSE_DIM>
          Max dimension for sparse vectors (overrides --dim)
      --sparse-idf
          Create sparse vectors with the IDF modifier (BM25-style scoring). Required by --search-idf-corpus
      --jsonl-updates <JSONL_UPDATES>
          Path to the jsonl file to save update timings TIP: Use `qdrant/mri` to visualize the timings
      --jsonl-searches <JSONL_SEARCHES>
          Path to the jsonl file to save search timings TIP: Use `qdrant/mri` to visualize the timings
      --jsonl-rps <JSONL_RPS>
          Path to the jsonl file to save rps timings TIP: Use `qdrant/mri` to visualize the timings
      --absolute-time <ABSOLUTE_TIME>
          Use timestamp instead of relative time in jsonl Default is relative time [possible values: true, false]
      --shard-key <SHARD_KEY>
          Use custom sharding for collection and upsert points to the specified sharding key
      --tenants <TENANTS>
          Use tenant optimization for field index [possible values: true, false]
      --uuid-query <UUID_QUERY>
          Use a custom UUID as filter when searching
      --search-quality
          Bench for search quality / accurracy too
      --full-scan-threshold <FULL_SCAN_THRESHOLD>
          Set a custom full-scan threshold
      --search-idf-corpus
          Compute sparse-vector IDF statistics over the points matching the query filter instead of the whole collection. Requires a filter to be generated (e.g. `-k`), and sparse vectors created with the IDF modifier
  -h, --help
          Print help
  -V, --version
          Print version

Integers can be suffixed with k/M/G/T/ki/Mi/Gi/Ti, and underscores can be inserted to make them more readable, e.g., `--num-vectors 100k --offset 1_000_000`.
```

API KEY:

```bash
export QDRANT_API_KEY='X3CXTPlA....lLZi8y5gA'
```

or

```bash
docker run -it --rm -e QDRANT_API_KEY='X3CXTPlA....lLZi8y5gA' qdrant/bfb:dev ./bfb .....
```

### Export results in json/csv:

```bash
./bfb --json out.json ...
cat out.json | jq '[.rps, .server_timings, .full_timings] | first | @csv' >> out.csv
```

## Shell Completion

BFB supports shell completion for command-line arguments and options. The completion feature is available through a hidden `complete` command.

To install completion scripts for your shell:

```bash
# Install completion for your current shell (auto-detected)
bfb complete

# Install completion for a specific shell
bfb complete --shell bash
bfb complete --shell fish
bfb complete --shell zsh

# Print completion script to stdout (useful for manual installation)
bfb complete --print --shell bash
```

The completion script will be automatically installed to the appropriate location for your shell. Unless you request them printed to stdout.

After installation, restart your shell or source your shell configuration file to enable completions.

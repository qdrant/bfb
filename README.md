# BFB

Benchmarking tool for the [Qdrant](https://github.com/qdrant/qdrant) project

## Usage

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
          Store results to csv
      --p9 <P9>
          Number of 9 digits to show in p99* results [default: 2]
      --collection-name <COLLECTION_NAME>
          Name of the collection to use [default: benchmark]
      --distance <DISTANCE>
          Distance function used for comparing vectors [default: Cosine]
      --datatype <DATATYPE>
          Vector datatypes (Uint8, Float16, Float32)
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

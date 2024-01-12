# BFB

Benchmarking tool for the [Qdrant](https://github.com/qdrant/qdrant) project

```bash
Usage: bfb [OPTIONS]

Options:
      --uri <URI>
          Qdrant service URI [default: http://localhost:6334]
      --fbin <FBIN>
          Source of data to upload - fbin file. Random if not specified
  -n, --num-vectors <NUM_VECTORS>
          Number of points to upload [default: 100000]
      --vectors-per-point <VECTORS_PER_POINT>
          Number of named vectors per point [default: 1]
  -m, --max-id <MAX_ID>
          If set, will use vector ids within range [0, max_id) To simulate overwriting existing vectors
  -d, --dim <DIM>
          Number of dimensions in each dense vector or max dimension for sparse vectors [default: 128]
  -t, --threads <THREADS>
          Number of worker threads to use [default: 2]
  -p, --parallel <PARALLEL>
          Number of parallel requests to send [default: 2]
  -b, --batch-size <BATCH_SIZE>
          [default: 100]
  -T, --throttle <BPS>
          Throttle updates, in batches per second. [default=no throttling]
      --throttle-search <SPS>
          Throttle searches, in searches per second. [default=no throttling]
      --skip-create
          Skip creating a collection
      --create-if-missing
          Create if not exists. Avoid re-creating collection
      --skip-wait-index
          Skip wait until collection is indexed after upload
      --skip-upload
          Skip uploading new points
      --search
          Perform search
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
      --mmap-threshold <MMAP_THRESHOLD>
          Store vectors on disk
      --indexing-threshold <INDEXING_THRESHOLD>
          Index vectors on disk
      --segments <SEGMENTS>
          Number of segments
      --max-segment-size <MAX_SEGMENT_SIZE>
          Do not create segments larger this size (in kilobytes)
      --on-disk-payload
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
      --keywords <KEYWORDS>
          Use keyword payloads. Defines how many different keywords there are in the payload
      --float-payloads
          Use float payloads
      --int-payloads <INT_PAYLOADS>
          Use integer payloads
      --set-payload
          Use separate request to set payload on just upserted points
      --hnsw-ef-construct <HNSW_EF_CONSTRUCT>
          `hnsw_ef_construct` parameter used during index
      --hnsw-m <HNSW_M>
          `hnsw_m` parameter used during index
      --search-hnsw-ef <SEARCH_HNSW_EF>
          `hnsw_ef` parameter used during search
      --search-with-payload
          Whether to request payload in search results
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
      --ignore-errors
          Keep going on search error
      --quantization <QUANTIZATION>
          [possible values: none, scalar, product-x4, product-x8, product-x16, product-x32, product-x64]
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
          Number of named vectors per point [default: 1]
      --sparse-dim <SPARSE_DIM>
          Max dimension for sparse vectors (overrides --dim)
  -h, --help
          Print help
  -V, --version
          Print version
```

API KEY:

```bash
export QDRANT_API_KEY='X3CXTPlA....lLZi8y5gA'
```

or

```bash
docker run -it --rm -e QDRANT_API_KEY='X3CXTPlA....lLZi8y5gA' ./bfb .....
```

### Export results in json/csv:

```bash
./bfb --json out.json ...
cat out.json | jq '[.rps, .server_timings, .full_timings] | first | @csv' >> out.csv
```

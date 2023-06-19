# BFB

Benchmarking tool for the [Qdrant](https://github.com/qdrant/qdrant) project

```
Usage: bfb [OPTIONS]

Options:
      --uri <URI>
          Qdrant service URI [default: http://localhost:6334]
      --fbin <FBIN>
          Source of data to upload - fbin file. Random if not specified
  -n, --num-vectors <NUM_VECTORS>
          [default: 100000]
      --vectors-per-point <VECTORS_PER_POINT>
          [default: 1]
  -m, --max-id <MAX_ID>
          If set, will use vector ids within range [0, max_id) To simulate overwriting existing vectors
  -d, --dim <DIM>
          [default: 128]
  -t, --threads <THREADS>
          [default: 2]
  -p, --parallel <PARALLEL>
          Number of parallel requests to send [default: 2]
  -b, --batch-size <BATCH_SIZE>
          [default: 100]
      --skip-create
          Skip creation of the collection
      --skip-wait-index
          If set, after upload will wait until collection is indexed
      --skip-upload
          Perform data upload
      --search
          Perform search
      --search-limit <SEARCH_LIMIT>
          Search limit [default: 10]
      --collection-name <COLLECTION_NAME>
          [default: benchmark]
      --distance <DISTANCE>
          [default: Cosine]
      --mmap-threshold <MMAP_THRESHOLD>
          Store vectors on disk
      --indexing-threshold <INDEXING_THRESHOLD>
          Index vectors on disk
      --segments <SEGMENTS>
          Number of segments
      --max-segment-size <MAX_SEGMENT_SIZE>
      --on-disk-payload
          On disk payload
      --on-disk-hnsw
          On disk hnsw
      --timing-threshold <TIMING_THRESHOLD>
          Log requests if the take longer than this [default: 0.1]
      --uuids
          Use UUIDs instead of sequential ids
      --keywords <KEYWORDS>
          Use keyword payloads. Defines how many different keywords there are in the payload
      --set-payload
          Use separate request to set payload on just upserted points
      --search-hnsw-ef <SEARCH_HNSW_EF>
          `hnsw_ef` parameter used during search
      --search-with-payload
          Whether to request payload in search results
      --wait-on-upsert
          wait on upsert
      --replication-factor <REPLICATION_FACTOR>
          replication factor [default: 1]
      --shards <SHARDS>
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
          [possible values: none, scalar]
      --quantization-in-ram <QUANTIZATION_IN_RAM>
          Keep quantized vectors in memory [possible values: true, false]
      --quantization-rescore <QUANTIZATION_RESCORE>
          Enable quantization re-score during search [possible values: true, false]
  -h, --help
          Print help
  -V, --version
          Print version
```

API KEY:

```
export QDRANT_API_KEY='X3CXTPlA....lLZi8y5gA'
```

or

```
docker run -it --rm -e QDRANT_API_KEY='X3CXTPlA....lLZi8y5gA' ./bfb .....
```
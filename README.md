# BFB

Benchmarking tool for the [Qdrant](https://github.com/qdrant/qdrant) project

```
Usage: bfb [OPTIONS]

Options:
      --uri <URI>
          Qdrant service URI [default: http://localhost:6334]
  -n, --num-vectors <NUM_VECTORS>
          [default: 100000]
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
      --timing-threshold <TIMING_THRESHOLD>
          Log requests if the take longer than this [default: 0.1]
      --uuids
          Use UUIDs instead of sequential ids
      --keywords <KEYWORDS>
          Use keyword payloads. Defines how many different keywords there are in the payload
      --search-hnsw-ef <SEARCH_HNSW_EF>
          `hnsw_ef` parameter used during search
  -h, --help
          Print help information
  -V, --version
          Print version information
```

API KEY:

```
export QDRANT_API_KEY='X3CXTPlA....lLZi8y5gA'
```

or

```
docker run -it --rm -e QDRANT_API_KEY='X3CXTPlA....lLZi8y5gA' ./bfb .....
```
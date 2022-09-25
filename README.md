```
USAGE:
    bfb [OPTIONS]

OPTIONS:
    -b, --batch-size <BATCH_SIZE>
            [default: 100]

        --collection-name <COLLECTION_NAME>
            [default: benchmark]

    -d, --dim <DIM>
            [default: 128]

    -h, --help
            Print help information

    -m, --max-id <MAX_ID>
            If set, will use vector ids within range [0, max_id) To simulate overwriting existing
            vectors

    -n, --num-vectors <NUM_VECTORS>
            [default: 100000]

    -t, --threads <THREADS>
            [default: 2]

        --uri <URI>
            Qdrant service URI [default: http://localhost:6334]

    -V, --version
            Print version information

        --wait-index <wait-index>
            If set, after upload will wait until collection is indexed [default: true]
```
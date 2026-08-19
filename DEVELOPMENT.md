## Project layout

```
src/
├── main.rs          # entry point: CLI parsing, runtime setup, command dispatch
├── args/            # clap Args struct + CLI value types (consistency, ordering)
├── config/          # YAML configs: upload (collection/vector/payload), search, schema reference
├── client.rs        # Qdrant client construction + multi-client retry
├── collection/      # collection (re)creation: from CLI flags / from YAML config
├── generators/      # data generation: points (legacy & config), queries, random primitives
├── search/          # search benchmarking processors (flag-driven & config-driven)
├── upload.rs        # upload pipeline (parallelism, batching, progress)
├── upsert.rs        # upsert request execution + timing
├── scroll.rs        # scroll benchmarking processor
├── query.rs         # search/scroll entry points
├── processor.rs     # Processor trait + Timing measurement type
├── results.rs       # unified `{config, results}` document written by --json
├── stats.rs         # benchmark run loops (parallel/RPS), stats output, throttling
├── save_jsonl.rs    # timing series export
├── fbin_reader.rs   # raw .fbin vector file reader
└── dataset/         # vector-db-benchmark dataset download + format readers (h5, tar, jsonl, sparse)
```

## Multi arch docker builds:

```sh
docker buildx create --name host-builder --driver docker-container --driver-opt network=host --use
docker buildx build --network=host --platform=linux/arm64,linux/amd64 -t qdrant/bfb:local . # Build and load in Docker

# QEMU emulation support for multi-arch builds
docker run --privileged --rm tonistiigi/binfmt --install all
docker run --platform=linux/arm64 --network=host qdrant/bfb:local /bfb # run bfb
docker run --platform=linux/arm64 --network=host -it qdrant/bfb:local /bin/bash # shell
```

## Releasing binaries

`.github/workflows/release.yml` builds static Linux (`x86_64`/`aarch64`
musl, via `cross`) and macOS (`aarch64`/`x86_64`) binaries and attaches them to
a GitHub release as `bfb-<target>.tar.gz` + `.sha256`. Those names are relied on
by `bfb self-update` (`src/self_update.rs`) and the README install snippet, so
keep them stable.

To cut a release:

```sh
# 1. bump `version` in Cargo.toml (and Cargo.lock), merge to dev
# 2. tag the merged commit with a matching `v<version>` tag and push it
git tag v0.2.0
git push origin v0.2.0
```

The workflow refuses tags that do not match the Cargo.toml version, since
`self-update` compares the release tag against `CARGO_PKG_VERSION`. A tag with
a `-` (e.g. `v0.2.0-rc1`) is published as a pre-release, which
`releases/latest` — and therefore `self-update` — skips unless passed via
`--tag`.

Run the workflow manually (`workflow_dispatch`) to exercise the build matrix
without publishing; the binaries are then available as workflow artifacts.

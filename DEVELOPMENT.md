## Multi arch docker builds:

```sh
docker buildx create --name host-builder --driver docker-container --driver-opt network=host --use
docker buildx build --network=host --platform=linux/arm64,linux/amd64 -t qdrant/bfb:local . # Build and load in Docker

# QEMU emulation support for multi-arch builds
docker run --privileged --rm tonistiigi/binfmt --install all
docker run --platform=linux/arm64 --network=host qdrant/bfb:local /bfb # run bfb
docker run --platform=linux/arm64 --network=host -it qdrant/bfb:local /bin/bash # shell
```

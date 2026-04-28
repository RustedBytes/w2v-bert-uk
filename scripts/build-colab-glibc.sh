#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

image_tag="${COLAB_GLIBC_IMAGE_TAG:-rust-asr-colab-glibc:ubuntu22.04-cuda12.4}"
base_image="${COLAB_BASE_IMAGE:-nvidia/cuda:12.4.1-devel-ubuntu22.04}"
rust_toolchain="${RUST_TOOLCHAIN:-stable}"
target_dir="${COLAB_TARGET_DIR:-/workspace/target/colab-glibc}"

if ! docker info >/dev/null 2>&1; then
    echo "Docker is required and the current user must be able to access the Docker daemon." >&2
    exit 1
fi

if [ "$#" -eq 0 ]; then
    set -- build \
        --release \
        --no-default-features \
        --features burn-cuda-backend,asr-cubecl-kernels \
        --bin asr-kernel-bench
fi

docker build \
    --build-arg "BASE_IMAGE=${base_image}" \
    --build-arg "RUST_TOOLCHAIN=${rust_toolchain}" \
    -t "${image_tag}" \
    -f "${repo_root}/docker/colab-glibc/Dockerfile" \
    "${repo_root}"

docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e "HOME=/tmp" \
    -e "CARGO_TARGET_DIR=${target_dir}" \
    -v "${repo_root}:/workspace" \
    -v rust-asr-cargo-registry:/usr/local/cargo/registry \
    -v rust-asr-cargo-git:/usr/local/cargo/git \
    "${image_tag}" \
    bash -lc '
        set -euo pipefail
        echo "container glibc: $(ldd --version | sed -n "1p")"
        cargo "$@"

        bench="${CARGO_TARGET_DIR}/release/asr-kernel-bench"
        if [ -x "${bench}" ]; then
            echo "built: ${bench}"
            max_glibc="$(
                readelf --version-info "${bench}" \
                    | grep -o "GLIBC_[0-9.]*" \
                    | sort -Vu \
                    | tail -n 1
            )"
            echo "max required glibc symbol: ${max_glibc:-unknown}"
        fi
    ' bash "$@"

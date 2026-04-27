ARG BASE_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ARG RUST_TOOLCHAIN=stable

ENV DEBIAN_FRONTEND=noninteractive
ENV RUSTUP_HOME=/usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo
ENV PATH=/usr/local/cargo/bin:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        binutils \
        build-essential \
        ca-certificates \
        clang \
        cmake \
        curl \
        git \
        libssl-dev \
        pkg-config \
        protobuf-compiler \
        python3 \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain "${RUST_TOOLCHAIN}" \
    && chmod -R a+w "${RUSTUP_HOME}" "${CARGO_HOME}"

WORKDIR /workspace

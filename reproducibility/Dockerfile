# Reproducible environment for AxisForge3D Phase VII and its extension.
#
#   docker build -t axisforge-repro -f reproducibility/Dockerfile .
#   docker run --rm -it axisforge-repro bash reproducibility/reproduce_statistics.sh
#
# It contains the Rust toolchain of rust-toolchain.toml, the exact Python environment of
# reproducibility/requirements-lock.txt, and the system packages GPD needs (GPD itself is cloned and built at the
# pinned commit by reproducibility/build_gpd.sh only when asked for).
#
# No dataset, model weight or third-party mesh is baked into the image: see tools/fetch/README.md.
# The image is not built or published by this repository; building it is the user's step.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git pkg-config python3.12 python3.12-venv python3.12-dev \
        cmake libeigen3-dev libpcl-dev libopencv-dev libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Rust: the toolchain pinned in rust-toolchain.toml (rustup installs it on first use in the workspace).
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none
ENV PATH=/root/.cargo/bin:$PATH

WORKDIR /work
COPY . /work

# Python: exact versions (CPU wheels of PyTorch).
RUN python3.12 -m venv /venv \
    && /venv/bin/pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
         -r reproducibility/requirements-lock.txt \
    && /venv/bin/pip install --no-cache-dir --no-deps -e .
ENV PATH=/venv/bin:$PATH OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1

# Build the engine (pins the toolchain and every crate through Cargo.lock).
RUN cargo build --release --locked -p axisforge-cli

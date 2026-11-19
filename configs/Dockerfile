# =============================================================================
# CONTAINERFILE — dt2_v4 (reconstructed)
# =============================================================================
# This Containerfile reconstructs the (lost) original recipe that produced the
# research image "dt2_v4.sif" used for the paper.
#
# Recovery method (forensic analysis of dt2_v4.sif, 2026-11):
#   * `singularity inspect --deffile`  -> built from docker-archive dt2_v4.podman
#      (podman build -> podman save -> singularity build), Singularity 3.8.4,
#      build date 2025-06-17.
#   * /var/log/apt/history.log inside the image preserved the VERBATIM
#     `apt-get` command lines of every RUN layer, with dates.
#   * `sysconfig.get_config_var('CONFIG_ARGS')` of the baked Python binary
#     preserved the exact CPython ./configure invocation.
#   * `pip freeze` inside the image (292 packages) matches
#     requirements-frozen.txt exactly.
#
# VALIDATION (2026-11-19): the image rebuilt from this Containerfile was
# compared against the original dt2_v4.sif: pip freeze is IDENTICAL
# (292/292 package pins, zero differences). The reconstructed SIF is
# deployed alongside the original as dt2_v4_reconstructed.sif.
#
# Not recoverable (documented, functionally irrelevant):
#   * The 2023-era upstream base image flavor (an NGC license file hints at an
#     NGC lineage before image_v1.podman). The CUDA apt layers below are
#     replayed verbatim, reproducing the same dpkg state on top of the
#     official NVIDIA CUDA 12.1.1 cuDNN8 devel image.
#   * The original pip install ordering (collapsed here into the frozen list).
#
# Build (from the repository root, the build context must be the repo root):
#   podman build -f configs/Dockerfile -t dt2_v4 .
#   podman save -o dt2_v4.podman dt2_v4
#   singularity build dt2_v4.sif docker-archive://dt2_v4.podman
# =============================================================================

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# DEVIATION from the original recipe (corporate build environment): apt sources
# are forced to https because the corporate proxy accepts Basic authentication
# on CONNECT (https) tunnels but its plain-HTTP path requires NTLM.
RUN sh -c "sed -i 's|http://|https://|g' /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources 2>/dev/null || true"

# --- [2023-11-10] CUDA library pins recorded in the research image -----------
# (replayed verbatim from the image's apt history so the dpkg state matches)
RUN apt-get update -o Acquire::Retries=10 && \
    apt-get install -y --no-install-recommends -o Acquire::Retries=10 \
        gnupg2 curl ca-certificates && \
    apt-get install -y --no-install-recommends -o Acquire::Retries=10 \
        cuda-cudart-12-1=12.1.105-1 cuda-compat-12-1 && \
    apt-get install -y --no-install-recommends -o Acquire::Retries=10 \
        cuda-libraries-12-1=12.1.1-1 \
        libnpp-12-1=12.1.0.40-1 \
        cuda-nvtx-12-1=12.1.105-1 \
        libcusparse-12-1=12.1.0.106-1 \
        libcublas-12-1=12.1.3.1-1 \
        libnccl2=2.17.1-1+cuda12.1 && \
    apt-get install -y --no-install-recommends -o Acquire::Retries=10 \
        cuda-cudart-dev-12-1=12.1.105-1 \
        cuda-command-line-tools-12-1=12.1.1-1 \
        cuda-minimal-build-12-1=12.1.1-1 \
        cuda-libraries-dev-12-1=12.1.1-1 \
        cuda-nvml-dev-12-1=12.1.105-1 \
        cuda-nvprof-12-1=12.1.105-1 \
        libnpp-dev-12-1=12.1.0.40-1 \
        libcusparse-dev-12-1=12.1.0.106-1 \
        libcublas-dev-12-1=12.1.3.1-1 \
        libnccl-dev=2.17.1-1+cuda12.1 \
        cuda-nsight-compute-12-1=12.1.1-1 && \
    apt-get install -y --no-install-recommends -o Acquire::Retries=10 \
        libcudnn8=8.9.0.131-1+cuda12.1 \
        libcudnn8-dev=8.9.0.131-1+cuda12.1 && \
    rm -rf /var/lib/apt/lists/*

# --- [2024-05-07] CPython build dependencies (verbatim + documented addition) -
# DEVIATION from the original recipe: the -dev header packages below are added
# because the original build machine provided them from its (unrecoverable)
# base image; without them CPython compiles without zlib/ssl/bz2/lzma/sqlite3/
# ctypes/readline/curses/tkinter/gdbm, and `make install` (ensurepip) fails.
RUN apt-get update -o Acquire::Retries=10 && \
    apt-get install -y --no-install-recommends -o Acquire::Retries=10 \
        bzip2 ca-certificates curl g++ gcc git graphviz hwloc \
        libgdbm6 libjpeg8 libncurses5 libpng16-16 libreadline8 \
        lzip lzma numactl sqlite3 ssh tcl tk unzip uuid uuid-dev \
        vim wget xml2 xmlsec1 xz-utils zlib1g \
        zlib1g-dev libssl-dev libbz2-dev liblzma-dev libsqlite3-dev \
        libffi-dev libreadline-dev libncurses-dev libgdbm-dev tk-dev \
        tcl-dev && \
    rm -rf /var/lib/apt/lists/*

# --- [2024-05-07 13:27] CPython 3.11.9 built from source ---------------------
# (exact configure flags recovered from the baked binary)
RUN curl -fsSL https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz \
        -o /tmp/Python-3.11.9.tgz && \
    tar -xzf /tmp/Python-3.11.9.tgz -C /usr/local/src && \
    cd /usr/local/src/Python-3.11.9 && \
    ./configure --prefix=/usr/local/python \
                --disable-shared \
                --enable-optimizations \
                --with-lto \
                CFLAGS=-fno-semantic-interposition \
                LDFLAGS=-fno-semantic-interposition && \
    make -j4 && make install && \
    ln -sf /usr/local/python/bin/python3.11 /usr/local/python/bin/python && \
    ln -sf /usr/local/python/bin/pip3.11 /usr/local/python/bin/pip && \
    rm -rf /usr/local/src/Python-3.11.9 /tmp/Python-3.11.9.tgz

ENV PATH="/usr/local/python/bin:${PATH}"

# --- [2025-04-10] runtime/tooling layers (verbatim) --------------------------
RUN apt-get update -o Acquire::Retries=10 && \
    apt-get install -y --no-install-recommends -o Acquire::Retries=10 \
        python3-pip python3-tk vim wget curl nodejs npm procps git && \
    apt-get install -y -o Acquire::Retries=10 \
        texlive-latex-extra latexmk lmodern && \
    rm -rf /var/lib/apt/lists/*
# NOTE: texlive layer is required for the PyLaTeX-based report generation
#       (src/pdf/) used to produce the paper's figures and tables.

# NOTE: `make -j4` (not -j$(nproc)) is deliberate: the reference build ran on
# a shared development server, and saturating all cores would impact other
# users. Adjust if building on a dedicated machine.

# --- [2025-04/06] Python environment (292 pinned packages) -------------------
# Offline install from the pre-downloaded wheel cache (.wheels/). The cache
# lives OUTSIDE the repository, at /scr01/ntbs/spatial-support-vmb-wheels on
# harpia-dev2 (with a SHA256SUMS manifest); copy it into .wheels/ at the
# repository root before building (see docs/installation.md). This makes the
# build immune to proxy rate limiting and guarantees the exact pinned
# artifacts are used.
COPY .wheels /tmp/wheels
COPY requirements-frozen.txt /tmp/requirements-frozen.txt
RUN python -m pip install --no-index --find-links=/tmp/wheels \
    --upgrade pip==24.0 && \
    python -m pip install --no-index --find-links=/tmp/wheels wheel && \
    python -m pip install --no-index --find-links=/tmp/wheels \
    --no-build-isolation \
    -r /tmp/requirements-frozen.txt && \
    python -m pip uninstall -y wheel && \
    rm -rf /tmp/wheels /tmp/requirements-frozen.txt

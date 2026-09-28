#!/bin/sh
# Entry wrapper that applies the NVIDIA Grace / aarch64 workarounds needed to run
# the cu13 torch + MACE stack on GB10 (DGX Spark). Both were found and verified on
# a live GB10 (2026-09-28); both are aarch64-specific, so they're gated on arch and
# the x86_64 image is unaffected.
#
#  1. OMP_NUM_THREADS must be bounded. Unset, the bundled NVPL/ARM-Compute libs
#     spawn unbounded OpenMP threads at static-init on the many-core Grace CPU and
#     corrupt the allocator ("malloc(): corrupted top size") before any user code
#     runs. Any bounded value fixes it; default 8, caller-overridable.
#  2. jemalloc via LD_PRELOAD. e3nn's GPU forward pass trips a glibc-allocator
#     corruption ("free(): corrupted unsorted chunks") on aarch64; jemalloc
#     sidesteps it cleanly. Not a CUDA problem — an allocator interaction.
#
# torch cu13 wheels carry sm_120 kernels and forward-compat to GB10's sm_121, so
# no source-built torch is needed.

: "${OMP_NUM_THREADS:=8}"
export OMP_NUM_THREADS

if [ "$(uname -m)" = "aarch64" ]; then
    _jem=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
    if [ -f "$_jem" ]; then
        export LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}$_jem"
    fi
fi

exec "$@"

#!/bin/bash
# =============================================================================
# Grid search runner — parameterized version of the original gs_*.sh scripts
# =============================================================================
# Runs src/apps/grid_search_parallel.py with the same conventions used by the
# research experiments (label_id, dataset_type, dataset_resolution, ...).
#
# The MLflow tracking URI is passed explicitly; by default a LOCAL file store
# is used so the script is self-contained. Point --tracking-uri at your
# existing MLflow server to reproduce the original logging setup.
#
# Example (CPU smoke run):
#   bash scripts/run_grid_search.sh --model_arch unet --device cpu \
#        --max_epochs 1 --dataset_max_samples 5 --cv 2
#
# Example (original research configuration):
#   bash scripts/run_grid_search.sh --model_arch transunet --device cuda:all \
#        --max_epochs 110 --dataset_max_samples -1
# =============================================================================

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUNDIR="${RUNDIR:-$(mktemp -d "${TMPDIR:-/tmp}/spatial-support-vmb-gs-XXXXXX")}"
mkdir -p "$RUNDIR"

# Datasets are resolved relative to the working directory (./data/).
if [ ! -e "$RUNDIR/data" ]; then
    ln -s "$REPO/data/${DATA_SUBSET:-shallower}" "$RUNDIR/data"
fi

# Defaults follow the original research runs; overridable via env/args.
LABEL_ID="${LABEL_ID:-GS_$(date +%Y%m%d_%H%M%S)}"
MAX_EPOCHS="${MAX_EPOCHS:-110}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
SEED="${SEED:-0}"
TEST_PERC="${TEST_PERC:-0.5}"
DATASET_TYPE="${DATASET_TYPE:-shallow}"
DATASET_RESOLUTION="${DATASET_RESOLUTION:-high}"
MODEL_ARCH="${MODEL_ARCH:-transunet}"
UNEMBED_TYPE="${UNEMBED_TYPE:-paper}"
FILTER_SIZE="${FILTER_SIZE:-13}"
DEVICE="${DEVICE:-cuda:all}"
TRACKING_URI="${TRACKING_URI:-file://$RUNDIR/mlruns}"
TRACKING_NAME="${TRACKING_NAME:-deep_tomography_2.0}"

# Any extra arguments are forwarded verbatim (e.g., --cv 2).
EXTRA_ARGS=(${EXTRA_ARGS:-})

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export MPLBACKEND=Agg

echo "[grid-search] workdir : $RUNDIR"
echo "[grid-search] label   : $LABEL_ID"

cd "$RUNDIR"
PYTHONPATH="$REPO" python "$REPO/src/apps/grid_search_parallel.py" \
    --label_id="$LABEL_ID" \
    --experiment_tracking_uri="$TRACKING_URI" \
    --experiment_tracking_name="$TRACKING_NAME" \
    --dataset_max_samples="$MAX_SAMPLES" \
    --max_epochs="$MAX_EPOCHS" \
    --seed="$SEED" \
    --test_perc="$TEST_PERC" \
    --dataset_type="$DATASET_TYPE" \
    --dataset_resolution="$DATASET_RESOLUTION" \
    --model_arch="$MODEL_ARCH" \
    --unembed_type="$UNEMBED_TYPE" \
    --conv_filter_size_x="$FILTER_SIZE" \
    --conv_filter_size_z="$FILTER_SIZE" \
    --device="$DEVICE" \
    --verbose_level=1 \
    "${EXTRA_ARGS[@]}" \
    1>"$LABEL_ID.out" 2>"$LABEL_ID.err"

echo "[grid-search] done. logs: $RUNDIR/$LABEL_ID.out(.err)"

#!/bin/bash
# =============================================================================
# Refit (final training) runner — parameterized version of the original
# refit_*.sh scripts.
# =============================================================================
# After a grid search, refit retrains the BEST configuration (identified by its
# MLflow run id) with a chosen seed, logging to a dedicated refit experiment
# ("dt.<resolution>res.refit.seed.<seed>" in the original runs).
#
# Usage:
#   RUNDIR=/path/to/working/dir \
#   GS_BEST_RUN_ID=<mlflow run id of the best grid-search config> \
#   bash scripts/run_refit.sh
#
# Required inputs: a completed grid search logged to the SAME tracking URI
# (the best-run id comes from the grid-search analysis).
# =============================================================================

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUNDIR="${RUNDIR:-$(mktemp -d "${TMPDIR:-/tmp}/spatial-support-vmb-refit-XXXXXX")}"
mkdir -p "$RUNDIR"

if [ ! -e "$RUNDIR/data" ]; then
    ln -s "$REPO/data/${DATA_SUBSET:-shallower}" "$RUNDIR/data"
fi

# ---- required / research defaults -------------------------------------------
: "${GS_BEST_RUN_ID:?Set GS_BEST_RUN_ID to the MLflow run id of the best grid-search configuration}"

LABEL_ID="${LABEL_ID:-REFIT_$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-0}"
DATASET_TYPE="${DATASET_TYPE:-shallow}"
DATASET_RESOLUTION="${DATASET_RESOLUTION:-high}"
MODEL_ARCH="${MODEL_ARCH:-transunet}"
UNEMBED_TYPE="${UNEMBED_TYPE:-paper}"
FILTER_SIZE="${FILTER_SIZE:-13}"
DILATION_SIZE="${DILATION_SIZE:-1}"
ATT_NUM_LAYERS="${ATT_NUM_LAYERS:--1}"
MAX_EPOCHS="${MAX_EPOCHS:-110}"
DEVICE="${DEVICE:-cuda:all}"
TRACKING_URI="${TRACKING_URI:-file://$RUNDIR/mlruns}"
TRACKING_NAME="${TRACKING_NAME:-deep_tomography_2.0}"

# Any extra arguments are forwarded verbatim.
EXTRA_ARGS=(${EXTRA_ARGS:-})

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export MPLBACKEND=Agg

echo "[refit] workdir : $RUNDIR"
echo "[refit] label   : $LABEL_ID"
echo "[refit] best GS run id: $GS_BEST_RUN_ID"

cd "$RUNDIR"
PYTHONPATH="$REPO" python "$REPO/src/apps/refit.py" \
    --label_id="$LABEL_ID" \
    --experiment_tracking_uri="$TRACKING_URI" \
    --experiment_tracking_name="$TRACKING_NAME" \
    --gs_best_run_id="$GS_BEST_RUN_ID" \
    --gs_best_run_experiment_tracking_name="$TRACKING_NAME" \
    --dataset_max_samples="${MAX_SAMPLES:--1}" \
    --max_epochs="$MAX_EPOCHS" \
    --seed="$SEED" \
    --dataset_type="$DATASET_TYPE" \
    --dataset_resolution="$DATASET_RESOLUTION" \
    --model_arch="$MODEL_ARCH" \
    --unembed_type="$UNEMBED_TYPE" \
    --conv_filter_size_x="$FILTER_SIZE" \
    --conv_filter_size_z="$FILTER_SIZE" \
    --conv_dilation_size_x="$DILATION_SIZE" \
    --conv_dilation_size_z="$DILATION_SIZE" \
    --att_num_layers="$ATT_NUM_LAYERS" \
    --refit \
    --device="$DEVICE" \
    --verbose_level=1 \
    "${EXTRA_ARGS[@]}" \
    1>"$LABEL_ID.out" 2>"$LABEL_ID.err"

echo "[refit] done. logs: $RUNDIR/$LABEL_ID.out(.err)"

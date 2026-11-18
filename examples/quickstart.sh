#!/bin/bash
# =============================================================================
# QUICKSTART — miniature grid search on the included data subset
# =============================================================================
# Runs the real package entry point (src/apps/grid_search_parallel.py) on the
# small "shallower" data subset shipped in data/, with:
#   * no restricted research data,
#   * no MLflow server (local file-based tracking),
#   * CPU only, a few minutes on a typical workstation/server.
#
# This is a SOFTWARE smoke test: it validates that the package installs and
# runs end-to-end (data loading -> preprocessing -> training -> MLflow
# logging). It does NOT reproduce the paper's scientific results.
#
# Usage:
#   bash examples/quickstart.sh
# =============================================================================

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/spatial-support-vmb-quickstart-XXXXXX")"

echo "[quickstart] repository : $REPO"
echo "[quickstart] workdir    : $WORKDIR"

# The pipeline resolves datasets relative to the working directory (./data/).
# Point it at the shallow subset included in the repository.
ln -s "$REPO/data/shallower" "$WORKDIR/data"

# Keep the footprint small on shared machines.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export MPLBACKEND=Agg

cd "$WORKDIR"

PYTHONPATH="$REPO" python "$REPO/src/apps/grid_search_parallel.py" \
    --label_id=QUICKSTART \
    --experiment_tracking_uri="file://$WORKDIR/mlruns" \
    --experiment_tracking_name=quickstart \
    --dataset_type=shallow \
    --dataset_resolution=high \
    --model_arch=unet \
    --unembed_type=paper \
    --conv_filter_size_x=3 --conv_filter_size_z=3 \
    --max_epochs=1 \
    --dataset_max_samples=5 \
    --test_perc=0.5 \
    --cv=2 \
    --seed=0 \
    --device=cpu \
    --verbose_level=1

# --- smoke-test assertions ---------------------------------------------------
test -d "$WORKDIR/mlruns" || { echo "FAIL: MLflow runs directory not created"; exit 1; }
echo ""
echo "[quickstart] SUCCESS"
echo "[quickstart] MLflow runs recorded under: $WORKDIR/mlruns"
echo "[quickstart] workdir kept for inspection: $WORKDIR"
echo "[quickstart] (remove it with: rm -rf $WORKDIR)"

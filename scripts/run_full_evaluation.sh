#!/bin/bash
set -e

# Navigate to repo directory (portable: resolved from this script's location,
# override with REPO_ROOT if invoking from elsewhere).
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

# Set environment
export PYTHONPATH=.
export TURKICOCR_ONNX_USE_GPU=1

echo "=========================================================="
echo "Starting AIST benchmark suite..."
echo "Timestamp: $(date)"
echo "=========================================================="

PYTHON_EXEC="${PYTHON_EXEC:-python3}"
ASSET_ROOT="${ASSET_ROOT:-${TURKICOCR_ASSET_ROOT:-./assets}}"

# Build/reuse line crops for the three selected datasets.
"${PYTHON_EXEC}" scripts/prepare_pipeline_crops.py \
  --all-datasets \
  --resume \
  --providers CPUExecutionProvider

# Run the publication leaderboard, including the INT8 deployment variant.
"${PYTHON_EXEC}" scripts/run_pipeline_recognition_eval.py \
  --all-datasets \
  --config configs/eval_baselines.yaml \
  --out "${ASSET_ROOT}/outputs/pipeline_eval" \
  --gpus 0,1,2,3

echo "=========================================================="
echo "Benchmark suite finished. Packaging results..."
echo "Timestamp: $(date)"
echo "=========================================================="

# Package the final results into the lightweight AIST release zip.
"${PYTHON_EXEC}" scripts/package_results.py \
  --asset-root "${ASSET_ROOT}" \
  --repo-root . \
  --name turkicocr_aist_benchmark_suite_20260707 \
  --out "${ASSET_ROOT}/outputs/turkicocr_aist_benchmark_suite_20260707.zip"

echo "=========================================================="
echo "Successfully completed evaluation and packaged results."
echo "Timestamp: $(date)"
echo "=========================================================="

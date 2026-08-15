#!/usr/bin/env bash
set -euo pipefail

ASSET_ROOT="${ASSET_ROOT:-${TURKICOCR_ASSET_ROOT:-./assets}}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_NAME="${RUN_NAME:-turkicocr_svtrv2_b_rec_line_v1}"
CHECKPOINT="${CHECKPOINT:-}"

cd "${REPO_ROOT}"

export TMPDIR="${ASSET_ROOT}/tmp"
export PIP_CACHE_DIR="${ASSET_ROOT}/pip-cache"
export HF_HOME="${ASSET_ROOT}/huggingface"
export PYTHONPATH="${ASSET_ROOT}/openocr/repo:${REPO_ROOT}"

python scripts/check_assets.py \
  --config configs/assets.yaml \
  --asset-root "${ASSET_ROOT}" \
  --out "${ASSET_ROOT}/outputs/assets/asset_audit.json"

python scripts/check_repo_ready.py \
  --out "${ASSET_ROOT}/outputs/audit/repo_ready.json"

python scripts/build_line_only_assets.py \
  --manifest-dir "${ASSET_ROOT}/outputs/recognition/manifests" \
  --suffix line_only \
  --crop-type line

python scripts/train_svtrv2_b_rec.py \
  --config configs/train_svtrv2_b_rec_line.yaml \
  --asset-root "${ASSET_ROOT}" \
  --run-name "${RUN_NAME}" \
  --dry-run

if [[ -n "${CHECKPOINT}" ]]; then
  python scripts/run_post_training_pipeline.py \
    --asset-root "${ASSET_ROOT}" \
    --repo-root "${REPO_ROOT}" \
    --run-name "${RUN_NAME}" \
    --checkpoint "${CHECKPOINT}" \
    --skip-baselines
fi

echo "SVTRv2-B reproduce checks completed."

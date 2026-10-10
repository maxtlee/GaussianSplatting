#!/usr/bin/env bash
# Eiffel study: lighting compensation (before / during training) and the
# number of Gaussians. Each run writes metrics and a render pair; the chosen
# run also writes the 120-frame orbit GIF.
set -u
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
mkdir -p eiffel/experiments

run () {  # processed tag n appearance loss extra...
  echo "=== $2 ($1, n=$3, $4/$5) ==="
  $PY eiffel/train_eiffel.py --processed "$1" --tag "$2" --n "$3" \
      --steps "${STEPS:-2500}" --bands "${BANDS:-24}" --lr 1e-2 \
      --appearance "$4" --loss "$5" "${@:6}" \
      --out eiffel/experiments || echo "  FAILED $2"
}

# Lighting: none / pre-training normalization / during-training correction / both
run eiffel/data/processed      none   8000 none  mse
run eiffel/data/processed_norm pre    8000 none  mse
run eiffel/data/processed      during 8000 train huber
run eiffel/data/processed_norm both   8000 train huber --gif

# Gaussian count at the best-so-far setting
run eiffel/data/processed_norm both_n4000  4000 train huber
run eiffel/data/processed_norm both_n12000 12000 train huber --bands 32

echo "suite complete"

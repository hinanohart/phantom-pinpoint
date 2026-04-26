#!/usr/bin/env bash
# Reproduce every Parquet under results/ and every figure under figures/
# from a clean checkout.  Run from the repository root.
set -euo pipefail
export PYTHONHASHSEED=0

if [[ ! -d ".venv" ]]; then
    python3 -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate
pip install --upgrade pip >/dev/null
pip install -e ".[dev]" pyarrow seaborn >/dev/null

mkdir -p results figures

echo "[reproduce_all] running pytest"
pytest -q

for exp in 01_baseline 02_audience_effect 03_vague_vs_sharp 04_ablations 05_repeated_trigger \
           06_sensitivity_sweep 07_identifiability 09_location_decomposition 10_effect_sizes; do
    echo "[reproduce_all] running experiments/${exp}"
    python -m "experiments.${exp}"
done

echo "[reproduce_all] all experiments completed.  Inspect results/ and figures/."

#!/usr/bin/env bash
set -euo pipefail

# Resolve the repo root
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Logs
mkdir -p "$PROJECT_DIR/logs"

export PYTHONUNBUFFERED=1

echo "== Step 1/3: Preprocessing data for all cities =="
python "$PROJECT_DIR/scripts/scripts_preprocess/preprocess_boulder.py"
python "$PROJECT_DIR/scripts/scripts_preprocess/preprocess_dundee.py"
python "$PROJECT_DIR/scripts/scripts_preprocess/preprocess_palo_alto.py"
python "$PROJECT_DIR/scripts/scripts_preprocess/preprocess_perth.py"

echo "== Step 2/3: Training models and making predictions =="
python "$PROJECT_DIR/scripts/scripts_forecast/train_forecast.py"

echo "== Step 3/3: Computing metrics and summarizing results =="
python "$PROJECT_DIR/scripts/scripts_metrics/metrics.py"

echo "Pipeline complete!"
echo "Results saved under: $PROJECT_DIR/results/"
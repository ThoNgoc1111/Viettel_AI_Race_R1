#!/bin/sh
# Helper to run the MATLAB energy sim container with a mounted model and scenarios.
# Usage: ./tools/run_with_model.sh </full/host/path/to/ppo_best.pth> [image]

MODEL_HOST_PATH=${1:-/mnt/d/my_models/ppo_best.pth}
IMAGE=${2:-matlab-energy-sim:local}

APP_DIR="$(pwd)"
OUTPUT_DIR="$APP_DIR/output"
SCENARIOS_DIR="$APP_DIR/scenarios"

if [ ! -f "$MODEL_HOST_PATH" ]; then
  echo "Model file not found at $MODEL_HOST_PATH"
  echo "Pass the host model path as first arg, e.g. ./tools/run_with_model.sh /mnt/d/models/ppo_best.pth"
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

docker run --rm \
  -e MODEL_PATH=/app/energy_agent/models/ppo_best.pth \
  -e PYTHONPATH=/app \
  -v "$OUTPUT_DIR:/app/output" \
  -v "$SCENARIOS_DIR:/app/scenarios:ro" \
  -v "$MODEL_HOST_PATH:/app/energy_agent/models/ppo_best.pth:ro" \
  $IMAGE \
  /bin/sh -c "mkdir -p /app/mcr_cache/R2025a/main_r0/main_run_sce/simulation/scenarios && cp -r /app/scenarios/* /app/mcr_cache/R2025a/main_r0/main_run_sce/simulation/scenarios/ && ./run_main_run_scenarios.sh /opt/mcr/R2025a"

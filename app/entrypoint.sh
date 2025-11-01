#!/bin/bash
# Start the MATLAB compiled binary in the background
"$@" &
PID=$!

# Wait up to 20 seconds for MATLAB Runtime to create its cache folder
for i in {1..20}; do
  CACHE_DIR=$(find $MCR_CACHE_ROOT/R2025a -maxdepth 1 -type d -name "main_r*" | head -n1)
  if [ -n "$CACHE_DIR" ]; then
    echo "Found MATLAB Runtime cache at $CACHE_DIR"
    mkdir -p "$CACHE_DIR/main_run_sce/simulation/scenarios"
    cp /app/scenarios/*.json "$CACHE_DIR/main_run_sce/simulation/scenarios/"
    echo "Scenarios copied into runtime cache."
    break
  fi
  sleep 1
done

# Wait for the MATLAB process to finish
wait $PID

#!/bin/sh
# Wrapper for execution of deployed applications with scenario sync + output export

exe_name=$0
exe_dir=`dirname "$0"`

if [ "x$1" = "x" ]; then
  echo "Usage: $0 <deployedMCRroot> args"
else
  MCRROOT="$1"
  echo "Setting up MATLAB Runtime environment..."
  LD_LIBRARY_PATH=.:${MCRROOT}/runtime/glnxa64
  LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MCRROOT}/bin/glnxa64
  LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MCRROOT}/sys/os/glnxa64
  LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MCRROOT}/sys/opengl/lib/glnxa64
  export LD_LIBRARY_PATH

  shift 1
  # --- Sync scenarios into the latest cache folder ---
  CACHE_BASE=/app/mcr_cache/R2025a

  # Pre-create a canonical cache dir (main_r0) and copy scenarios there to reduce
  # race conditions where the compiled app creates a cache dir after the wrapper
  # has already checked for files. This makes the first-run deterministic.
  CANONICAL_DIR="$CACHE_BASE/main_r0/main_run_sce/simulation/scenarios"
  mkdir -p "$CANONICAL_DIR"
  if [ -d /app/scenarios ]; then
    cp -r /app/scenarios/* "$CANONICAL_DIR" 2>/dev/null || true
    echo "Pre-populated canonical cache dir: $CANONICAL_DIR"
  else
    echo "Warning: /app/scenarios not found, cannot pre-populate canonical cache dir"
  fi
  # Try to find an existing cache dir. If not found, wait a short time for the compiled app to create it
  CACHE_DIR=$(find "$CACHE_BASE" -maxdepth 1 -type d -name "main_r*" | sort | tail -n1)
  if [ -z "$CACHE_DIR" ]; then
    echo "No cache directory found; will wait up to 8 seconds for the runtime to create one..."
    # attempt multiple short waits to allow the cache dir to be created (race condition on first run)
    attempts=0
    while [ $attempts -lt 8 ] && [ -z "$CACHE_DIR" ]; do
      sleep 1
      CACHE_DIR=$(find "$CACHE_BASE" -maxdepth 1 -type d -name "main_r*" | sort | tail -n1)
      attempts=$((attempts + 1))
    done
  fi

  if [ -n "$CACHE_DIR" ]; then
    echo "Copying scenarios into all detected cache dirs under $CACHE_BASE"
    # Copy into every main_r* directory to avoid races where the app picks a different cache folder
    for d in $(find "$CACHE_BASE" -maxdepth 1 -type d -name "main_r*" | sort); do
      echo "  -> $d/main_run_sce/simulation/scenarios/"
      mkdir -p "$d/main_run_sce/simulation/scenarios"
      cp -r /app/scenarios/* "$d/main_run_sce/simulation/scenarios/" 2>/dev/null || true
    done
  else
    echo "Warning: no cache folder found after waiting; scenarios may not be visible to the compiled app."
    echo "Continuing anyway — the wrapper will try to copy energies.txt out of the exe dir after run."
  fi

  # --- Diagnostics: verify that scenario JSONs exist in the cache dirs ---
  echo "Verifying scenario files in cache directories..."
  expected_count=0
  if ls /app/scenarios/*.json >/dev/null 2>&1; then
    expected_files=$(ls /app/scenarios/*.json | xargs -n1 basename)
    expected_count=$(echo "$expected_files" | wc -l)
    echo "Expected scenario files (from /app/scenarios):"
    echo "$expected_files"
  else
    echo "No scenario JSONs found in /app/scenarios. Please add them to /app/scenarios and rebuild or mount them." 
  fi

  missing_any=0
  for d in $(find "$CACHE_BASE" -maxdepth 1 -type d -name "main_r*" | sort); do
    scen_dir="$d/main_run_sce/simulation/scenarios"
    echo "Contents of $scen_dir:"
    ls -la "$scen_dir" 2>/dev/null || echo "  (empty or missing)"

    if [ "$expected_count" -gt 0 ]; then
      for ef in $expected_files; do
        if [ ! -f "$scen_dir/$ef" ]; then
          echo "MISSING: $ef in $scen_dir"
          # attempt to copy the missing file again
          cp -v "/app/scenarios/$ef" "$scen_dir/" 2>/dev/null || true
          if [ ! -f "$scen_dir/$ef" ]; then
            missing_any=1
          else
            echo "Recovered: $ef copied to $scen_dir"
          fi
        fi
      done
    fi
  done

  if [ "$missing_any" -eq 1 ]; then
    echo "Warning: some scenario files are still missing in cache directories. The binary may skip those scenarios or write zero energy values." >&2
  else
    echo "All scenario files present in cache directories (or no expected files found)."
  fi

  # --- Run the binary in background and monitor cache folders to avoid race conditions ---
  mkdir -p /app/output
  # Ensure Python can import the shipped package from /app first
  export PYTHONPATH=/app:${PYTHONPATH:-}
  echo "PYTHONPATH set to: $PYTHONPATH"
  # Emit Python sys.path and attempt to locate energy_agent and its models for debugging
  if command -v python3 >/dev/null 2>&1; then
    python3 - <<'PYCODE' > /app/output/python_path_debug.txt 2>&1 || true
import sys, importlib
import json
info = {'sys_path': sys.path}
try:
    mod = importlib.import_module('energy_agent')
    info['energy_agent_file'] = getattr(mod, '__file__', None)
    try:
        m2 = importlib.import_module('energy_agent.models')
        info['energy_agent_models_file'] = getattr(m2, '__file__', None)
    except Exception as e:
        info['models_import_error'] = str(e)
except Exception as e:
    info['energy_agent_import_error'] = str(e)
print(json.dumps(info, indent=2))
PYCODE
    echo "Wrote Python import debug to /app/output/python_path_debug.txt"
  else
    echo "python3 not found; skipping Python import debug"
  fi

  echo "Starting binary and capturing stdout/stderr to /app/output/run_stdout.log"
  # Start the binary in background, capture its PID
  "${exe_dir}/main_run_scenarios" "$@" > /app/output/run_stdout.log 2>&1 &
  BIN_PID=$!

  # While binary is running, keep copying scenarios into any main_r* cache dirs that appear
  echo "Monitoring for new MCR cache dirs and copying scenarios while binary runs (PID=$BIN_PID)"
  while kill -0 "$BIN_PID" 2>/dev/null; do
    for d in $(find "$CACHE_BASE" -maxdepth 1 -type d -name "main_r*" | sort); do
      scen_dir="$d/main_run_sce/simulation/scenarios"
      if [ ! -d "$scen_dir" ]; then
        mkdir -p "$scen_dir"
      fi
      # copy any missing files (preserve existing)
      for f in /app/scenarios/*; do
        [ -f "$f" ] || continue
        fname=$(basename "$f")
        if [ ! -f "$scen_dir/$fname" ]; then
          echo "Copying missing $fname -> $scen_dir"
          cp -v "$f" "$scen_dir/" 2>/dev/null || true
        fi
      done
    done
    sleep 1
  done

  # Wait for process to fully exit and capture exit code
  wait "$BIN_PID" || true
  BIN_EXIT=$?
  echo "Binary exited with code $BIN_EXIT"

  # --- Export energies.txt ---
  if [ -f "${exe_dir}/energies.txt" ]; then
    mkdir -p /app/output
    cp "${exe_dir}/energies.txt" /app/output/energies.txt
    echo "energies.txt copied to /app/output/"
  else
    echo "Warning: energies.txt not found in ${exe_dir}"
  fi
fi
exit

#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

run_name="run-fresh-$(date -u +%Y%m%dT%H%M%S)"
run_dir="outputs/maple_idlm/$run_name"
mkdir "$run_dir"
nohup .venv/bin/python scripts/maple_idlm/launch.py \
  --run-name "$run_name" --hours 12 \
  >> "$run_dir/training.log" 2>&1 < /dev/null &
echo "Maple started (PID $!). Log: $PWD/$run_dir/training.log"

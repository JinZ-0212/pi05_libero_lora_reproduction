#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/home/linux/Documents/pi05-libero-20261118
EXP_DIR="$TASK_ROOT/experiments/pi05-object-lora-20261118"
SCRIPT_DIR="$EXP_DIR/scripts"
cd "$TASK_ROOT"
export OPENPI_DATA_HOME="$TASK_ROOT/cache/openpi"
export LIBERO_CONFIG_PATH="$TASK_ROOT/config/libero"
export PYTHONPATH="$SCRIPT_DIR:$TASK_ROOT/sources/libero${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export OMP_NUM_THREADS=4
export MALLOC_ARENA_MAX=2
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
server_pid=''
monitor_pid=''
cleanup() {
  task_code=$?
  if [ -n "$server_pid" ]; then kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; fi
  if [ -n "$monitor_pid" ]; then kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true; fi
  printf '%s\n' "$task_code" > "$EXP_DIR/runner-exit-code.txt"
  date -Is > "$EXP_DIR/end-time.txt"
}
trap cleanup EXIT
sha256sum "$EXP_DIR/protocol.json" "$SCRIPT_DIR/serve_pilot.py" "$SCRIPT_DIR/eval_pilot.py" "$SCRIPT_DIR/run_evaluation_only.sh" > "$EXP_DIR/evaluation-provenance.sha256"
for step in 0 500; do
  if [ "$step" -eq 0 ]; then condition=initial; else condition=trained; fi
  if [ -e "$EXP_DIR/evaluation/$condition" ]; then echo 'Refusing to overwrite previous evaluation'; exit 3; fi
  occupied=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)
  if [ "$occupied" -ge 1024 ]; then echo 'GPU occupied; stopping.'; exit 6; fi
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw --format=csv -l 2 > "$EXP_DIR/evaluation-$condition-gpu.csv" &
  monitor_pid=$!
  server-env311/bin/python -u "$SCRIPT_DIR/serve_pilot.py" --step "$step" > "$EXP_DIR/logs/$condition-server.log" 2>&1 &
  server_pid=$!
  ready=false
  for attempt in $(seq 1 180); do
    if curl -sf --max-time 1 http://127.0.0.1:8004/healthz >/dev/null; then ready=true; break; fi
    if ! kill -0 "$server_pid" 2>/dev/null; then echo 'Model server failed.'; exit 4; fi
    sleep 2
  done
  if [ "$ready" != true ]; then echo 'Model readiness timeout.'; exit 5; fi
  echo "EVALUATION_START $condition"
  set +e
  timeout 600 sim-env/bin/python -u "$SCRIPT_DIR/eval_pilot.py" --condition "$condition" --suite libero_object --experiment pi05-object-lora-20261118 --port 8004 > "$EXP_DIR/logs/$condition-evaluation.log" 2>&1
  eval_code=$?
  set -e
  printf '%s\n' "$eval_code" > "$EXP_DIR/evaluation-$condition-exit-code.txt"
  echo "EVALUATION_END $condition EXIT=$eval_code"
  kill "$server_pid" "$monitor_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
  server_pid=''
  monitor_pid=''
  if [ "$eval_code" -ne 0 ]; then exit "$eval_code"; fi
done
echo PILOT_COMPLETE

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
if [ -e "$EXP_DIR/start-time.txt" ]; then echo 'Refusing to overwrite prior run.'; exit 3; fi
occupied=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)
if [ "$occupied" -ge 1024 ]; then echo 'GPU occupied; stopping.'; exit 6; fi
date -Is > "$EXP_DIR/start-time.txt"
sha256sum "$EXP_DIR/protocol.json" "$EXP_DIR/dataset-manifest.json" "$SCRIPT_DIR"/*.py "$SCRIPT_DIR"/*.sh source-revisions.json > "$EXP_DIR/provenance.sha256"
cp source-revisions.json "$EXP_DIR/source-revisions.json"
cp results/server-packages.txt "$EXP_DIR/server-packages.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$EXP_DIR/hardware.csv"
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
server-env311/bin/python - <<'PY'
from pathlib import Path
import hashlib, tarfile
folder=Path('experiments/pi05-object-lora-20261118')
p=folder/'object-data.tar'
with p.open('rb') as f:
    assert hashlib.file_digest(f,'sha256').hexdigest()==(folder/'data-tar-sha256.txt').read_text().strip()
with tarfile.open(p) as archive:
    for member in archive.getmembers():
        target=(folder/member.name).resolve()
        assert target.is_relative_to((folder/'raw').resolve()), member.name
        assert member.isfile(), member.name
    archive.extractall(folder,filter='data')
print('DATA_ARCHIVE_VERIFIED_AND_EXTRACTED')
PY
JAX_PLATFORMS=cpu timeout 300 server-env311/bin/python -u "$SCRIPT_DIR/prepare_data.py" > "$EXP_DIR/logs/data-preparation.log" 2>&1
echo DATA_PREPARATION_COMPLETE
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw --format=csv -l 1 > "$EXP_DIR/training-gpu.csv" &
monitor_pid=$!
echo TRAINING_START
set +e
timeout --signal=TERM --kill-after=15s 1800 server-env311/bin/python -u "$SCRIPT_DIR/train_pilot.py" > "$EXP_DIR/logs/training.log" 2>&1
train_code=$?
set -e
printf '%s\n' "$train_code" > "$EXP_DIR/training-exit-code.txt"
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
monitor_pid=''
echo "TRAINING_END EXIT=$train_code"
if [ "$train_code" -ne 0 ]; then exit "$train_code"; fi
for step in 0 500; do
  if [ "$step" -eq 0 ]; then condition=initial; else condition=trained; fi
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

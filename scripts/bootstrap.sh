#!/usr/bin/env bash
set -euo pipefail
REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
TASK_ROOT="${PI05_WORKDIR:-$REPO_DIR/runtime}"
export PI05_WORKDIR="$TASK_ROOT"
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/'; exit 2; }
mkdir -p "$TASK_ROOT/sources" "$TASK_ROOT/cache"
export UV_CACHE_DIR="$TASK_ROOT/cache/uv"
export UV_HTTP_TIMEOUT=120
clone_pinned() {
  task_url=$1
  task_rev=$2
  task_path=$3
  if [ -e "$task_path" ]; then
    test -d "$task_path/.git" || { echo "Existing source is not a Git checkout: $task_path"; exit 3; }
    test "$(git -C "$task_path" rev-parse HEAD)" = "$task_rev" || { echo "Unexpected source revision: $task_path"; exit 3; }
  else
    git init "$task_path"
    git -C "$task_path" remote add origin "$task_url"
    git -C "$task_path" fetch --depth 1 origin "$task_rev"
    git -C "$task_path" checkout --detach FETCH_HEAD
  fi
}
clone_pinned https://github.com/Physical-Intelligence/openpi.git 215abfb217dbac7d5f1273282331b9b1866c0479 "$TASK_ROOT/openpi-pinned"
clone_pinned https://github.com/Lifelong-Robot-Learning/LIBERO.git f78abd68ee283de9f9be3c8f7e2a9ad60246e95c "$TASK_ROOT/sources/libero"
clone_pinned https://github.com/huggingface/lerobot.git 0cf864870cf29f4738d3ade893e6fd13fbd7cdb5 "$TASK_ROOT/sources/lerobot"
if [ ! -x "$TASK_ROOT/server-env311/bin/python" ]; then
  uv venv --python "${PI05_SERVER_PYTHON:-3.11.18}" "$TASK_ROOT/server-env311"
fi
uv pip install --python "$TASK_ROOT/server-env311/bin/python" -r "$REPO_DIR/environment/server-requirements.lock.txt"
uv pip install --python "$TASK_ROOT/server-env311/bin/python" --no-deps -e "$TASK_ROOT/openpi-pinned" -e "$TASK_ROOT/openpi-pinned/packages/openpi-client" -e "$TASK_ROOT/sources/lerobot"
if [ ! -x "$TASK_ROOT/sim-env/bin/python" ]; then
  uv venv --python "${PI05_SIM_PYTHON:-3.10.14}" "$TASK_ROOT/sim-env"
fi
uv pip install --python "$TASK_ROOT/sim-env/bin/python" -r "$REPO_DIR/environment/sim-requirements.txt"
uv pip install --python "$TASK_ROOT/sim-env/bin/python" --no-deps -e "$TASK_ROOT/sources/libero" -e "$TASK_ROOT/openpi-pinned/packages/openpi-client"
"$TASK_ROOT/server-env311/bin/python" "$REPO_DIR/scripts/init_workspace.py"
"$TASK_ROOT/sim-env/bin/python" - <<'PY'
from pathlib import Path
import os,yaml
task_root=Path(os.environ['PI05_WORKDIR']).resolve()
libero=task_root/'sources/libero/libero/libero'
folder=task_root/'config/libero'
folder.mkdir(parents=True,exist_ok=True)
data={'benchmark_root':str(libero),'bddl_files':str(libero/'bddl_files'),'init_states':str(libero/'init_files'),'datasets':str(task_root/'data/libero'),'assets':str(libero/'assets')}
(folder/'config.yaml').write_text(yaml.safe_dump(data))
PY
echo 'BOOTSTRAP_COMPLETE. Base weights and expert data are separate downloads.'

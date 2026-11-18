"""Bounded Linux entrypoints for the original 500 -> 2000 update protocol.

Portable orchestration is new; the historical GPU runs used records/*/scripts/.
Use --dry-run to inspect commands without starting downloads, training or inference.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

from project_paths import REPO, ROOT, experiment
from init_workspace import initialize

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('stage', choices=['prepare','train500','eval0','eval500','train2000','eval2000','verify500','verify2000'])
parser.add_argument('--dry-run', action='store_true')
args = parser.parse_args()
continuation = args.stage.endswith('2000')
phase = 'continuation2000' if continuation else 'pilot500'
exp = experiment(phase)
scripts = REPO / 'scripts' / phase
server_python = ROOT / 'server-env311/bin/python'
sim_python = ROOT / 'sim-env/bin/python'
env = os.environ.copy()
env.update(PI05_WORKDIR=str(ROOT), OPENPI_DATA_HOME=str(ROOT/'cache/openpi'),
           LIBERO_CONFIG_PATH=str(ROOT/'config/libero'), MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
           XLA_PYTHON_CLIENT_MEM_FRACTION='0.90', OMP_NUM_THREADS='4', MALLOC_ARENA_MAX='2',
           TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', WANDB_MODE='disabled')
env['PYTHONPATH'] = os.pathsep.join([str(scripts), str(REPO/'scripts'), str(ROOT/'sources/libero'), env.get('PYTHONPATH','')])

def command(python, script, *extra):
    return [str(python), '-u', str(script), *map(str, extra)]

def plan():
    if args.stage == 'prepare':
        return [command(sys.executable, REPO/'scripts/download_data.py'), command(server_python, scripts/'prepare_data.py')]
    if args.stage.startswith('train'):
        return [command(server_python, scripts/('train_continue.py' if continuation else 'train_pilot.py'))]
    if args.stage.startswith('verify'):
        return [command(server_python, scripts/'verify_export.py')]
    step = int(args.stage.removeprefix('eval'))
    return [command(server_python, scripts/'serve_pilot.py', '--step', step),
            command(sim_python, scripts/'eval_pilot.py', '--condition', 'initial' if step==0 else 'trained',
                    '--suite', 'libero_object', '--experiment', exp.name, '--port', 8004)]

commands = plan()
if args.dry_run:
    print(json.dumps({'stage':args.stage, 'runtime':str(ROOT), 'commands':commands,
                      'environment_overrides': {k:env[k] for k in ['PI05_WORKDIR','PYTHONPATH','LIBERO_CONFIG_PATH','OPENPI_DATA_HOME']}}, indent=2))
    raise SystemExit(0)
if os.name != 'posix':
    raise SystemExit('GPU/simulation execution targets Linux. Offline checks and --dry-run also support Windows.')
initialize()

def check_gpu_idle():
    response = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    memories = [int(value.strip()) for value in response.splitlines()]
    if len(memories) != 1 or memories[0] >= 1024:
        raise RuntimeError('This runner requires one idle GPU (<1024 MiB used)')

def record_launch():
    log = exp/'logs'/f'portable-{args.stage}.json'
    if log.exists():
        raise RuntimeError(f'Refusing to overwrite launch record: {log}')
    hashes={str(file.relative_to(REPO)):hashlib.sha256(file.read_bytes()).hexdigest()
            for file in [*scripts.glob('*.py'),REPO/'scripts/pipeline.py',REPO/'scripts/project_paths.py']}
    log.write_text(json.dumps({'start_utc':datetime.now(timezone.utc).isoformat(),'commands':commands,
                               'stage':args.stage,'script_sha256':hashes},indent=2))
    return log

monitor = service = None
streams = []
log = record_launch()
start = time.monotonic()
code = 1
try:
    if args.stage.startswith(('train','eval')):
        check_gpu_idle()
        gpu_stream=(exp/f'portable-{args.stage}-gpu.csv').open('w')
        streams.append(gpu_stream)
        monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,memory.used,utilization.gpu,power.draw','--format=csv','-l','1'],stdout=gpu_stream,stderr=subprocess.STDOUT)
    if args.stage == 'train2000':
        source=experiment('pilot500')
        if not (source/'checkpoints/pi05_object_lora_3090/pilot500/500').exists():
            raise RuntimeError('Step-500 full checkpoint is required; an adapter alone cannot restore AdamW state')
        if not (exp/'assets').exists():
            shutil.copytree(source/'assets',exp/'assets')
    if args.stage.startswith('eval'):
        condition='initial' if args.stage=='eval0' else 'trained'
        if (exp/'evaluation'/condition).exists():
            raise RuntimeError('Refusing to overwrite previous evaluation')
        try:
            with urllib.request.urlopen('http://127.0.0.1:8004/healthz',timeout=1):
                raise RuntimeError('Port 8004 is already occupied by a server')
        except OSError:
            pass
        service_stream=(exp/'logs'/f'portable-{args.stage}-server.log').open('w')
        streams.append(service_stream)
        service=subprocess.Popen(commands[0],cwd=ROOT,env=env,stdout=service_stream,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+360
        while True:
            if service.poll() is not None:
                raise RuntimeError('Policy server exited; inspect server log')
            try:
                with urllib.request.urlopen('http://127.0.0.1:8004/healthz',timeout=1):
                    break
            except OSError:
                if time.monotonic()>deadline:
                    raise TimeoutError('Policy server readiness deadline exceeded')
                time.sleep(2)
        with (exp/'logs'/f'portable-{args.stage}-evaluation.log').open('w') as stream:
            subprocess.run(commands[1],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=600,check=True)
    else:
        if args.stage in ['prepare','verify500','verify2000']:
            env['JAX_PLATFORMS']='cpu'
        with (exp/'logs'/f'portable-{args.stage}.log').open('w') as stream:
            for cmd in commands:
                subprocess.run(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,
                               timeout=1800 if args.stage.startswith('train') else 1800 if args.stage=='prepare' else 300,check=True)
    code=0
finally:
    for process in [service,monitor]:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    for stream in streams:
        stream.close()
    entry=json.loads(log.read_text())
    entry.update(exit_code=code,elapsed_seconds=time.monotonic()-start,end_utc=datetime.now(timezone.utc).isoformat())
    log.write_text(json.dumps(entry,indent=2))
print('STAGE_COMPLETE',args.stage)

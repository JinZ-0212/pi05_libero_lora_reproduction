"""Verify the recorded pipeline using only Python's standard library (no GPU)."""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

from project_paths import REPO

root=Path('\\\\?\\'+str(REPO.resolve())) if os.name=='nt' else REPO
records=root/'records'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]

count=0
for line in (records/'SHA256SUMS').read_text().splitlines():
    digest,name=line.split('  ',1)
    file=records/name
    assert file.is_file(), name
    assert hashlib.sha256(file.read_bytes()).hexdigest()==digest, name
    count+=1

pilot=records/'step500'
continued=records/'step2000'
for folder,start,end in [(pilot,1,500),(continued,501,2000)]:
    training=read(folder/'training-summary.json')
    assert training['status']=='completed' and not training['errors']
    assert training['completed_steps']==end
    assert training['original_parameters_unchanged']
    metrics=rows(folder/'training-metrics.jsonl')
    assert [r['step'] for r in metrics]==list(range(start,end+1))
    assert all(math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) and r['grad_norm']>0 for r in metrics)
    assert read(folder/'frozen-before-sha256.json')==read(folder/'frozen-after-sha256.json')
    before,after=read(folder/'adapters-before-sha256.json'),read(folder/'adapters-after-sha256.json')
    assert set(before)==set(after) and sum(before[k]!=after[k] for k in before)==20
    assert read(folder/'adapter-export-verification.json')['status']=='verified'
assert read(continued/'frozen-before-sha256.json')==read(pilot/'frozen-after-sha256.json')
assert read(continued/'adapters-before-sha256.json')==read(pilot/'adapters-after-sha256.json')
assert read(continued/'training-summary.json')['optimizer_counters']==[500,500]
assert read(continued/'sampler-resume.json')['verified_prior_draws']==500
data=read(pilot/'data-audit.json')
manifest=read(pilot/'dataset-manifest.json')
train=[r for r in manifest['episodes'] if r['split']=='train']
validation=[r for r in manifest['episodes'] if r['split']=='validation']
assert len(train)==40 and len(validation)==10
assert not {r['episode_index'] for r in train}&{r['episode_index'] for r in validation}
assert sum(r['length'] for r in train)==5934 and sum(r['length'] for r in validation)==1452
assert data['all_parquet_sha256_verified'] and data['episode_sets_disjoint']

evaluations={0:pilot/'evaluation/initial/libero_object',500:pilot/'evaluation/trained/libero_object',2000:continued/'evaluation/trained/libero_object'}
results={}
for step,folder in evaluations.items():
    summary=read(folder/'summary.json')
    episodes=rows(folder/'episodes.jsonl')
    assert summary['status']=='completed' and not summary['errors'] and summary['completed_episodes']==10
    assert len(episodes)==10 and all(r['status']=='completed' for r in episodes)
    assert summary['successes']==sum(r['success'] for r in episodes)==0
    assert len(list((folder/'videos').glob('*.mp4')))==10
    results[str(step)]={'successes':summary['successes'],'episodes':len(episodes)}
for name in ['initial-states.jsonl','first-observations.jsonl']:
    baseline={r['task']:r for r in rows(evaluations[0]/name)}
    for step in [500,2000]:
        assert {r['task']:r for r in rows(evaluations[step]/name)}==baseline
for folder,phase in [(pilot,'after'),(continued,'step2000')]:
    diagnostics=rows(folder/'fixed-losses.jsonl')
    means=read(folder/'training-summary.json')['fixed_loss_after']
    for split in ['train','validation']:
        values=[r['loss'] for r in diagnostics if r['phase']==phase and r['split']==split]
        assert len(values)==10 and math.isclose(statistics.mean(values),means[split],rel_tol=1e-10)

# The archived failed transport attempt is explicitly not a valid 10-episode score.
failed=read(pilot/'attempts/002-first-inference-keepalive-timeout/evaluation/initial/libero_object/summary.json')
assert failed['completed_episodes']==0 and failed['errors']

# Packaging changed root resolution only in the copied numerical execution scripts.
for phase,label in [('pilot500','step500'),('continuation2000','step2000')]:
    for file in (root/'scripts'/phase).glob('*.py'):
        source=(records/label/'scripts'/file.name).read_text(encoding='utf-8')
        expected=source.replace("ROOT = Path('/home/linux/Documents/pi05-libero-20261118')",'from project_paths import ROOT')
        expected=expected.replace("root = Path('/home/linux/Documents/pi05-libero-20261118')",'from project_paths import ROOT as root')
        assert file.read_text(encoding='utf-8')==expected, file.name
for file in (root/'scripts').rglob('*.py'):
    ast.parse(file.read_text(encoding='utf-8'),filename=str(file))
for file in (root/'scripts').rglob('*'):
    if file.is_file() and file.name != 'check_artifacts.py':
        assert b'/home/linux/' not in file.read_bytes(), str(file)
large=[str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and '.git' not in p.parts and p.stat().st_size>=100*1024*1024]
assert not large, large
print(json.dumps({'status':'verified','historical_files_sha256_checked':count,'training_updates':2000,
                  'valid_simulation_episodes':30,'results':results,'portable_numerical_scripts':'path-only edits verified',
                  'gpu_used':False,'caveat':'Portable orchestration has not been used to rerun the GPU experiments.'},indent=2))

"""Fetch a pinned, episode-disjoint Object subset; no credentials or publication."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time
import urllib.request

folder = Path(__file__).resolve().parent
raw = folder / 'raw'
raw.mkdir(exist_ok=True)
host = 'https://huggingface.co'
repo = 'physical-intelligence/libero'

def request_json(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)

info = request_json(f'{host}/api/datasets/{repo}')
revision = info['sha']
print('DATASET_REVISION', revision, flush=True)
metadata = {}
for name in ['info.json', 'tasks.jsonl', 'episodes.jsonl', 'stats.json']:
    url = f'{host}/datasets/{repo}/resolve/{revision}/meta/{name}'
    with urllib.request.urlopen(url, timeout=30) as response:
        content = response.read()
    path = raw / 'meta' / name
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(content)
    metadata[name] = hashlib.sha256(content).hexdigest()
tasks = [json.loads(line) for line in (raw / 'meta/tasks.jsonl').read_text().splitlines()]
episodes = [json.loads(line) for line in (raw / 'meta/episodes.jsonl').read_text().splitlines()]
comparison = json.loads((folder.parent / 'extended-libero/comparison.json').read_text())
object_tasks = [r['task'] for r in comparison['paired_tasks'] if r['suite'] == 'libero_object']
assert len(object_tasks) == 10
task_ids = {row['task']: row['task_index'] for row in tasks}
selected = []
for task in object_tasks:
    candidates = [e for e in episodes if e['tasks'] == [task]]
    candidates.sort(key=lambda e: e['episode_index'])
    random.Random(42 + task_ids[task]).shuffle(candidates)
    assert len(candidates) >= 5, (task, len(candidates))
    for i, episode in enumerate(candidates[:5]):
        selected.append({**episode, 'task_index': task_ids[task], 'task': task,
                         'split': 'train' if i < 4 else 'validation'})
trees = {}
for chunk in sorted({e['episode_index'] // 1000 for e in selected}):
    rows = request_json(f'{host}/api/datasets/{repo}/tree/{revision}/data/chunk-{chunk:03d}?limit=1000')
    trees.update({r['path']: r for r in rows if r['type'] == 'file'})
for episode in selected:
    index = episode['episode_index']
    path = f'data/chunk-{index//1000:03d}/episode_{index:06d}.parquet'
    row = trees[path]
    episode.update(path=path, bytes=row['size'], expected_sha256=row['lfs']['oid'])
total = sum(e['bytes'] for e in selected)
assert total < 3_000_000_000, total
manifest = {'repo_id': repo, 'revision': revision, 'selected_at_utc': datetime.now(timezone.utc).isoformat(),
            'selection': 'LIBERO-Object: deterministic random 4 training + 1 validation episodes per task; seed=42+task_index',
            'metadata_sha256': metadata, 'expected_bytes': total, 'episodes': selected}
(folder / 'dataset-manifest.json').write_text(json.dumps(manifest, indent=2))
print('SELECTED', len(selected), 'TRAIN', sum(e['split']=='train' for e in selected), 'BYTES', total, flush=True)

def fetch(episode):
    path = raw / episode['path']
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = episode['expected_sha256']
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
        return episode['episode_index'], path.stat().st_size, 'cached'
    temp = path.with_suffix('.part')
    for attempt in range(3):
        try:
            url = f"{host}/datasets/{repo}/resolve/{revision}/{episode['path']}?download=true"
            digest = hashlib.sha256()
            count = 0
            with urllib.request.urlopen(url, timeout=60) as response, temp.open('wb') as f:
                while block := response.read(1024 * 1024):
                    count += len(block)
                    if count > episode['bytes']:
                        raise ValueError('Download exceeds declared file length')
                    digest.update(block)
                    f.write(block)
            assert count == episode['bytes'], (count, episode['bytes'])
            assert digest.hexdigest() == expected
            temp.replace(path)
            return episode['episode_index'], count, 'downloaded'
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)

with ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(fetch, e) for e in selected]
    for done, future in enumerate(as_completed(futures), 1):
        print('FILE', done, '/', len(selected), *future.result(), flush=True)
print('DATASET_DOWNLOAD_COMPLETE', flush=True)

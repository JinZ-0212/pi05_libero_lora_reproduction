"""Download exactly the recorded manifest revision and verify all 50 files."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import PurePosixPath
import time
import urllib.request

from project_paths import REPO, experiment

manifest = json.loads((REPO / 'configs/dataset-manifest.json').read_text())
raw = experiment('pilot500') / 'raw'
revision = manifest['revision']
base = f"https://huggingface.co/datasets/{manifest['repo_id']}/resolve/{revision}"

def fetch(relative, expected_hash, expected_bytes=None):
    parts = PurePosixPath(relative)
    if parts.is_absolute() or '..' in parts.parts:
        raise ValueError(relative)
    destination = raw.joinpath(*parts.parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == expected_hash:
        print('VERIFIED_CACHED', relative, flush=True)
        return
    temporary = destination.with_suffix(destination.suffix + '.part')
    for attempt in range(3):
        try:
            hasher = hashlib.sha256()
            count = 0
            with urllib.request.urlopen(f'{base}/{relative}?download=true', timeout=90) as response, temporary.open('wb') as output:
                for block in iter(lambda: response.read(1024 * 1024), b''):
                    count += len(block)
                    if expected_bytes is not None and count > expected_bytes:
                        raise ValueError('Download exceeded manifest size')
                    hasher.update(block)
                    output.write(block)
            if expected_bytes is not None and count != expected_bytes:
                raise ValueError('Wrong file size')
            if hasher.hexdigest() != expected_hash:
                raise ValueError('SHA256 mismatch')
            temporary.replace(destination)
            print('VERIFIED_DOWNLOADED', relative, count, flush=True)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)

if __name__ == '__main__':
    for name, digest in manifest['metadata_sha256'].items():
        fetch(f'meta/{name}', digest)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda row: fetch(row['path'], row['expected_sha256'], row['bytes']), manifest['episodes']))
    print('DATA_READY', len(manifest['episodes']), manifest['expected_bytes'], revision)

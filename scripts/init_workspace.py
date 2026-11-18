"""Seed a fresh runtime without modifying the archived evidence or old runs."""
from datetime import datetime, timezone
import hashlib
import json
import shutil

from project_paths import REPO, ROOT, experiment

def initialize():
    for phase in ['pilot500', 'continuation2000']:
        target = experiment(phase)
        target.mkdir(parents=True, exist_ok=True)
        (target / 'logs').mkdir(exist_ok=True)
        for source, name in [(REPO / 'configs/dataset-manifest.json', 'dataset-manifest.json'),
                             (REPO / 'configs/source-revisions.json', 'source-revisions.json'),
                             (REPO / 'configs' / phase / 'protocol.json', 'protocol.json')]:
            destination = target / name
            if destination.exists():
                if destination.read_bytes() != source.read_bytes():
                    raise RuntimeError(f'Refusing to replace a different existing file: {destination}')
            else:
                shutil.copyfile(source, destination)
        metadata = target / 'portable-workspace.json'
        if not metadata.exists():
            metadata.write_text(json.dumps({
                'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'purpose': 'New execution workspace; historical results live under repository records/.',
                'runtime_root': str(ROOT), 'repository': str(REPO),
                'protocol_template_sha256': hashlib.sha256((target / 'protocol.json').read_bytes()).hexdigest(),
            }, indent=2), encoding='utf-8')
    print('WORKSPACE_READY', ROOT)

if __name__ == '__main__':
    initialize()

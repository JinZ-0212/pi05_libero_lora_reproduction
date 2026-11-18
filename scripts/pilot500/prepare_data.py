import hashlib
from io import BytesIO
import json
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

from openpi.shared import normalize
from lora_common import EXP, RAW, ASSET_ID, make_config, read_manifest, EpisodeDataset

manifest = read_manifest()
train, validation = set(), set()
states, actions = [], []
for episode in manifest['episodes']:
    path = RAW / episode['path']
    assert path.stat().st_size == episode['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == episode['expected_sha256']
    table = pq.read_table(path, columns=['state', 'actions', 'episode_index', 'frame_index', 'task_index'])
    assert table.num_rows == episode['length']
    assert set(table['episode_index'].to_pylist()) == {episode['episode_index']}
    assert table['frame_index'].to_pylist() == list(range(table.num_rows))
    assert set(table['task_index'].to_pylist()) == {episode['task_index']}
    state = np.asarray(table['state'].to_pylist(), dtype=np.float32)
    action = np.asarray(table['actions'].to_pylist(), dtype=np.float32)
    assert state.shape == (table.num_rows, 8) and action.shape == (table.num_rows, 7)
    assert np.isfinite(state).all() and np.isfinite(action).all()
    if episode['split'] == 'train':
        train.add(episode['episode_index'])
        states.append(state)
        actions.append(action)
    else:
        validation.add(episode['episode_index'])
assert len(train) == 40 and len(validation) == 10 and not train & validation
stats = {}
for key, arrays in [('state', states), ('actions', actions)]:
    values = np.concatenate(arrays)
    stats[key] = normalize.NormStats(mean=values.mean(axis=0), std=values.std(axis=0),
                                    q01=np.quantile(values, 0.01, axis=0),
                                    q99=np.quantile(values, 0.99, axis=0))
    assert np.all(stats[key].q99 - stats[key].q01 > 1e-6), (key, stats[key])
config = make_config()
normalize.save(config.assets_dirs / ASSET_ID, stats)
dataset = EpisodeDataset('train', config)
obs, targets = dataset[0]
assert targets.shape == (1, 10, 32) and np.isfinite(targets).all()
assert obs.images['base_0_rgb'].shape == (1, 224, 224, 3)
# All examples are selected by task order, not training performance.
canvas = Image.new('RGB', (256 * 4, 284 * 10), 'white')
draw = ImageDraw.Draw(canvas)
tasks = list(dict.fromkeys(e['task'] for e in manifest['episodes']))
for row, task in enumerate(tasks):
    for split_idx, split in enumerate(['train', 'validation']):
        episode = next(e for e in manifest['episodes'] if e['task'] == task and e['split'] == split)
        table = pq.read_table(RAW / episode['path'], columns=['image', 'wrist_image'])
        for view_idx, key in enumerate(['image', 'wrist_image']):
            entry = table[key][0].as_py()
            im = Image.open(BytesIO(entry['bytes'])).convert('RGB')
            col = split_idx * 2 + view_idx
            canvas.paste(im, (col * 256, row * 284 + 28))
            draw.text((col * 256 + 3, row * 284 + 5), f'{split} / task {row} / {key}', fill='black')
canvas.save(EXP / 'data-preview.png')
audit = {'status': 'validated', 'repo_id': manifest['repo_id'], 'revision': manifest['revision'],
         'train_episodes': len(train), 'validation_episodes': len(validation),
         'train_frames': sum(len(a) for a in states),
         'validation_frames': sum(e['length'] for e in manifest['episodes'] if e['split'] == 'validation'),
         'episode_sets_disjoint': True, 'all_parquet_sha256_verified': True,
         'normalization_fit': 'training frames only; each source frame counted once; exact numpy quantiles',
         'train_model_target_shape': list(targets.shape), 'data_adapter': 'official LIBERO transforms',
         'raw_gripper_action_values': np.unique(np.concatenate(actions)[:, -1]).tolist(),
         'preview_selection': 'first selected train and validation episode of each task',
         'limitations': '40 train episodes and 10 validation episodes are a pilot subset, not the full Object training set.'}
(EXP / 'data-audit.json').write_text(json.dumps(audit, indent=2))
print('DATA_PREPARED', json.dumps(audit), flush=True)

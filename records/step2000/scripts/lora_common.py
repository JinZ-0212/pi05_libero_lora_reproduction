from collections import OrderedDict
import dataclasses
from io import BytesIO
import json
from pathlib import Path

import flax.nnx as nnx
import jax
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from openpi import transforms
from openpi.models import model as model_lib
from openpi.models.pi0_config import Pi0Config
from openpi.shared import nnx_utils
from openpi.training import config as configs, optimizer, weight_loaders

ROOT = Path('/home/linux/Documents/pi05-libero-20261118')
EXP = ROOT / 'experiments/pi05-object-lora-2000-20261118'
PARENT = ROOT / 'experiments/pi05-object-lora-20261118'
RAW = PARENT / 'raw'
ASSET_ID = 'local/libero_object_small'

def make_config():
    model = Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False,
                      paligemma_variant='gemma_2b_lora', action_expert_variant='gemma_300m_lora')
    # Freeze every original weight, including vision and action projections.
    # Only the inserted LoRA matrices receive optimizer states and updates.
    return dataclasses.replace(
        configs.get_config('pi05_libero'), name='pi05_object_lora_3090', exp_name='continuation2000',
        model=model, freeze_filter=nnx.Not(nnx_utils.PathRegex('.*lora.*')),
        data=configs.LeRobotLiberoDataConfig(repo_id=ASSET_ID, extra_delta_transform=False),
        assets_base_dir=str(EXP / 'assets'), checkpoint_base_dir=str(EXP / 'checkpoints'),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(ROOT / 'assets/pi05_base/params')),
        batch_size=1, num_workers=0, num_train_steps=2000, seed=42, ema_decay=None,
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=20, peak_lr=1e-4,
                                                  decay_steps=500, decay_lr=1e-5),
        optimizer=optimizer.AdamW(clip_gradient_norm=1.0),
        log_interval=10, save_interval=500, keep_period=500,
        wandb_enabled=False, overwrite=False, resume=False,
    )

def read_manifest():
    return json.loads((EXP / 'dataset-manifest.json').read_text())

class EpisodeDataset:
    def __init__(self, split, config):
        self.episodes = [e for e in read_manifest()['episodes'] if e['split'] == split]
        self.frames = [(i, j) for i, e in enumerate(self.episodes) for j in range(e['length'])]
        self.cache = OrderedDict()
        self.horizon = config.model.action_horizon
        self.config = config.data.create(config.assets_dirs, config.model)
        self.transform = transforms.compose([
            *self.config.repack_transforms.inputs,
            *self.config.data_transforms.inputs,
            transforms.Normalize(self.config.norm_stats, use_quantiles=self.config.use_quantile_norm),
            *self.config.model_transforms.inputs,
        ])

    def __len__(self):
        return len(self.frames)

    def table(self, episode_idx):
        if episode_idx not in self.cache:
            self.cache[episode_idx] = pq.read_table(RAW / self.episodes[episode_idx]['path'])
            while len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(episode_idx)
        return self.cache[episode_idx]

    def __getitem__(self, index):
        episode_idx, frame_idx = self.frames[index]
        episode = self.episodes[episode_idx]
        table = self.table(episode_idx)
        assert table.num_rows == episode['length']
        def image(key):
            entry = table[key][frame_idx].as_py()
            assert entry['bytes'] is not None
            return np.asarray(Image.open(BytesIO(entry['bytes'])).convert('RGB'))
        actions = [table['actions'][min(frame_idx + k, table.num_rows - 1)].as_py()
                   for k in range(self.horizon)]
        raw = {'image': image('image'), 'wrist_image': image('wrist_image'),
               'state': np.asarray(table['state'][frame_idx].as_py(), dtype=np.float32),
               'actions': np.asarray(actions, dtype=np.float32), 'prompt': episode['task']}
        transformed = self.transform(raw)
        batch = jax.tree.map(lambda x: np.asarray(x)[None], transformed)
        return model_lib.Observation.from_dict(batch), batch['actions']

    def data_config(self):
        return self.config

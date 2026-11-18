import dataclasses
import functools
import hashlib
import importlib.util
import json
import logging
import sys
import time
import traceback

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.training import checkpoints, sharding
from lora_common import ROOT, EXP, make_config, EpisodeDataset

logging.basicConfig(level=logging.INFO, force=True)
config = make_config()
spec = importlib.util.spec_from_file_location('official_train', ROOT / 'openpi-pinned/scripts/train.py')
official = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = official
spec.loader.exec_module(official)
metrics_path = EXP / 'training-metrics.jsonl'
if metrics_path.exists() or (EXP / 'training-summary.json').exists():
    raise RuntimeError('Refusing to overwrite training records')

def append(name, row):
    with (EXP / name).open('a') as f:
        f.write(json.dumps(row) + '\n')

def parameter_audit(params):
    rows = []
    for path, leaf in params.flat_state():
        value = leaf.value
        rows.append({'path': '/'.join(map(str, path)), 'shape': list(value.shape),
                     'dtype': str(value.dtype), 'parameters': int(np.prod(value.shape)),
                     'trainable': 'lora' in '/'.join(map(str, path))})
    return rows

def fingerprints(params, trainable):
    result = {}
    for path, leaf in params.flat_state():
        name = '/'.join(map(str, path))
        if ('lora' in name) == trainable:
            array = np.asarray(jax.device_get(leaf.value))
            result[name] = hashlib.sha256(array.tobytes()).hexdigest()
    return result

@jax.jit
def eval_loss(state, rng, observation, actions):
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    return jnp.mean(model.compute_loss(rng, observation, actions, train=False))

def evaluate_fixed(state, datasets, phase):
    records = []
    for split, dataset in datasets.items():
        offset = 0
        for episode_index, episode in enumerate(dataset.episodes):
            if split == 'train' and episode_index % 4 != 0:
                offset += episode['length']
                continue
            frame_index = offset + episode['length'] // 2
            obs, target = dataset[frame_index]
            value = float(eval_loss(state, jax.random.key(1000 + episode['task_index']), obs, target))
            assert np.isfinite(value)
            records.append({'phase': phase, 'split': split, 'task': episode['task'],
                            'episode_index': episode['episode_index'],
                            'frame_in_episode': episode['length'] // 2, 'loss': value})
            offset += episode['length']
    for row in records:
        append('fixed-losses.jsonl', row)
    return {split: float(np.mean([r['loss'] for r in records if r['split'] == split])) for split in datasets}

started = time.perf_counter()
summary = {'status': 'running', 'planned_steps': config.num_train_steps, 'completed_steps': 0,
           'base_checkpoint': str(ROOT / 'assets/pi05_base'), 'training_mode': 'LoRA only; all original parameters frozen',
           'batch_size': 1, 'action_horizon': 10, 'seed': 42, 'ema': False,
           'wandb_enabled': False, 'errors': [], 'limitations': 'Pilot: 40 expert trajectories, 500 updates; no expected benchmark reproduction.'}
manager = None
try:
    jax.config.update('jax_compilation_cache_dir', str(EXP / 'jax-cache'))
    train = EpisodeDataset('train', config)
    validation = EpisodeDataset('validation', config)
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(1)
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    print('INITIALIZING_TRAIN_STATE', flush=True)
    state, state_sharding = official.init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(state)
    audit = parameter_audit(state.params)
    assert all('lora' in row['path'] for row in audit if row['trainable'])
    trainable_count = sum(row['parameters'] for row in audit if row['trainable'])
    frozen_count = sum(row['parameters'] for row in audit if not row['trainable'])
    assert 0 < trainable_count < frozen_count
    (EXP / 'parameter-audit.json').write_text(json.dumps(audit, indent=2))
    summary.update(trainable_parameters=trainable_count, frozen_parameters=frozen_count,
                   trainable_fraction=trainable_count/(trainable_count+frozen_count),
                   train_frames=len(train), validation_frames=len(validation))
    print('TRAIN_STATE_READY', trainable_count, frozen_count, flush=True)
    frozen_before = fingerprints(state.params, False)
    adapters_before = fingerprints(state.params, True)
    (EXP / 'frozen-before-sha256.json').write_text(json.dumps(frozen_before, indent=2))
    (EXP / 'adapters-before-sha256.json').write_text(json.dumps(adapters_before, indent=2))
    print('FIXED_LOSS_BEFORE', flush=True)
    summary['fixed_loss_before'] = evaluate_fixed(state, {'train': train, 'validation': validation}, 'before')
    print('FIXED_LOSS_BEFORE_RESULT', summary['fixed_loss_before'], flush=True)
    manager, resuming = checkpoints.initialize_checkpoint_dir(config.checkpoint_dir, keep_period=500,
                                                               overwrite=False, resume=False)
    assert not resuming
    checkpoints.save_state(manager, state, train, 0)
    manager.wait_until_finished()
    print('INITIAL_CHECKPOINT_SAVED', config.checkpoint_dir / '0', flush=True)
    pstep = jax.jit(functools.partial(official.train_step, config),
                   in_shardings=(replicated, state_sharding, data_sharding),
                   out_shardings=(state_sharding, replicated), donate_argnums=(1,))
    sampler = np.random.default_rng(config.seed)
    training_start = time.perf_counter()
    for step in range(config.num_train_steps):
        frame_index = int(sampler.integers(len(train)))
        batch = jax.device_put(train[frame_index], data_sharding)
        tick = time.perf_counter()
        with sharding.set_mesh(mesh):
            state, info = pstep(train_rng, state, batch)
        info = {key: float(value) for key, value in jax.device_get(info).items()}
        assert all(np.isfinite(value) for value in info.values()), info
        entry = {'step': step + 1, 'source_frame_index': frame_index,
                 'step_seconds_including_first_compile': time.perf_counter() - tick,
                 'elapsed_training_seconds': time.perf_counter() - training_start,
                 'learning_rate': float(config.lr_schedule.create()(step)), **info}
        append('training-metrics.jsonl', entry)
        summary['completed_steps'] = step + 1
        if step < 3 or (step + 1) % 25 == 0:
            print('TRAIN_STEP', json.dumps(entry), flush=True)
    summary['training_seconds'] = time.perf_counter() - training_start
    print('CHECKING_PARAMETER_UPDATES', flush=True)
    frozen_after = fingerprints(state.params, False)
    adapters_after = fingerprints(state.params, True)
    assert frozen_before == frozen_after, 'Original parameter changes detected'
    changed = [name for name in adapters_before if adapters_before[name] != adapters_after[name]]
    assert changed, 'No LoRA matrices changed'
    summary.update(original_parameters_unchanged=True, changed_lora_tensors=len(changed),
                   lora_tensor_count=len(adapters_before))
    (EXP / 'frozen-after-sha256.json').write_text(json.dumps(frozen_after, indent=2))
    (EXP / 'adapters-after-sha256.json').write_text(json.dumps(adapters_after, indent=2))
    summary['fixed_loss_after'] = evaluate_fixed(state, {'train': train, 'validation': validation}, 'after')
    adapter_arrays = {'/'.join(map(str, path)): np.asarray(jax.device_get(leaf.value))
                      for path, leaf in state.params.flat_state() if 'lora' in '/'.join(map(str, path))}
    np.savez_compressed(EXP / 'lora-adapters-step500.npz', **adapter_arrays)
    del adapter_arrays
    checkpoints.save_state(manager, state, train, config.num_train_steps)
    manager.wait_until_finished()
    summary.update(status='completed', initial_checkpoint=str(config.checkpoint_dir / '0'),
                   final_checkpoint=str(config.checkpoint_dir / str(config.num_train_steps)))
    print('TRAINING_COMPLETE', json.dumps(summary), flush=True)
except BaseException as error:
    summary['status'] = 'error'
    summary['errors'].append({'error': repr(error), 'traceback': traceback.format_exc()})
    raise
finally:
    summary['elapsed_seconds'] = time.perf_counter() - started
    (EXP / 'training-summary.json').write_text(json.dumps(summary, indent=2))
    if manager is not None:
        manager.close()

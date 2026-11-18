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
from lora_common import ROOT, EXP, PARENT, make_config, EpisodeDataset

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
    for path, leaf in params.flat_state().items():
        value = leaf.value
        rows.append({'path': '/'.join(map(str, path)), 'shape': list(value.shape),
                     'dtype': str(value.dtype), 'parameters': int(np.prod(value.shape)),
                     'trainable': 'lora' in '/'.join(map(str, path))})
    return rows

def fingerprints(params, trainable):
    result = {}
    for path, leaf in params.flat_state().items():
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
summary = {'status': 'running', 'start_step': 500, 'planned_steps': 2000,
           'additional_updates': 1500, 'completed_steps': 500, 'errors': [],
           'training_mode': 'LoRA only; all original parameters frozen',
           'batch_size': 1, 'seed': 42, 'learning_rate_after_resume': 1e-5,
           'source_checkpoint': str(PARENT / 'checkpoints/pi05_object_lora_3090/pilot500/500')}
manager = source_manager = None
try:
    jax.config.update('jax_compilation_cache_dir', str(PARENT / 'jax-cache'))
    train = EpisodeDataset('train', config)
    validation = EpisodeDataset('validation', config)
    train_rng, init_rng = jax.random.split(jax.random.key(config.seed))
    mesh = sharding.make_mesh(1)
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    print('RESTORING_TRAIN_STATE', flush=True)
    shape, state_sharding = official.init_train_state(config, init_rng, mesh, resume=True)
    source_manager, resuming = checkpoints.initialize_checkpoint_dir(
        PARENT / 'checkpoints/pi05_object_lora_3090/pilot500',
        keep_period=500, overwrite=False, resume=True)
    assert resuming
    state = checkpoints.restore_state(source_manager, shape, train, step=500)
    jax.block_until_ready(state)
    source_manager.close()
    source_manager = None
    assert int(state.step) == 500
    frozen_before = fingerprints(state.params, False)
    adapters_before = fingerprints(state.params, True)
    assert frozen_before == json.loads((PARENT / 'frozen-after-sha256.json').read_text())
    assert adapters_before == json.loads((PARENT / 'adapters-after-sha256.json').read_text())
    optimizer_audit = []
    for path, array in jax.tree_util.tree_flatten_with_path(state.opt_state)[0]:
        value = np.asarray(jax.device_get(array))
        optimizer_audit.append({'path': str(path), 'shape': list(value.shape),
                                'sha256': hashlib.sha256(value.tobytes()).hexdigest(),
                                'scalar': value.item() if value.ndim == 0 else None})
    counters = [r['scalar'] for r in optimizer_audit if r['scalar'] is not None]
    assert counters and all(c == 500 for c in counters), counters
    (EXP / 'optimizer-restored-audit.json').write_text(json.dumps(optimizer_audit, indent=2))
    (EXP / 'frozen-before-sha256.json').write_text(json.dumps(frozen_before, indent=2))
    (EXP / 'adapters-before-sha256.json').write_text(json.dumps(adapters_before, indent=2))
    sampler = np.random.default_rng(config.seed)
    old_metrics = [json.loads(s) for s in (PARENT / 'training-metrics.jsonl').read_text().splitlines()]
    assert len(old_metrics) == 500
    for row in old_metrics:
        assert int(sampler.integers(len(train))) == row['source_frame_index']
    (EXP / 'sampler-resume.json').write_text(json.dumps({'verified_prior_draws': 500, 'next_state': sampler.bit_generator.state}, indent=2))
    schedule = config.lr_schedule.create()
    for step in [500, 1000, 1999]:
        assert np.isclose(float(schedule(step)), 1e-5, rtol=1e-5)
    summary.update(optimizer_state_restored=True, optimizer_counters=counters,
                   sampler_sequence_restored=True, train_frames=len(train), validation_frames=len(validation))
    summary['fixed_loss_before'] = evaluate_fixed(state, {'train': train, 'validation': validation}, 'step500')
    previous = json.loads((PARENT / 'training-summary.json').read_text())['fixed_loss_after']
    assert all(np.isclose(summary['fixed_loss_before'][k], v, rtol=1e-5, atol=1e-7) for k,v in previous.items())
    print('RESUME_VERIFIED', json.dumps(summary), flush=True)
    manager, resuming = checkpoints.initialize_checkpoint_dir(config.checkpoint_dir,
        keep_period=500, overwrite=False, resume=False)
    assert not resuming
    pstep = jax.jit(functools.partial(official.train_step, config),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated), donate_argnums=(1,))
    training_start = time.perf_counter()
    for step in range(500, 2000):
        frame_index = int(sampler.integers(len(train)))
        batch = jax.device_put(train[frame_index], data_sharding)
        tick = time.perf_counter()
        with sharding.set_mesh(mesh):
            state, info = pstep(train_rng, state, batch)
        info = {key: float(value) for key,value in jax.device_get(info).items()}
        assert all(np.isfinite(value) for value in info.values()), info
        assert info['grad_norm'] > 0
        entry = {'step': step+1, 'source_frame_index': frame_index,
                 'step_seconds_including_first_compile': time.perf_counter()-tick,
                 'elapsed_training_seconds': time.perf_counter()-training_start,
                 'learning_rate': float(schedule(step)), **info}
        append('training-metrics.jsonl', entry)
        summary['completed_steps'] = step+1
        if step == 500 or (step+1) % 50 == 0:
            print('TRAIN_STEP', json.dumps(entry), flush=True)
    summary['training_seconds'] = time.perf_counter()-training_start
    assert int(state.step) == 2000
    frozen_after = fingerprints(state.params, False)
    adapters_after = fingerprints(state.params, True)
    assert frozen_before == frozen_after
    changed = [name for name in adapters_before if adapters_before[name] != adapters_after[name]]
    assert changed
    summary.update(original_parameters_unchanged=True, changed_lora_tensors=len(changed), lora_tensor_count=len(adapters_before))
    (EXP / 'frozen-after-sha256.json').write_text(json.dumps(frozen_after, indent=2))
    (EXP / 'adapters-after-sha256.json').write_text(json.dumps(adapters_after, indent=2))
    summary['fixed_loss_after'] = evaluate_fixed(state, {'train': train, 'validation': validation}, 'step2000')
    arrays = {'/'.join(map(str,path)): np.asarray(jax.device_get(leaf.value))
              for path,leaf in state.params.flat_state().items() if 'lora' in '/'.join(map(str,path))}
    np.savez_compressed(EXP / 'lora-adapters-step2000.npz', **arrays)
    del arrays
    checkpoints.save_state(manager, state, train, 2000)
    manager.wait_until_finished()
    summary.update(status='completed', final_checkpoint=str(config.checkpoint_dir / '2000'))
    print('TRAINING_COMPLETE', json.dumps(summary), flush=True)
except BaseException as error:
    summary['status'] = 'error'
    summary['errors'].append({'error': repr(error), 'traceback': traceback.format_exc()})
    raise
finally:
    summary['elapsed_seconds'] = time.perf_counter()-started
    (EXP / 'training-summary.json').write_text(json.dumps(summary, indent=2))
    if source_manager is not None:
        source_manager.close()
    if manager is not None:
        manager.close()

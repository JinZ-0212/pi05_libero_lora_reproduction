"""Instrument the pinned official evaluator without changing control or task outcomes."""
import argparse
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import sys
import time
import traceback

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--condition', choices=['initial', 'trained'], required=True)
parser.add_argument('--suite', choices=['libero_object'], required=True)
parser.add_argument('--experiment', required=True)
parser.add_argument('--port', type=int, default=8004)
cli = parser.parse_args()
root = Path('/home/linux/Documents/pi05-libero-20261118')
output = root / 'experiments' / cli.experiment / 'evaluation' / cli.condition / cli.suite
output.mkdir(parents=True, exist_ok=True)
if any((output / name).exists() for name in ['summary.json', 'episodes.jsonl', 'inference.jsonl']):
    raise RuntimeError('Refusing to overwrite prior evaluation records')

spec = importlib.util.spec_from_file_location('official_libero_eval', root / 'openpi-pinned/examples/libero/main.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
# Use the predeclared initial-state index 1, not the earlier index-0 cases.
original_factory = module.benchmark.get_benchmark_dict()['libero_object']
def shifted_factory():
    suite = original_factory()
    original_states = suite.get_task_init_states
    suite.get_task_init_states = lambda task_id: original_states(task_id)[1:]
    return suite
module.benchmark.get_benchmark_dict()['libero_object'] = shifted_factory
args = module.Args(host='127.0.0.1', port=cli.port, task_suite_name=cli.suite,
                   num_trials_per_task=1, replan_steps=5, seed=7,
                   video_out_path=str(output / 'videos'))
task_suite = module.benchmark.get_benchmark_dict()[cli.suite]()
expected_episodes = task_suite.n_tasks
assert expected_episodes == 10
records, errors, calls = [], [], []
steps = 0
first_prompts = set()
active_env = []

def append_record(name, entry):
    with (output / name).open('a') as f:
        f.write(json.dumps(entry) + '\n')

class ResultsHandler(logging.Handler):
    task = None
    def emit(self, record):
        message = record.getMessage()
        if message.startswith('\nTask: '):
            self.task = message.strip().removeprefix('Task: ')
        if message.startswith('Caught exception:'):
            errors.append({'task': self.task, 'error': message})
            if 'NONFINITE_' in message:
                raise SystemExit('Non-finite numerical result; stopping')
        if message.startswith('Success: '):
            entry = {'task': self.task, 'initial_state_index': 1,
                     'success': message == 'Success: True',
                     'status': 'error' if any(e.get('task') == self.task for e in errors) else 'completed'}
            records.append(entry)
            append_record('episodes.jsonl', entry)
            print('EPISODE_RESULT', cli.condition, cli.suite, len(records), entry['success'], flush=True)

logging.basicConfig(level=logging.INFO, force=True)
logging.getLogger().addHandler(ResultsHandler())
original_infer = module._websocket_client_policy.WebsocketClientPolicy.infer

def timed_infer(self, obs):
    started = time.perf_counter()
    state = np.asarray(obs['observation/state'])
    if not np.isfinite(state).all():
        raise ValueError('NONFINITE_OBSERVATION')
    result = original_infer(self, obs)
    elapsed_ms = (time.perf_counter() - started) * 1000
    actions = np.asarray(result['actions'])
    if not np.isfinite(actions).all():
        raise ValueError('NONFINITE_ACTIONS')
    if obs['prompt'] not in first_prompts:
        first_prompts.add(obs['prompt'])
        append_record('first-observations.jsonl', {
            'task': obs['prompt'], 'state': state.tolist(),
            'image_sha256': hashlib.sha256(np.asarray(obs['observation/image']).tobytes()).hexdigest(),
            'wrist_image_sha256': hashlib.sha256(np.asarray(obs['observation/wrist_image']).tobytes()).hexdigest(),
        })
    append_record('actions.jsonl', {'call': len(calls), 'prompt': obs['prompt'],
                                   'state': state.tolist(), 'actions': actions.tolist()})
    entry = {'call': len(calls), 'rpc_ms': elapsed_ms, 'server_timing': result.get('server_timing'),
             'policy_timing': result.get('policy_timing'), 'action_shape': list(actions.shape)}
    calls.append(entry)
    append_record('inference.jsonl', entry)
    return result

module._websocket_client_policy.WebsocketClientPolicy.infer = timed_infer
original_get_env = module._get_libero_env

def tracked_get_env(*params, **kwargs):
    if active_env:
        active_env.pop().close()
    env, description = original_get_env(*params, **kwargs)
    active_env.append(env)
    original_set = env.set_init_state
    original_step = env.step
    def record_initial(state):
        state = np.asarray(state)
        append_record('initial-states.jsonl', {'task': description, 'initial_state_index': 1,
                      'state': state.tolist(), 'sha256': hashlib.sha256(state.tobytes()).hexdigest()})
        return original_set(state)
    def record_step(action):
        global steps
        values = np.asarray(action)
        if not np.isfinite(values).all():
            raise ValueError('NONFINITE_EXECUTED_ACTION')
        append_record('executed-actions.jsonl', {'step': steps, 'task': description, 'action': values.tolist()})
        steps += 1
        return original_step(action)
    env.set_init_state = record_initial
    env.step = record_step
    return env, description

module._get_libero_env = tracked_get_env
start = time.perf_counter()
status = 'running'
try:
    module.eval_libero(args)
    status = 'completed' if not errors and len(records) == expected_episodes else 'incomplete'
except BaseException as error:
    status = 'error'
    errors.append({'error': repr(error), 'traceback': traceback.format_exc()})
    raise
finally:
    for env in active_env:
        env.close()
    completed = [r for r in records if r['status'] == 'completed']
    summary = {
        'status': status, 'checkpoint': str(root / 'experiments' / cli.experiment / 'checkpoints/pi05_object_lora_3090/pilot500' / ('0' if cli.condition == 'initial' else '500')),
        'condition': cli.condition, 'suite': cli.suite, 'seed': args.seed, 'initial_state_index': 1,
        'replan_steps': 5, 'flow_steps': 10, 'episodes_per_task': 1,
        'expected_episodes': expected_episodes, 'completed_episodes': len(completed),
        'successes': sum(r['success'] for r in completed),
        'success_rate': sum(r['success'] for r in completed) / len(completed) if completed else None,
        'model_calls': len(calls), 'elapsed_seconds': time.perf_counter() - start,
        'executed_steps_including_settling': steps, 'weight_updates': 0 if cli.condition == 'initial' else 500,
        'action_horizon': 10,
        'discrete_state_input': False,
        'normalization': 'Object pilot: fitted on 40 training episodes only',
        'errors': errors,
        'limitations': 'Pilot before/after comparison: 40 expert training episodes, 500 LoRA updates, one simulator initial state per task, environment seed=7. The initial-state index 1 was predeclared and is not guaranteed disjoint from demonstration collection states. This is not an official benchmark reproduction.',
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('SUITE_SUMMARY', json.dumps(summary), flush=True)
if status != 'completed':
    raise SystemExit(2)

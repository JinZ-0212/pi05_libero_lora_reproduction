import argparse
import logging
import time

import jax

from openpi.policies.libero_policy import make_libero_example
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from lora_common import make_config

parser = argparse.ArgumentParser()
parser.add_argument('--step', choices=[0, 500], type=int, required=True)
args = parser.parse_args()
logging.basicConfig(level=logging.INFO, force=True)
config = make_config()
policy = policy_config.create_trained_policy(config, config.checkpoint_dir / str(args.step),
                                            sample_kwargs={'num_steps': 10})
tick = time.perf_counter()
example = make_libero_example()
policy.infer(example)
# Precompile before opening the websocket, then restore the declared initial RNG.
policy._rng = jax.random.key(0)
logging.info('PILOT_POLICY_PREWARMED seconds=%.3f rng_reset=0', time.perf_counter() - tick)
logging.info('PILOT_POLICY_READY step=%s', args.step)
websocket_policy_server.WebsocketPolicyServer(policy=policy, host='127.0.0.1', port=8004,
                                              metadata=policy.metadata).serve_forever()

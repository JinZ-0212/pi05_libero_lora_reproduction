"""Reconstruct the policy from exported LoRA weights plus official base weights.

This is an additional portable loader; the actual pilot evaluation uses the full
Orbax checkpoints via serve_pilot.py. Keep the paired training normalization.
"""
import argparse
from pathlib import Path

import flax.traverse_util
import jax.numpy as jnp
import numpy as np

from openpi import transforms
from openpi.models import model as model_lib
from openpi.policies import policy as policy_lib
from openpi.shared import normalize
from openpi.serving import websocket_policy_server
from lora_common import ROOT, EXP, ASSET_ID, make_config

def load_exported_policy(base_params, adapters, norm_folder):
    config = make_config()
    params = model_lib.restore_params(Path(base_params), dtype=jnp.bfloat16)
    flat = flax.traverse_util.flatten_dict(params, sep='/')
    with np.load(adapters, allow_pickle=False) as archive:
        assert archive.files and all('lora' in key for key in archive.files)
        assert not set(flat) & set(archive.files)
        flat.update({key:jnp.asarray(archive[key],dtype=jnp.bfloat16) for key in archive.files})
    params = flax.traverse_util.unflatten_dict(flat, sep='/')
    model = config.model.load(params, remove_extra_params=False)
    data = config.data.create(config.assets_dirs, config.model)
    norm = normalize.load(Path(norm_folder))
    return policy_lib.Policy(
        model,
        transforms=[*data.data_transforms.inputs,
                    transforms.Normalize(norm,use_quantiles=data.use_quantile_norm),
                    *data.model_transforms.inputs],
        output_transforms=[*data.model_transforms.outputs,
                           transforms.Unnormalize(norm,use_quantiles=data.use_quantile_norm),
                           *data.data_transforms.outputs],
        sample_kwargs={'num_steps':10},
    )

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--base-params',default=str(ROOT/'assets/pi05_base/params'))
    parser.add_argument('--adapters',default=str(EXP/'lora-adapters-step500.npz'))
    parser.add_argument('--norm-folder',default=str(make_config().assets_dirs/ASSET_ID))
    parser.add_argument('--port',type=int,default=8004)
    args=parser.parse_args()
    policy=load_exported_policy(args.base_params,args.adapters,args.norm_folder)
    websocket_policy_server.WebsocketPolicyServer(policy=policy,host='127.0.0.1',port=args.port,
                                                  metadata=policy.metadata).serve_forever()

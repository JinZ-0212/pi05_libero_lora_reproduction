"""CPU reconstruction check for portable adapter export; no training or rollout."""
import hashlib
import json

import flax.traverse_util
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from lora_common import ROOT, EXP, make_config

expected_frozen=json.loads((EXP/'frozen-after-sha256.json').read_text())
expected_adapters=json.loads((EXP/'adapters-after-sha256.json').read_text())
params=model_lib.restore_params(ROOT/'assets/pi05_base/params',dtype=jnp.bfloat16)
flat=flax.traverse_util.flatten_dict(params,sep='/')
assert set(flat)==set(expected_frozen)
for name,array in flat.items():
    assert hashlib.sha256(np.asarray(array).tobytes()).hexdigest()==expected_frozen[name],name
with np.load(EXP/'lora-adapters-step2000.npz',allow_pickle=False) as archive:
    assert set(archive.files)==set(expected_adapters)
    for name in archive.files:
        assert hashlib.sha256(archive[name].tobytes()).hexdigest()==expected_adapters[name],name
        flat[name]=jnp.asarray(archive[name],dtype=jnp.bfloat16)
params=flax.traverse_util.unflatten_dict(flat,sep='/')
model=make_config().model.load(params,remove_extra_params=False)
result={'status':'verified','base_parameters_match_training_frozen_hashes':True,
        'exported_adapters_match_training_final_hashes':True,
        'combined_model_passes_strict_parameter_tree_and_shape_load':True,
        'device':'CPU','scope':'Parameter reconstruction only; actual rollouts use the saved full Orbax checkpoints.'}
(EXP/'adapter-export-verification.json').write_text(json.dumps(result,indent=2))
print('ADAPTER_EXPORT_VERIFIED',json.dumps(result),flush=True)

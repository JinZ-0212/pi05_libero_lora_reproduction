import csv
import hashlib
import json
from pathlib import Path
import statistics
import zipfile

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/home/linux/Documents/pi05-libero-20261118')
EXP = ROOT / 'experiments/pi05-object-lora-2000-20261118'
PARENT = ROOT / 'experiments/pi05-object-lora-20261118'
OUT = EXP / 'comparison'
OUT.mkdir(exist_ok=True)

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

training = json.loads((EXP / 'training-summary.json').read_text())
assert training['status'] == 'completed' and training['completed_steps'] == 2000
assert training['original_parameters_unchanged'] and training['changed_lora_tensors'] > 0
assert training['optimizer_state_restored'] and training['sampler_sequence_restored']
assert not training['errors']
assert (EXP / 'runner-exit-code.txt').read_text().strip() == '0'
metrics = rows(EXP / 'training-metrics.jsonl')
assert [r['step'] for r in metrics] == list(range(501,2001))
assert all(np.isfinite(r['loss']) and np.isfinite(r['grad_norm']) and r['grad_norm'] > 0 for r in metrics)
assert all(np.isclose(r['learning_rate'], 1e-5) for r in metrics)
assert json.loads((EXP/'frozen-after-sha256.json').read_text()) == json.loads((PARENT/'frozen-after-sha256.json').read_text())
expected_adapters = json.loads((EXP/'adapters-after-sha256.json').read_text())
with np.load(EXP/'lora-adapters-step2000.npz', allow_pickle=False) as adapters:
    assert set(adapters.files) == set(expected_adapters)
    for name in adapters.files:
        assert hashlib.sha256(adapters[name].tobytes()).hexdigest() == expected_adapters[name]
original_training = json.loads((PARENT/'training-summary.json').read_text())
folders = {500: EXP/'reference/step500-evaluation/libero_object',
           2000: EXP/'evaluation/trained/libero_object'}
evaluations, episodes, observations, states = {}, {}, {}, {}
for step, folder in folders.items():
    evaluation = json.loads((folder/'summary.json').read_text())
    assert evaluation['status'] == 'completed' and not evaluation['errors']
    assert evaluation['completed_episodes'] == 10 and evaluation['weight_updates'] == step
    evaluations[str(step)] = evaluation
    episodes[step] = {r['task']:r for r in rows(folder/'episodes.jsonl')}
    observations[step] = {r['task']:r for r in rows(folder/'first-observations.jsonl')}
    states[step] = {r['task']:r for r in rows(folder/'initial-states.jsonl')}
assert (EXP/'evaluation-trained-exit-code.txt').read_text().strip() == '0'
paired = []
for task, before in episodes[500].items():
    left, right = observations[500][task], observations[2000][task]
    identical = (states[500][task]['sha256'] == states[2000][task]['sha256']
        and left['image_sha256'] == right['image_sha256']
        and left['wrist_image_sha256'] == right['wrist_image_sha256']
        and np.array_equal(left['state'], right['state']))
    assert identical, task
    paired.append({'task':task, 'step500_success':before['success'],
                   'step2000_success':episodes[2000][task]['success'],
                   'initial_sim_state_robot_state_and_images_identical':identical})
assert len(paired) == 10
norm = EXP/'assets/pi05_object_lora_3090/local/libero_object_small/norm_stats.json'
assert norm.read_bytes() == (PARENT/'assets/pi05_object_lora_3090/local/libero_object_small/norm_stats.json').read_bytes()
assert norm.read_bytes() == (Path(training['final_checkpoint'])/'assets/local/libero_object_small/norm_stats.json').read_bytes()
gpu_rows = list(csv.reader((EXP/'training-gpu.csv').open()))[1:]
peak = max(int(r[1].strip().split()[0]) for r in gpu_rows if len(r)>1)
sample_task = paired[0]['task']
segment = sample_task.replace(' ','_')
frames_by_step = {}
canvas = Image.new('RGB',(672,512),'white')
pen = ImageDraw.Draw(canvas)
for row,step in enumerate([500,2000]):
    videos = list((folders[step]/'videos').glob(f'rollout_{segment}_*.mp4'))
    assert len(videos) == 1
    reader=imageio.get_reader(videos[0])
    frames=[f for f in reader]
    reader.close()
    frames_by_step[step]=frames
    for col,fraction in enumerate([0,.5,1]):
        index=round((len(frames)-1)*fraction)
        canvas.paste(Image.fromarray(frames[index]),(224*col,256*row+32))
        pen.text((224*col+3,256*row+8),f'step {step} / frame {index}',fill='black')
canvas.save(OUT/'preview.png')
with imageio.get_writer(OUT/'step500-vs2000.mp4',fps=10) as writer:
    for index in range(max(len(v) for v in frames_by_step.values())):
        frame=Image.new('RGB',(448,256),'white')
        draw=ImageDraw.Draw(frame)
        for col,step in enumerate([500,2000]):
            source=frames_by_step[step]
            selected=min(index,len(source)-1)
            frame.paste(Image.fromarray(source[selected]),(224*col,32))
            end=' / ended' if index>=len(source) else ''
            draw.text((224*col+3,8),f'step {step} / {selected}{end}',fill='black')
        writer.append_data(np.asarray(frame))
summary = {'training':training, 'original_training':original_training,
           'evaluations':evaluations, 'paired_tasks':paired,
           'all_10_initial_cases_identical':True, 'verified_adapter_tensors':len(expected_adapters),
           'training_peak_device_vram_MiB':peak,
           'warm_step_median_seconds':statistics.median(r['step_seconds_including_first_compile'] for r in metrics[1:]),
           'first_50_continuation_loss_mean':statistics.mean(r['loss'] for r in metrics[:50]),
           'last_50_continuation_loss_mean':statistics.mean(r['loss'] for r in metrics[-50:]),
           'sample_task':sample_task,'sample_selection':'First task in evaluator order, regardless of success.',
           'limitations':['One training seed and one previously inspected simulator state per task.','Learning rate remains at original schedule floor 1e-5.','Fixed validation diagnostic covers only 10 midpoint frames.','Expert collection and simulator initial-state disjointness not audited.']}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
with (OUT/'paired-results.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(paired[0]))
    writer.writeheader()
    writer.writerows(paired)
print('CONTINUATION_ANALYSIS_COMPLETE',json.dumps(summary),flush=True)

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
EXP = ROOT / 'experiments/pi05-object-lora-20261118'
out = EXP / 'comparison'
out.mkdir(exist_ok=True)

def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

training = json.loads((EXP / 'training-summary.json').read_text())
assert training['status'] == 'completed' and training['completed_steps'] == 500
assert training['original_parameters_unchanged'] and training['changed_lora_tensors'] > 0
assert (EXP / 'runner-exit-code.txt').read_text().strip() == '0'
metrics = read_lines(EXP / 'training-metrics.jsonl')
assert len(metrics) == 500
assert [r['step'] for r in metrics] == list(range(1,501))
assert all(np.isfinite(r['loss']) and np.isfinite(r['grad_norm']) for r in metrics)
gpu_rows = list(csv.reader((EXP / 'training-gpu.csv').open()))[1:]
peak_memory = max(int(row[1].strip().split()[0]) for row in gpu_rows if len(row) > 1)
episodes, observations, initial_states, evals = {}, {}, {}, {}
for condition, step in [('initial', 0), ('trained', 500)]:
    folder = EXP / 'evaluation' / condition / 'libero_object'
    summary = json.loads((folder / 'summary.json').read_text())
    assert summary['status'] == 'completed' and not summary['errors'] and summary['completed_episodes'] == 10
    assert (EXP / f'evaluation-{condition}-exit-code.txt').read_text().strip() == '0'
    episodes[condition] = {r['task']:r for r in read_lines(folder / 'episodes.jsonl')}
    observations[condition] = {r['task']:r for r in read_lines(folder / 'first-observations.jsonl')}
    initial_states[condition] = {r['task']:r for r in read_lines(folder / 'initial-states.jsonl')}
    calls = read_lines(folder / 'inference.jsonl')
    warm = [r['server_timing']['infer_ms'] for r in calls[5:]]
    evals[condition] = {'summary': summary, 'warm_server_median_ms': statistics.median(warm),
                        'warm_server_p95_ms': float(np.percentile(warm,95))}
    checkpoint = Path(training['initial_checkpoint' if step == 0 else 'final_checkpoint'])
    source_norm = EXP / 'assets/pi05_object_lora_3090/local/libero_object_small/norm_stats.json'
    assert (checkpoint / 'assets/local/libero_object_small/norm_stats.json').read_bytes() == source_norm.read_bytes()

paired = []
for task, initial in episodes['initial'].items():
    before, after = observations['initial'][task], observations['trained'][task]
    initial_state_equal = initial_states['initial'][task]['sha256'] == initial_states['trained'][task]['sha256']
    image_equal = before['image_sha256'] == after['image_sha256'] and before['wrist_image_sha256'] == after['wrist_image_sha256']
    state_difference = float(np.max(np.abs(np.asarray(before['state'])-np.asarray(after['state']))))
    paired.append({'task':task,'initial_state_index':1,'initial_success':initial['success'],
                   'trained_success':episodes['trained'][task]['success'],
                   'initial_sim_state_identical':initial_state_equal,'first_images_identical':image_equal,
                   'first_robot_state_max_abs_difference':state_difference})
assert len(paired) == 10
assert all(r['initial_sim_state_identical'] and r['first_images_identical'] and r['first_robot_state_max_abs_difference']==0 for r in paired)

sample_task = paired[0]['task']
segment = sample_task.replace(' ','_')
frames_by_condition, sample_videos = {}, {}
canvas = Image.new('RGB',(672,512),'white')
draw=ImageDraw.Draw(canvas)
for row, condition in enumerate(['initial','trained']):
    files=list((EXP/'evaluation'/condition/'libero_object/videos').glob(f'rollout_{segment}_*.mp4'))
    assert len(files)==1
    video=files[0]
    sample_videos[condition]=str(video)
    reader=imageio.get_reader(video)
    frames=[frame for frame in reader]
    reader.close()
    frames_by_condition[condition]=frames
    for col,fraction in enumerate([0,0.5,1]):
        index=round((len(frames)-1)*fraction)
        canvas.paste(Image.fromarray(frames[index]),(224*col,256*row+32))
        draw.text((224*col+3,256*row+8),f'{condition} / frame {index}',fill='black')
canvas.save(out/'preview.png')
with imageio.get_writer(out/'before-after.mp4',fps=10) as writer:
    for index in range(max(len(frames) for frames in frames_by_condition.values())):
        frame=Image.new('RGB',(448,256),'white')
        pen=ImageDraw.Draw(frame)
        for col,condition in enumerate(['initial','trained']):
            frames=frames_by_condition[condition]
            source_index=min(index,len(frames)-1)
            frame.paste(Image.fromarray(frames[source_index]),(224*col,32))
            end=' / ended' if index>=len(frames) else ''
            pen.text((224*col+3,8),f'{condition} / {source_index}{end}',fill='black')
        writer.append_data(np.asarray(frame))
summary={'training':training,'data_audit':json.loads((EXP/'data-audit.json').read_text()),
         'training_peak_device_vram_MiB':peak_memory,
         'warm_training_step_median_seconds_excluding_first':statistics.median(r['step_seconds_including_first_compile'] for r in metrics[1:]),
         'first_50_training_loss_mean':statistics.mean(r['loss'] for r in metrics[:50]),
         'last_50_training_loss_mean':statistics.mean(r['loss'] for r in metrics[-50:]),
         'evaluation':evals,'paired_tasks':paired,'all_10_initial_cases_identical':True,
         'sample_task':sample_task,'sample_selection':'First task in evaluator order, regardless of success.',
         'sample_videos':sample_videos,
         'limitations':['500-update training pilot, not a converged policy.','Validation loss uses one fixed midpoint per held-out trajectory.','Simulation uses one initial state per task; demonstration collection state overlap not audited.']}
(out/'summary.json').write_text(json.dumps(summary,indent=2))
with (out/'paired-results.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(paired[0]))
    writer.writeheader()
    writer.writerows(paired)
print(json.dumps(summary,indent=2))

archive=EXP/'pi05-object-lora-pilot-results.zip'
excluded={'raw','checkpoints','jax-cache','__pycache__'}
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for file in sorted(EXP.rglob('*')):
        relative=file.relative_to(EXP)
        if not file.is_file() or file==archive or any(p in excluded for p in relative.parts):
            continue
        if file.name in ['object-data.tar','archive-sha256.txt']:
            continue
        z.write(file,relative,compress_type=zipfile.ZIP_STORED if file.suffix=='.npz' else zipfile.ZIP_DEFLATED)
    z.write(ROOT/'pi05-base-manifest.json','base-checkpoint-manifest.json')
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert len([n for n in z.namelist() if n.endswith('.mp4') and not n.startswith('attempts/')])==21
with archive.open('rb') as f:
    hasher=hashlib.sha256()
    for block in iter(lambda:f.read(1024*1024),b''):
        hasher.update(block)
    digest=hasher.hexdigest()
(EXP/'archive-sha256.txt').write_text(digest+'\n')
print('PILOT_ARCHIVE_READY',archive.stat().st_size,digest)

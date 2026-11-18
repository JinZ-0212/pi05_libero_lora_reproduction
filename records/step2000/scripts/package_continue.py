from pathlib import Path
import hashlib
import zipfile

EXP=Path('/home/linux/Documents/pi05-libero-20261118/experiments/pi05-object-lora-2000-20261118')
archive=EXP/'pi05-lora-step2000-results.zip'
excluded={'raw','checkpoints','jax-cache','__pycache__'}
assert (EXP/'README.md').exists() and (EXP/'pi05-lora-step2000-training.png').exists()
with zipfile.ZipFile(archive,'w') as package:
    for file in sorted(EXP.rglob('*')):
        relative=file.relative_to(EXP)
        if not file.is_file() or file==archive or any(p in excluded for p in relative.parts):
            continue
        if file.name=='archive-sha256.txt':
            continue
        package.write(file,relative,compress_type=zipfile.ZIP_STORED if file.suffix=='.npz' else zipfile.ZIP_DEFLATED)
with zipfile.ZipFile(archive) as package:
    assert package.testzip() is None
    assert len([n for n in package.namelist() if n.startswith('evaluation/') and n.endswith('.mp4')])==10
hasher=hashlib.sha256()
with archive.open('rb') as stream:
    for block in iter(lambda:stream.read(1024*1024),b''):
        hasher.update(block)
(EXP/'archive-sha256.txt').write_text(hasher.hexdigest()+'\n')
print('CONTINUATION_PACKAGE_READY',archive.stat().st_size,hasher.hexdigest(),flush=True)

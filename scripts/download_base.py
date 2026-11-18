import base64
import concurrent.futures
import hashlib
import json
import pathlib
import shutil
import threading
import time
import urllib.request
import google_crc32c

from project_paths import ROOT as root, REPO
manifest=json.loads((REPO/'configs/base-checkpoint-manifest.json').read_text(encoding='utf-8-sig'))
dest=root/'assets/pi05_base'
chunk_dir=root/'cache/pi05-base-chunks'
chunk_dir.mkdir(parents=True,exist_ok=True)
chunk_size=4*1024*1024
jobs=[]
objects=[]
for item in manifest['items']:
    rel=item['name'].removeprefix('checkpoints/pi05_base/')
    target=dest/rel
    size=int(item['size'])
    if target.exists() and target.stat().st_size==size:
        continue
    partial=target.with_name(target.name+'.partial')
    prefix=partial.stat().st_size if partial.exists() else 0
    item.update(target=str(target),partial=str(partial),prefix=prefix,parts=[])
    for start in range(prefix,size,chunk_size):
        end=min(start+chunk_size,size)-1
        part=chunk_dir/(hashlib.sha256(item['name'].encode()).hexdigest()[:16]+f'.{start}')
        item['parts'].append(part)
        jobs.append((item,start,end,part))
    objects.append(item)

lock=threading.Lock()
finished_bytes=0
start_time=time.monotonic()
def fetch(job):
    global finished_bytes
    item,start,end,part=job
    length=end-start+1
    if not(part.exists() and part.stat().st_size==length):
        for attempt in range(5):
            try:
                request=urllib.request.Request('https://storage.googleapis.com/openpi-assets/'+item['name']+'?generation='+item['generation'],headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(request,timeout=90) as response:
                    if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{item["size"]}':
                        raise RuntimeError('range response mismatch')
                    with part.open('wb') as output:
                        shutil.copyfileobj(response,output,1024*1024)
                if part.stat().st_size!=length:
                    raise RuntimeError('chunk size mismatch')
                break
            except Exception as error:
                print('RETRY',start,str(error),flush=True)
                if attempt==4:
                    raise
                time.sleep(2)
    with lock:
        finished_bytes+=length
        if finished_bytes//(256*1024*1024)!=(finished_bytes-length)//(256*1024*1024):
            elapsed=time.monotonic()-start_time
            print(f'PROGRESS {finished_bytes/1e9:.2f} GB {finished_bytes/elapsed/1e6:.2f} MB/s',flush=True)

print('SEGMENTS',len(jobs),'WORKERS',8,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    list(pool.map(fetch,jobs))
for item in objects:
    target=pathlib.Path(item['target'])
    partial=pathlib.Path(item['partial'])
    target.parent.mkdir(parents=True,exist_ok=True)
    if partial.exists() and partial.stat().st_size!=item['prefix']:
        raise RuntimeError('prefix changed unexpectedly')
    with partial.open('ab') as output:
        for part in item['parts']:
            with part.open('rb') as data:
                shutil.copyfileobj(data,output,4*1024*1024)
    if partial.stat().st_size!=int(item['size']):
        raise RuntimeError('assembled size mismatch')
    digest=google_crc32c.Checksum()
    with partial.open('rb') as data:
        while block:=data.read(4*1024*1024):
            digest.update(block)
    if base64.b64encode(digest.digest()).decode()!=item['crc32c']:
        raise RuntimeError('checksum mismatch '+str(target))
    partial.rename(target)
    for part in item['parts']:
        part.unlink()
    print('CRC32C_VERIFIED',target,flush=True)
for item in manifest['items']:
    target=dest/item['name'].removeprefix('checkpoints/pi05_base/')
    digest=google_crc32c.Checksum()
    with target.open('rb') as data:
        while block:=data.read(4*1024*1024):
            digest.update(block)
    if base64.b64encode(digest.digest()).decode()!=item['crc32c']:
        raise RuntimeError('final checksum mismatch '+str(target))
import re
for part in chunk_dir.iterdir():
    if part.is_file() and re.fullmatch(r'[0-9a-f]{16}\.\d+',part.name):
        part.unlink()
print('CHECKPOINT_COMPLETE',flush=True)

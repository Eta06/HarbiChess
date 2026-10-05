"""Root serial transport; same fixed clock for smoke,21chunks,aggregate/readback."""
import base64
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

STAGE = Path('/workspace/HarbiChess/experiments/ufuk/fresh-final8192-artifact-capsule-v2')
OUT = Path('/workspace/work/harbichess/continuation-20261005/fresh-final8192-dispatch')
RAM = Path('/dev/shm/harbichess-fresh-final8192-transport-20261005')
OUT.mkdir(exist_ok=False)
RAM.mkdir(exist_ok=False)
sys.path.insert(0, str(STAGE))
import artifact_capsule as receiver
from launcher_inputs import dispatch_inputs

raw = (STAGE/'manifest-8192-DRAFT.json').read_bytes()
manifest_sha = hashlib.sha256(raw).hexdigest()
m = json.loads(raw)
control = json.loads((STAGE/'transport-control-PENDING.json').read_text())
END = control['deadline_epoch']
receiver.ACTIVE_CONTROL = control
receiver.validate(m)
originals = {r['path']: Path(r['original_path']).read_bytes() for r in m['files']}
for r in m['files']:
    assert receiver.sha(originals[r['path']]) == r['sha256']
data = receiver.capsule(originals)
receiver.verify(data, m)
(RAM/'capsule.tar.gz').write_bytes(data)
workflow = 'fresh-final8192-artifact-capsule-v2.yml'
repo = 'Eta06/HarbiChess'
runs_path = f'repos/{repo}/actions/workflows/{workflow}/runs?per_page=100'
rows = []

def guard():
    receiver.release.check_clock(control)
    if sum(p.stat().st_size for p in RAM.rglob('*') if p.is_file()) > 8*1024**2:
        raise RuntimeError('local transport RAM8MiB ceiling')

def gh(arguments, input_path=None):
    guard()
    command = ['gh', 'api', *arguments]
    if input_path is not None:
        command += ['--input', str(input_path)]
    result = subprocess.run(command, capture_output=True, timeout=min(60, max(1, END-time.time())))
    if result.returncode:
        raise RuntimeError('github-api-code-'+str(result.returncode))
    return json.loads(result.stdout) if result.stdout.strip() else None

def dispatch(mode, ordinal=0, run_ids=()):
    guard()
    payload = '' if mode=='aggregate' else base64.b64encode(data[ordinal*receiver.CHUNK:(ordinal+1)*receiver.CHUNK]).decode()
    request = dispatch_inputs(raw, manifest_sha, mode, ordinal=ordinal, payload=payload, run_ids=run_ids)
    path = RAM/f'{mode}-{ordinal}.request.json'
    path.open('xb').write(request)
    os.chmod(path, 0o600)
    prior = {r['id'] for r in gh([runs_path])['workflow_runs']}
    gh(['--method', 'POST', f'repos/{repo}/actions/workflows/{workflow}/dispatches'], path)
    title = f'Final8192 {mode} / {manifest_sha[:12]} / {ordinal}'
    ident = None
    observed_first = time.time()
    while True:
        guard()
        if ident is None:
            matches = [r for r in gh([runs_path])['workflow_runs'] if r['id'] not in prior and r['display_title']==title and r['event']=='workflow_dispatch']
            if len(matches)>1:
                raise RuntimeError('ambiguous-owned-dispatch')
            run = matches[0] if matches else None
            if run:
                ident = run['id']
        else:
            run = gh([f'repos/{repo}/actions/runs/{ident}'])
        if run and run['status']=='completed':
            record = {k: run[k] for k in ['id','status','conclusion','head_sha','html_url','display_title','created_at','updated_at']}
            record.update(mode=mode,ordinal=ordinal,manifest_sha256=manifest_sha,original_deadline_epoch=END)
            (OUT/f'{mode}-{ordinal}.result.json').open('x').write(json.dumps(record,indent=2)+'\n')
            rows.append(record)
            print(json.dumps(record), flush=True)
            if run['conclusion']!='success':
                raise RuntimeError('owned-workflow-'+str(run['conclusion']))
            return ident
        if time.time()-observed_first>600:
            raise TimeoutError('single-dispatch600 observational ceiling')
        time.sleep(5)

try:
    ids = [dispatch('chunk',0)]
    # Actual upload-artifact ZIP/API/schema/source readback before allotherchunks.
    run, artifacts, archive = receiver.fetch(ids[0], receiver.artifact_name(m,0),m,manifest_sha)
    a = artifacts[0]
    assert len(archive)==a['size_in_bytes'], 'smoke actual ZIP/API size mismatch'
    import io
    import zipfile
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        entries=z.infolist()
        assert len(entries)==1 and entries[0].filename=='chunk.bin'
        part=z.read(entries[0])
        assert len(part)==m['chunks'][0]['bytes'] and receiver.sha(part)==m['chunks'][0]['sha256']
    smoke={'status':'PASS-one-real-tempchunk-ZIP-source-API-size-SHA-not-public-backup','run_id':ids[0],'archive_bytes':len(archive),'artifact_name':a['name'],'artifact_chunk_sha256':receiver.sha(part),'source_seals':run['source_blobs'],'finished_epoch':time.time()}
    (OUT/'chunk0-smoke-result.json').open('x').write(json.dumps(smoke,indent=2)+'\n')
    for ordinal in range(1,len(m['chunks'])):
        ids.append(dispatch('chunk',ordinal))
    dispatch('aggregate',0,ids)
    status='all21-tempchunks-and-capsule-workflow-success-await-anonymous-readback'
except BaseException as exc:
    status='failed-preserved'
    error=repr(exc)
    print(json.dumps({'status':status,'error':error}),flush=True)
result={'status':status,'rows':rows,'original_first_epoch':control['started_epoch'],'original_deadline_epoch':END,'finished_epoch':time.time(),'manifest_sha256':manifest_sha,'GPU_used':False,'new_paid_compute':False,'public_backup_not_yet_independently_readback':True}
if status=='failed-preserved':
    result['error']=error
(OUT/'cohort-result.json').open('x').write(json.dumps(result,indent=2)+'\n')

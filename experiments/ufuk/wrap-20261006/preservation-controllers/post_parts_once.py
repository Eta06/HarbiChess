"""Exact fixed parts, one POST each; preserve failures, never repeat ambiguities."""
import base64
import hashlib
import json
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).parent
ROOT = Path('/dev/shm/harbichess-nnue-state-v9-parts-v1')
control = json.loads((HERE / 'parts-control-PENDING.json').read_bytes())
assert control['first'] <= time.time() < control['deadline']
for row in control['parts']:
    raw = Path(row['local_path']).read_bytes()
    assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
    body = ROOT / f"part{row['index']:02d}-POST-body.json"
    with body.open('x') as stream:
        json.dump(dict(content=base64.b64encode(raw).decode(), encoding='base64'), stream)
    started = HERE / f"part{row['index']:02d}-POST-started.json"
    with started.open('x') as stream:
        json.dump(dict(attempts=1, epoch=time.time(), expected_git_blob_sha1=row['git_blob_sha1'],
                       original_first=control['first'], original_deadline=control['deadline'],
                       no_automatic_retry=True), stream)
    p = subprocess.run(['gh', 'api', '--method', 'POST', 'repos/Eta06/HarbiChess/git/blobs',
                        '--input', str(body)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                       timeout=min(30, control['deadline'] - time.time()))
    result = dict(index=row['index'], returncode=p.returncode, finished_epoch=time.time())
    if p.returncode == 0:
        assert len(p.stdout) < 8192
        data = json.loads(p.stdout)
        assert data['sha'] == row['git_blob_sha1']
        result['git_blob_sha1'] = data['sha']
    with (HERE / f"part{row['index']:02d}-POST-result.json").open('x') as stream:
        json.dump(result, stream)
    print(json.dumps(result), flush=True)
    assert p.returncode == 0, 'part POST failed; preserve and reconcile, no automatic repeat'
assert time.time() < control['deadline']
with (HERE / 'all-parts-POSTed.json').open('x') as stream:
    json.dump(dict(status='all-eight-exact-Gitblobs-POSTed-not-public', finished_epoch=time.time(),
                   expected_parts=8, public_Release_verified=False), stream)

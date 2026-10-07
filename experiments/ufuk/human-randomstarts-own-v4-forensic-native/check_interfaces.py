"""Bounded import/CLI help witness only; never calls a model, search or optimizer."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

p=Path(__file__).resolve().parent
names=('root_bank','prepare_bank','prepare_collection','metadata_factory','run_collection','audit_collection_six','convert','contracts','specs','prove','train','admit_parent','parent_seal','ledger','initialize','prepare_zero')
first=time.time(); result=[]
env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
for name in names:
    r=subprocess.run([sys.executable,str(p/(name+'.py')),'--help'],cwd=p,env=env,capture_output=True,timeout=max(.1,60-(time.time()-first)))
    result.append(dict(file=name+'.py',returncode=r.returncode,stdout_sha256=hashlib.sha256(r.stdout).hexdigest(),stderr=r.stderr.decode()))
    if r.returncode:
        break
with (p/'cli-interfaces.json').open('x') as f:
    json.dump(dict(status='PASS-imports-and-CLI-help-only' if len(result)==len(names) and all(x['returncode']==0 for x in result) else 'FAIL',first=first,finished=time.time(),interfaces=result),f,indent=2)
if any(x['returncode'] for x in result):
    raise SystemExit(1)

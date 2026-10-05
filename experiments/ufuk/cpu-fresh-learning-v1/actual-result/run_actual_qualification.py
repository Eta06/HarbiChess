import hashlib,json,os,signal,subprocess,time,shutil
from pathlib import Path
BASE=Path('/workspace/work/harbichess/cpu-fresh-learning-v1-actual')
ROOT=Path('/workspace/HarbiChess'); STAGE=ROOT/'experiments/ufuk/cpu-fresh-learning-v1'
CHECKOUT=Path('/workspace/work/harbichess/cpu-additive-source-6fcc8b4')
REG=json.loads((BASE/'qualification-registration.json').read_text());END=REG['deadline_epoch'];FIRST=REG['first_epoch']
assert FIRST<=time.time()<END and END-FIRST==600 and END<=1791273600
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(BASE/'qualification-protocol.json')==REG['protocol_sha256']
assert sha(STAGE/'train.py')==REG['trainer_sha256'] and sha(STAGE/'qualify.py')==REG['qualifier_sha256']
os.sched_setaffinity(0,{3}); RAM=Path('/dev/shm/harbichess-fresh-MC-native-proof-20261005');RAM.mkdir(exist_ok=False)
ROWS=[]; child=None
ENV={**os.environ,'PYTHONPATH':str(CHECKOUT/'src'),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
def guard():
 if time.time()>=END:raise TimeoutError('original-shared600-clock-expired')
 if shutil.disk_usage('/workspace').free<REG['workspace_floor']:raise RuntimeError('workspace-floor-breached')
 if sum(p.stat().st_size for p in RAM.rglob('*') if p.is_file())>REG['RAMstage_limit_bytes']:raise RuntimeError('sealed-RAM-cap-breached')
try:
 for seed in REG['seeds']:
  guard();command=[str(ROOT/'.venv/bin/python'),str(STAGE/'qualify.py'),'--checkout',str(CHECKOUT),'--output',str(RAM/str(seed)),'--weights','/workspace/work/harbichess/a100/restoration/local-rehearsal-content/harbichess-inputs/initial-e8.safetensors','--protocol',str(BASE/'qualification-protocol.json'),'--journal',f'/dev/shm/harbichess-fresh-E0-{seed}/actions-00002048.json.gz','--actor-config',f'/workspace/work/harbichess/cpu-fresh-selfplay-v2-actual/registration/{seed}-E0-actor-config.json','--journal-helper',str(ROOT/'experiments/ufuk/cpu-fresh-selfplay-v2/journal_v2.py'),'--feature-helper',str(STAGE/'features.py'),'--source-commit','6fcc8b476d25495d1c9c413e55b2c7ba4794013e','--seed',str(seed),'--first-epoch',str(FIRST),'--deadline-epoch',str(END)]
  (BASE/f'{seed}-qualification-command.json').write_text(json.dumps(command,indent=2)+'\n')
  with (BASE/f'{seed}-qualification.stdout.log').open('x') as out,(BASE/f'{seed}-qualification.stderr.log').open('x') as err:
   child=subprocess.Popen(command,cwd=CHECKOUT,env=ENV,stdout=out,stderr=err,start_new_session=True)
   while child.poll() is None:guard();time.sleep(.5)
  code=child.returncode;child=None
  if code:raise RuntimeError(f'qualification-failed-seed-{seed}-code-{code}')
  path=RAM/str(seed)/'result.json';r=json.loads(path.read_text());assert r['status']=='PASS-native-restart-not-strength';ROWS.append({'seed':seed,'result_sha256':sha(path),'result':r});print('PASS',seed,flush=True)
 status='PASS-both-realdata-whole8-pause4-freshresume8-all6native-perseed-NOT-strength'
except BaseException as exc:
 status='failed-preserved';error=repr(exc)
finally:
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=5)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
r={'status':status,'rows':ROWS,'original_first_epoch':FIRST,'original_deadline_epoch':END,'finished_epoch':time.time(),'GPU_used':False,'strength_success':False,'RAMstage':str(RAM),'controller_sha256':sha(Path(__file__))}
if status=='failed-preserved':r['error']=error
(BASE/'qualification-cohort-result.json').write_text(json.dumps(r,indent=2)+'\n');print(status,flush=True)

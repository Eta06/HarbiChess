from pathlib import Path
import json,hashlib,subprocess,tarfile,io,sys,shlex
R=Path('/workspace/HarbiChess');S=Path('/workspace/work/harbichess/source8-actual-qualification');H=R/'experiments/ufuk/own-terminal-visited-loss/frozen-helper-v1';Q=R/'experiments/ufuk/own-terminal-visited-loss/actual-qualification-v1';source='3be5b87db27a0fbde83464e7ea7157f0d9a76ae4';remote='/content/harbichess-fullgame-method2-inputs/source8-3be5-actual-qualification-v1';repo='/content/HarbiChess-visited-loss-3be5b87';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
bundle=S/'source.bundle';assert not bundle.exists();subprocess.run(['git','bundle','create',str(bundle),'work'],cwd=R,check=True)
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as tf:
 tf.add(bundle,arcname='source.bundle')
 for folder in [H,Q]:
  for p in folder.iterdir():
   if p.is_file():tf.add(p,arcname=p.name)
body=buf.getvalue();digest=hashlib.sha256(body).hexdigest();cfgsha=sha(Q/'unit-config.json')
code=f'''import sys,io,tarfile,hashlib,subprocess,os,time,json
from pathlib import Path
b=sys.stdin.buffer.read();assert hashlib.sha256(b).hexdigest()=={digest!r};p=Path({remote!r});p.mkdir(exist_ok=False)
with tarfile.open(fileobj=io.BytesIO(b),mode='r:gz') as tf:
 assert all(m.isfile() and len(Path(m.name).parts)==1 for m in tf.getmembers());tf.extractall(p,filter='data')
source={source!r};repo=Path({repo!r});assert not repo.exists()
subprocess.run(['git','clone','--shared','--no-checkout','/content/HarbiChess-certificate-c022bc1',str(repo)],check=True,capture_output=True)
subprocess.run(['git','bundle','unbundle',str(p/'source.bundle')],cwd=repo,check=True,capture_output=True)
subprocess.run(['git','checkout','--detach',source],cwd=repo,check=True,capture_output=True)
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==source;assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();cfg=json.loads((p/'unit-config.json').read_text());assert sha(p/'unit-config.json')=={cfgsha!r}
for path,h in cfg['immutable_inputs'].items():assert sha(path)==h,path
protocol=json.loads((p/'protocol.json').read_text())
for path,h in protocol['fixed_inputs_SHA256'].items():assert sha(p/path)==h,path
for name in ['own8_profile_e1_owned.py','own8_qualify_e1.py','qualify_source_cli.py']:
 subprocess.run(['/usr/bin/python3',str(p/name),'--help'],cwd=p,check=True,capture_output=True,timeout=30)
assert time.time()+900<1791180000
unitargv=['/usr/bin/python3',str(p/'unit_owner_v3.py'),'--config',str(p/'unit-config.json'),'--config-sha256',{cfgsha!r}]
first=time.time();deadline=first+600
cliargv=['/usr/bin/python3',str(p/'qualify_source_cli.py'),'--checkout',str(repo),'--output','/content/harbichess-runs/visited-loss-source8-clean-CLI600-20261005','--weights','/content/harbichess-inputs/initial-e8.safetensors','--book','/content/harbichess-fullgame-method2-inputs/own-terminal-curriculum-preflight-20261005/book.json','--config',str(p/'tiny-config.json'),'--protocol',str(p/'protocol.json'),'--source-commit',source,'--started-epoch',str(first),'--deadline-epoch',str(deadline),'--memory-max-bytes',str(64*1024**3),'--disk-min-free-bytes',str(8*1024**3),'--expected-ledger-schema','pre-action-masked-search-behavior-v4','--execute']
rows=[]
for name,argv in [('unit',unitargv),('CLI',cliargv)]:
 with (p/(name+'-owner.stdout')).open('x') as out,(p/(name+'-owner.stderr')).open('x') as err:owner=subprocess.Popen(argv,cwd=p,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
 f=Path(f'/proc/{{owner.pid}}/stat').read_text().rsplit(')',1)[1].split();row={{'role':name,'pid':owner.pid,'startticks':int(f[19]),'pgid':os.getpgid(owner.pid),'argv':argv,'observed_epoch':time.time(),'original_CLI_first_epoch':first,'original_CLI_deadline_epoch':deadline}};rows.append(row)
 (p/(name+'-launch-owner.json')).write_text(json.dumps(row,indent=2)+'\\n')
receipt={{'status':'source8-original-independent600-unit95-and-CLI-launched','source_commit':source,'archive_sha256':{digest!r},'config_sha256':{cfgsha!r},'owners':rows,'no_budget_reset':True}}
(p/'qualification-launch.json').write_text(json.dumps(receipt,indent=2)+'\\n');print(json.dumps(receipt))'''
sys.path.insert(0,'/workspace/work/harbichess/a100');from ssh_colab_access import call
response=call('python3 -c '+shlex.quote(code),input=body,capture_output=True,timeout=120)
res={'returncode':response.returncode,'stdout':response.stdout.decode(),'stderr':response.stderr.decode()};(S/'qualification-launch-actual.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res));assert response.returncode==0

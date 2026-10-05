from pathlib import Path
import sys,shlex,json,hashlib
sys.path.insert(0,'/workspace/work/harbichess/a100');from ssh_colab_access import call
code='''import json,time,hashlib,subprocess,os,sys
from pathlib import Path
p=Path('/content/harbichess-fullgame-method2-inputs/source8-3be5-actual-qualification-v1');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();unit=Path('/content/harbichess-runs/visited-loss-source8-unit-CUDA-owner-20261005/result.json');cli=Path('/content/harbichess-runs/visited-loss-source8-clean-CLI600-20261005');repo=Path('/content/HarbiChess-visited-loss-3be5b87');source='3be5b87db27a0fbde83464e7ea7157f0d9a76ae4'
sys.path.insert(0,str(p));from own8_infrastructure_evidence import check_cli_receipt,check_unit_receipt
q=json.loads((cli/'result.json').read_text());check_cli_receipt(q,cli,source,sha);cases=json.loads((p/'unit-case-inventory.json').read_text());check_unit_receipt(json.loads(unit.read_text()),source,cases)
protocol=json.loads((p/'protocol.json').read_text())
for name,digest in protocol['fixed_inputs_SHA256'].items():assert sha(p/name)==digest,name
cfg=json.loads((p/'unit-config.json').read_text())
for path,digest in cfg['immutable_inputs'].items():assert sha(path)==digest,path
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==source;assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
inputs={'weights':'/content/harbichess-inputs/initial-e8.safetensors','book':'/content/harbichess-fullgame-method2-inputs/own-terminal-curriculum-preflight-20261005/book.json','config':str(p/'full-config.json'),'protocol':str(p/'protocol.json')}
manifest=p/'full-input-manifest.json'
with manifest.open('x') as stream:json.dump({k:sha(v) for k,v in inputs.items()},stream,indent=2,sort_keys=True);stream.write('\\n')
argv=['/usr/bin/python3',str(p/'own8_profile_e1_owned.py'),'--checkout',str(repo),'--output','/content/harbichess-runs/visited-loss-source8-fullshape900-20261005','--python','/usr/bin/python3']
for key,value in inputs.items():argv+=['--'+key,value]
for key,path in {'input-manifest':manifest,'qualification-receipt':cli/'result.json','unit-receipt':unit,'unit-case-inventory':p/'unit-case-inventory.json','audit-helper':p/'own8_qualify_e1.py'}.items():argv+=['--'+key,str(path),'--'+key+'-sha256',sha(path)]
argv+=['--qualification-run',str(cli),'--source-commit',source]
for flag,name in {'audit-core':'own8_audit_core.py','common-helper':'own8_audit_support.py','adapter-controls':'own8_adapter_controls.py','infrastructure-evidence':'own8_infrastructure_evidence.py','cli-helper':'qualify_source_cli.py'}.items():argv+=['--'+flag+'-sha256',sha(p/name)]
assert time.time()+900<1791180000;first=time.time();deadline=first+900;argv+=['--deadline-epoch',str(deadline)]
with (p/'fullprofile-owner.stdout').open('x') as out,(p/'fullprofile-owner.stderr').open('x') as err:owner=subprocess.Popen(argv,cwd=p,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
f=Path(f'/proc/{owner.pid}/stat').read_text().rsplit(')',1)[1].split();receipt={'status':'source8-original900-real128x256-K8-18RAW-6mutation-winning-and-visited-loss-profile-launched','source_commit':source,'pid':owner.pid,'startticks':int(f[19]),'pgid':os.getpgid(owner.pid),'first_observed_epoch':first,'absolute_original_deadline_epoch':deadline,'argv':argv,'qualification_unit95_sha256':sha(unit),'qualification_CLI600_sha256':sha(cli/'result.json'),'protocol_sha256':sha(p/'protocol.json'),'input_manifest_sha256':sha(manifest),'original900_budget_reset':False}
with (p/'fullprofile-launch-owner.json').open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\\n')
print(json.dumps(receipt))'''
r=call('python3 -c '+shlex.quote(code),capture_output=True,text=True,timeout=75);v={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr};p=Path('/workspace/work/harbichess/source8-actual-qualification/fullprofile-v2-launch-actual.json');assert not p.exists();p.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v));assert r.returncode==0

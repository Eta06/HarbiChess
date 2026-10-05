import sys,shlex,json
from pathlib import Path
sys.path.insert(0,'/workspace/work/harbichess/a100');from ssh_colab_access import call
code='''from pathlib import Path
import json,time
q=Path('/content/harbichess-fullgame-method2-inputs/source8-3be5-actual-qualification-v1');o={'observed_epoch':time.time()}
for k,p in {'unit':Path('/content/harbichess-runs/visited-loss-source8-unit-CUDA-owner-20261005'),'CLI':Path('/content/harbichess-runs/visited-loss-source8-clean-CLI600-20261005'),'profile':Path('/content/harbichess-runs/visited-loss-source8-fullshape900-20261005')}.items():
 if (p/'result.json').exists():
  v=json.loads((p/'result.json').read_text());o[k]={x:v[x] for x in v if x in ['status','error','started_epoch','finished_epoch','absolute_deadline_epoch','tests_passed','tests_skipped','tests_failed','torch_version','actualCUDA','device_name','whole_seconds','source_commit']}
 else:o[k]={'result_exists':False}
 if (p/'stdout.log').exists():o[k]['stdout_tail']=(p/'stdout.log').read_text()[-900:]
 for name in [k+'-owner.stderr',k+'-owner.stdout']:
  if (q/name).exists():o[k][name]=(q/name).read_text()[-900:]
print(json.dumps(o))'''
r=call('python3 -c '+shlex.quote(code),capture_output=True,text=True,timeout=60);v={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr};p=Path('/workspace/work/harbichess/source8-actual-qualification');(p/('observed-'+str(__import__('time').time_ns())+'.json')).write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))

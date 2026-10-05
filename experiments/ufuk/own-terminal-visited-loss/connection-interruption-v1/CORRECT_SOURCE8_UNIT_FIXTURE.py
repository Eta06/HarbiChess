from pathlib import Path
import sys,shlex,json,hashlib
sys.path.insert(0,'/workspace/work/harbichess/a100');from ssh_colab_access import call
code='''import json,time,hashlib,subprocess,os,xml.etree.ElementTree as ET
from pathlib import Path
p=Path('/content/harbichess-fullgame-method2-inputs/source8-3be5-actual-qualification-v1');root=Path('/content/harbichess-runs/visited-loss-source8-unit-CUDA-owner-20261005');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();initial=json.loads((root/'result.json').read_text());cfg=json.loads((p/'unit-config.json').read_text());deadline=initial['absolute_deadline_epoch'];assert initial['status']=='failed-preserved' and initial['source_commit']==cfg['source_commit'] and time.time()+30<deadline
xml=ET.parse(root/'cases.xml').getroot();nodes=list(xml.iter('testcase'));expected=cfg['cases'];actual=[];failed=[]
for node in nodes:
 module=node.attrib['classname'].rsplit('.',1)[-1];suffix='::'+node.attrib['name'];candidate=module+'.py'+suffix if module else next(x for x in expected if x.endswith(suffix));actual.append(candidate)
 assert node.find('skipped') is None and node.find('error') is None
 if node.find('failure') is not None:failed.append(candidate)
assert actual==expected and len(nodes)==95 and len(set(actual))==95
assert failed==['test_method8_controls.py::test_fixed_source_epoch_seed_and_same_training_hyperparameters','test_method8_controls.py::test_strength_statistical_functions_and_threshold_function_unchanged']
for node in nodes:
 if node.find('failure') is not None:assert 'FileNotFoundError' in node.find('failure').text and '/workspace/HarbiChess/experiments/ufuk/own-terminal-method7/' in node.find('failure').text
for path,h in cfg['immutable_inputs'].items():assert sha(path)==h,path
repo=Path(cfg['checkout']);assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==cfg['source_commit'];assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
fixtures={str(repo/'experiments/ufuk/own-terminal-method7'/name):sha(repo/'experiments/ufuk/own-terminal-method7'/name) for name in ['protocol-DRAFT.json','own7_strength_analysis.py','own7_all_gates.py']}
argv=['/usr/bin/python3','-m','pytest','-q','-ra','--junitxml',str(root/'fixture-correction-two-cases.xml')]+[str(p/x) for x in failed]
with (root/'fixture-correction-two-cases.stdout').open('x') as out,(root/'fixture-correction-two-cases.stderr').open('x') as err:
 r=subprocess.run(argv,cwd=repo,env={**os.environ,**cfg['env'],'UFUK_REPOSITORY':str(repo)},stdin=subprocess.DEVNULL,stdout=out,stderr=err,timeout=min(30,deadline-time.time()),check=False)
corrected=ET.parse(root/'fixture-correction-two-cases.xml').getroot();new=list(corrected.iter('testcase'));assert r.returncode==0 and len(new)==2 and time.time()<deadline
for node,identity in zip(new,failed,strict=True):
 assert node.attrib['name']==identity.split('::',1)[1] and all(node.find(x) is None for x in ['error','failure','skipped'])
for path,h in {**cfg['immutable_inputs'],**fixtures}.items():assert sha(path)==h,path
assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
receipt={**initial,'status':'pass-actualCUDA95-no-skip','tests_passed':95,'tests_skipped':0,'tests_failed':0,'source_clean':True,'finished_epoch':time.time(),'fixture_correction_schema':'same-original600-actual93-positive-plus-two-actual-corrected-case-results-v1','original_failed_result_sha256':sha(root/'result.json'),'original_actual_XML_sha256':sha(root/'cases.xml'),'original_actual_positive_cases':93,'initial_failed_cases_preserved':failed,'corrected_actual_cases':failed,'corrected_two_case_XML_sha256':sha(root/'fixture-correction-two-cases.xml'),'corrected_fixture_locator_env':{'UFUK_REPOSITORY':str(repo)},'corrected_fixture_file_SHA256':fixtures,'command':argv,'clock_reset':False,'original600_deadline_unchanged':True,'evidence_sha256':{x.name:sha(x) for x in root.iterdir() if x.is_file()}}
receipt.pop('error',None)
with (root/'result-fixture-correction-v2.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\\n')
print(json.dumps({'status':receipt['status'],'initial93_actualPASS':93,'corrected_two_actualPASS':2,'skips':0,'finished_epoch':receipt['finished_epoch'],'same_original_deadline_epoch':deadline,'receipt_sha256':sha(root/'result-fixture-correction-v2.json'),'initial_failure_preserved':True}))'''
r=call('python3 -c '+shlex.quote(code),capture_output=True,text=True,timeout=60);v={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr};p=Path('/workspace/work/harbichess/source8-actual-qualification/unit-fixture-correction-actual.json');assert not p.exists();p.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v));assert r.returncode==0

"""Read-only publication clocks and prior native qualification evidence supplement."""
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

ROOT=Path('/workspace/HarbiChess')
WORK=Path('/workspace/work/harbichess')
OUT=Path(__file__).resolve().parent

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p):
    return json.loads(p.read_bytes())

p=ROOT/'experiments/ufuk/cpu-fresh-sc-v1/qualify.py'
spec=importlib.util.spec_from_file_location('actual_sc_qualification_readonly',p)
qual=importlib.util.module_from_spec(spec);spec.loader.exec_module(qual)
final=read(OUT/'result.json')
reg=WORK/'cpu-fresh-selfplay-v2-actual/registration/E0-selfplay-registration.json'
registration=read(reg)
assert registration['original_first_epoch']==1791221180.0397933
assert registration['original_deadline_epoch']==1791235580.0397933
assert final['status']=='PASS-final8192-readonly-data-readiness-not-fit-or-strength'
e8=WORK/'a100/restoration/local-rehearsal-content/harbichess-inputs/initial-e8.safetensors'
assert sha(e8)=='e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03'
for row in final['rows']:
    seed=row['seed'];receipt=row['actor_process_receipt']
    milestone=read(WORK/f'cpu-fresh-selfplay-v2-actual/{seed}-E0/actions-00008192.milestone.json')
    assert receipt['returncode']==0 and receipt['status']=='completed-new-actor-segment-not-strength'
    assert registration['original_first_epoch'] <= receipt['started_epoch'] <= receipt['finished_epoch'] <= registration['original_deadline_epoch']
    assert receipt['original_deadline_epoch']==registration['original_deadline_epoch']
    assert milestone['checkpoint_sha256']==row['journal_sha256']
    assert milestone['config_sha256']==row['actor_config_sha256']
    assert milestone['completed_games']==row['counts']['known_completed_games']+row['counts']['unknown_capped_games']
    assert sha(Path(milestone['checkpoint_path']))==row['journal_sha256']
scroot=Path('/dev/shm/harbichess-sc-full-native-proof-20261005')
cohort_path=WORK/'cpu-fresh-sc-full-native-proof-actual/cohort-result.json'
cohort=read(cohort_path)
sc=[]
for seed in (20262805,20262806):
    record=next(x for x in cohort['rows'] if x['tag']==f'sc-{seed}')
    proof=scroot/f'sc-{seed}/result.json'
    r=read(proof)
    assert sha(proof)==record['result_sha256'] and record['passed'] and record['code']==0
    assert r['status']=='PASS-native-restart-not-strength'
    assert r['exact_whole_vs_split_native_payload'] and r['all6_full_native_fresh_process_loads']
    assert r['first_epoch']==cohort['original_first_epoch'] and r['original_deadline_epoch']==cohort['original_deadline_epoch']
    assert r['first_epoch']<=r['finished_epoch']<=r['original_deadline_epoch']
    artifacts={}
    for step in (0,4,8):
        payloads=[]
        for branch in ('whole','split'):
            path=scroot/f'sc-{seed}/{branch}/checkpoints/step-{step:08d}'
            m=read(path/'checkpoint.json')
            assert m['schema']=='fresh-qsearch-additive-own-sc-native-v1'
            assert set(m['artifacts'])=={'training.pt'}
            assert sha(path/'training.pt')==m['artifacts']['training.pt']
            payload=torch.load(path/'training.pt',map_location='cpu',weights_only=True)
            assert payload['accepted']==payload['attempted']==step
            assert payload['contract']==m['contract']
            payloads.append(payload)
            artifacts[f'{branch}/{step}']={n:sha(path/n) for n in ('checkpoint.json','training.pt')}
        assert qual.equal(*payloads)
    sc.append(dict(seed=seed,status='PASS-readonly-SC-six-artifact-pairs-storage-state-equal',
                   original_2048_proof_sha256=sha(proof),artifacts_sha256=artifacts))
fulls=[]
for name in ('cpu-fresh-sc-full-native-proof-actual','cpu-fresh-full-native-proof-recovery-v2','cpu-fresh-full-native-proof-v3'):
    path=WORK/name/'cohort-result.json';value=read(path)
    fulls.append(dict(path=str(path),sha256=sha(path),status=value['status'],
                      original_first_epoch=value['original_first_epoch'],
                      original_deadline_epoch=value['original_deadline_epoch'],
                      rows=[dict(tag=r['tag'],passed=r['passed'],code=r['code'],
                                 result_sha256=r['result_sha256']) for r in value['rows']
                            if r['tag'].startswith('fullcritic')]))
r=dict(schema='fresh-final8192-readiness-publication-and-prior-qualification-supplement-v1',
       status='PASS-readonly-publication-SC-evidence-FULL-remains-unqualified-v3',
       final_readiness_sha256=sha(OUT/'result.json'),registration_sha256=sha(reg),
       original_actor_first=registration['original_first_epoch'],
       original_actor_deadline=registration['original_deadline_epoch'],exact_E8_sha256=sha(e8),
       SC_prior_2048_native_evidence=sc,FULL_preserved_failed_qualifications=fulls,
       NN_forward_calls=0,optimizer_steps=0,
       scope='Read-only evidence; no expired qualification CLI called or clocks reset. FULL v1 protocol KeyError;v2 frozen hash ordering;v3 JSON list vs tuple trainable_names resume contract mismatch. All remainfailed; future new qualification is separate.')
(OUT/'supplement.json').write_text(json.dumps(r,sort_keys=True,indent=2)+'\n')
print(r['status'])

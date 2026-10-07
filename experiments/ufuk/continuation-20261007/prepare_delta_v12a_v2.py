"""ROOT freeze CLOSED teacher-bootstrapped candidate states and games; no hot actors or keys."""
import hashlib
import importlib.util
import json
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / 'nnue-state-v12a'
RAM = Path('/dev/shm')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def publish(p, j):
    with p.open('x') as f:
        json.dump(j,f,sort_keys=True,indent=2,allow_nan=False)


def main():
    spec=importlib.util.spec_from_file_location('delta_codec',OUT/'raw_capsule_v12a.py')
    codec=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codec)
    selected={}

    def include(p,role):
        files=sorted(p.rglob('*')) if p.is_dir() else [p]
        for f in files:
            if f.is_file() and not f.is_symlink() and not any(x.startswith('.') or x=='__pycache__' for x in f.relative_to(p if p.is_dir() else p.parent).parts):
                selected[str(f)]=(role,f.stat().st_size,sha(f))
    groups=['harbichess-NNUE-afterstate-data-v1','harbichess-NNUE-afterstate-proof-v1',
            'harbichess-NNUE-afterstate-fit-v1','harbichess-NNUE-ranking-data-v1',
            'harbichess-NNUE-ranking-corrective-proof-v3','harbichess-NNUE-ranking-corrective-fit-v3',
            'harbichess-NNUE-closedterminal-corrective-data-v3',
            'harbichess-NNUE-closedterminal-integrated-proof-v4',
            'harbichess-NNUE-closedterminal-integrated-fit-v4','harbichess-closedterminal-v1',
            'harbichess-pv-human-profile-v2','harbichess-pv-human-arena-v2']
    for group in groups:
        include(RAM/group,'closed-actual-native-data-trace-'+group)
    for group in ['tdlambda-known160-data','afterstate-known160-data']:
        include(RAM/'harbichess-continuation-20261007'/group,'closed-actual160-games-profile')
    dirs=['afterstate-actual-20262905','afterstate-actual-20262906','afterstate-root-independent-audit',
          'tdlambda-root-independent-audit','pv-independent-audit','pv-human-profile-v2','pv-human-arena-v2',
          'ranking-actual-20262905','ranking-recovery-actual-20262905',
          'ranking-corrective-v3-actual-20262905','ranking-corrective-v3-actual-20262906',
          'closedterminal-collection-20262905','closedterminal-collection-20262906',
          'closedterminal-actual-20262906','closedterminal-recovery-actual-20262906',
          'closedterminal-corrective-v3-actual-20262906','closedterminal-integrated-v4-actual-20262905',
          'closedterminal-integrated-v4-actual-20262906',
          'action-ranking-own-v1','action-ranking-own-v2-source-binding','afterstate-own-v1',
          'closed-terminal-own-v1','closed-terminal-own-v2-reconciliation',
          'closed-terminal-own-v3-producer-converter-binding','closed-terminal-own-integration-factory-v4',
          'variant-known160-v3-action-ranking','variant-known160-v4-MC',
          'MC-known160-runtime','ranking-known160-runtime']
    for group in dirs:
        include(BASE/group,'closed-source-spec-registration-failure-'+group)
    for name in ['MC-actual-audit-set.json','ownq-actual-audit-set.json','execute_ranking_chain_v3.py',
                 'execute_closedterminal_chain_v3.py','execute_closedterminal_chain_v4.py',
                 'execute_afterstate_known160_audit.py','execute_TD_known160_audit.py','prepare_delta_v12a.py']:
        include(BASE/name,'ROOT-closed-boundary-or-source')
    rows=[dict(path=p,member=f'files/{i:04}',role=role,bytes=n,sha256=h)
          for i,(p,(role,n,h)) in enumerate(sorted(selected.items()))]
    manifest=dict(schema=codec.SCHEMA,approval='ROOT-approved-exact-files',
                  limits=dict(files=2048,raw_bytes=codec.RAW_LIMIT,encoded_bytes=codec.ENCODED_LIMIT,
                              file_bytes=codec.FILE_LIMIT,header_bytes=1024*1024),
                  rows=rows,row_count=len(rows),raw_bytes=sum(r['bytes'] for r in rows),
                  dependencies=['V10b actual parent/source/protected graph','V11 actual TDnative/oldownQ160','core6fcc exact Git source'],
                  hot_actors_excluded=True,live_MC_ranking_games_excluded=True,
                  teacher_free_zero_bank_states_separate_deltaB=True,
                  scope='CLOSED negative arenas and new actual afterstate/rank/MC native data/proofs/Adam/RNG/replay; not strength')
    codec.checked_rows(manifest)
    first=time.time()
    clock=dict(schema='NNUE-own-Oct7-deltaA-release-clock-v12a',status='ROOT-approved-frozen-transport',
                               started_epoch=first,deadline_epoch=first+5400,operator_end_epoch=1791448916.685839)
    publish(OUT/'manifest.json',manifest)
    publish(OUT/'clock.json',clock)
    print(json.dumps(dict(files=len(rows),raw_bytes=manifest['raw_bytes'],max_file=max(r['bytes'] for r in rows),first=first,deadline=first+5400)))


if __name__ == '__main__':
    main()

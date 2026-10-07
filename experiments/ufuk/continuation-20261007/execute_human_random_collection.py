"""ROOT teacher-free procedural bank and current-parent actor phases; immutable independent clocks."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
STAGE = BASE / 'human-randomstarts-own-v2'
END = 1791448916.685839
PROTECTED = Path('/dev/shm/harbichess-continuation-20261007/metadata-20262906/protected-aliases.bin')
INFERENCE = Path('/workspace/work/harbichess/cpu-kingbucket-nnue-proposal')


def ref(p):
    return dict(path=str(p), sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())


def write(p, j):
    with p.open('x') as f:
        json.dump(j, f, indent=2, allow_nan=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, choices=[20262905, 20262906], required=True)
    p.add_argument('--cpu-core', type=int, choices=[1, 3], required=True)
    p.add_argument('--phase', choices=['bank', 'collect'], required=True)
    a = p.parse_args()
    os.sched_setaffinity(0, {a.cpu_core})
    records = BASE / f'human-random-{a.phase}-actual-{a.seed}'
    records.mkdir(exist_ok=False)
    bank = Path(f'/dev/shm/harbichess-human-randomstarts-bank-v2/{a.seed}')
    first = time.time()
    duration = 600 if a.phase == 'bank' else 7200
    deadline = min(first + duration, END)
    record = dict(schema='ROOT-teacher-free-procedural-phase-v2', status='running', phase=a.phase,
                  seed=a.seed, cpu_core=a.cpu_core, first=first, deadline=deadline,
                  helper=ref(__file__), source_inventory=ref(STAGE/'source-inventory.json'),
                  protected_aliases=ref(PROTECTED), commands=[], teacher_labels_used=False,
                  trained_teacher_weights_used=False, outcome_selection=False)
    write(records/'registration.json', record)

    def run(name, cmd):
        with (records/(name+'.stdout.log')).open('xb') as out, (records/(name+'.stderr.log')).open('xb') as err:
            r = subprocess.run(cmd, stdout=out, stderr=err, timeout=max(.001,deadline-time.time()))
        record['commands'].append(dict(name=name, command=cmd, returncode=r.returncode,
                                       finished=time.time(), stdout=ref(records/(name+'.stdout.log')),
                                       stderr=ref(records/(name+'.stderr.log'))))
        if r.returncode:
            raise RuntimeError('owned phase failed '+name)
        if time.time() >= deadline:
            raise TimeoutError('original immutable phase clock')
    clock = ['--first',str(first),'--deadline',str(deadline),'--operator-end-epoch',str(END)]
    try:
        if a.phase == 'bank':
            reg = records/'bank-registration.json'
            run('prepare-bank',[sys.executable,str(STAGE/'prepare_bank.py'),'--seed',str(a.seed),*clock,
                                '--protected-aliases',str(PROTECTED),'--output',str(reg)])
            run('generate-bank',[sys.executable,str(STAGE/'root_bank.py'),'--registration',str(reg),'--output',str(bank)])
            record.update(status='PASS-actual-rule-only-procedural-bank-not-strength', receipt=ref(bank/'receipt.json'))
        else:
            parent = BASE / f'human-zero-actual-{a.seed}'
            passed = json.loads((parent/'result.json').read_bytes())
            if passed['status'] != 'PASS-actual-zero-init-admission-and-synthetic-native-proof-not-strength':
                raise ValueError('literalzero native qualification must actually PASS')
            seal = records/'collection-build-seal.json'
            reg = records/'collection-registration.json'
            pool = Path(f'/dev/shm/harbichess-human-randomstarts-bank-v2/{a.seed}/collection-pool.json')
            run('prepare-collection',[sys.executable,str(STAGE/'prepare_collection.py'),*clock,
                 '--cpu-core',str(a.cpu_core),'--admission-seal',str(parent/'parent-admission-seal.json'),
                 '--admission-result',str(parent/'parent-admission-result.json'),
                 '--inference-directory',str(INFERENCE),'--extension',str(INFERENCE/'_kingbucket16.cpython-312-x86_64-linux-gnu.so'),
                 '--procedural-bank-receipt',str(bank/'receipt.json'),'--protected-aliases',str(PROTECTED),
                 '--root-pool-output',str(pool),'--output',str(seal)])
            run('metadata-factory',[sys.executable,str(STAGE/'metadata_factory.py'),'--seal',str(seal),'--output',str(reg)])
            run('collect-own',[sys.executable,str(STAGE/'run_collection.py'),'--registration',str(reg)])
            output = Path(json.loads(reg.read_bytes())['output_path'])
            record.update(status='PASS-actual-procedural-own-Q-collection-not-strength', collection_registration=ref(reg), receipt=ref(output/'receipt.json'))
    except BaseException as error:
        record.update(status='FAILED-preserved', error=repr(error))
        raise
    finally:
        record['finished'] = time.time()
        write(records/'result.json',record)


if __name__ == '__main__':
    main()

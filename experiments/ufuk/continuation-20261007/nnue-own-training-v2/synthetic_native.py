"""Single <=60s CPU4 synthetic-data integration; immutable teacher weights only."""
import argparse
import json
import time
from pathlib import Path

from convert import canonical, sha
from native import MATH
from prove import execute

HERE = Path(__file__).resolve().parent


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    first = time.time()
    a.output.mkdir(parents=True, exist_ok=False)
    parent = Path('/dev/shm/harbichess-NNUE-teacher-fit-v1/20262905/whole/candidate.pt')
    data = a.output / 'synthetic-data.json'
    data.write_bytes(canonical(dict(schema='own-kingbucket-sparse-training-data-v1',
                                   phase='own-learning', rows=[dict(indices=[0, 2, 8],
                                                                  prior_logit=0.1, target=0.2)])))
    provenance = a.output / 'synthetic-provenance.json'
    provenance.write_bytes(canonical(dict(synthetic_only=True, actual_own_labels_used=False)))
    original = json.loads(Path('/workspace/work/harbichess/continuation-20261005/'
                               'NNUE-teacher-fit-v1/20262905-contract.json').read_bytes())
    c = dict(phase='own-learning', updates=64, seed=20262905, math=MATH,
             feature_schema=original['feature_schema'], dataset_sha256=sha(data),
             source_sha256={str(HERE / n): sha(HERE / n)
                            for n in ['model.py', 'native.py', 'train.py']},
             execution_helpers_sha256={str(HERE / n): sha(HERE / n)
                                       for n in ['convert.py', 'contracts.py', 'prove.py']},
             execution_scope_schema='NNUE-own-execution-contract-v2', execution_mode='proof',
             operator_end_epoch=first + 60, original_first_epoch=first,
             original_deadline_epoch=first + 60, bootstrap_candidate_path=str(parent),
             bootstrap_candidate_sha256=sha(parent), target_provenance_path=str(provenance),
             target_provenance_sha256=sha(provenance),
             prior_helper_path=original['prior_helper_path'],
             prior_helper_sha256=original['prior_helper_sha256'],
             inference_source_sha256=original['inference_source_sha256'],
             core_source_repo=original['core_source_repo'],
             core_source_commit=original['core_source_commit'], synthetic_only=True)
    contract = a.output / 'contract.json'
    contract.write_bytes(canonical(c))
    reg = dict(schema='NNUE-own-training-orchestration-v2', status='registered', mode='proof',
               first=first, deadline=first + 60, operator_end_epoch=first + 60, cpu_core=4,
               seed=20262905, output=str(a.output / 'proof'), inputs={}, source_sha256={},
               parent_candidate=dict(path=str(parent), sha256=sha(parent)),
               dataset=dict(path=str(data), sha256=sha(data)),
               contract=dict(path=str(contract), sha256=sha(contract)),
               train=dict(path=str(HERE / 'train.py'), sha256=sha(HERE / 'train.py')),
               native=dict(path=str(HERE / 'native.py'), sha256=sha(HERE / 'native.py')),
               synthetic_only=True)
    rp = a.output / 'registration.json'
    rp.write_bytes(canonical(reg))
    reg['registration_sha256'] = sha(rp)
    execute(reg)

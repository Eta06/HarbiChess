"""Metadata-only MC spec augmentation binding; no native or model imports."""
import copy
import hashlib
import json

import pytest
from admission import mc_original_inputs


def put(path, value):
    path.write_text(json.dumps(value, sort_keys=True))
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def fixture(tmp_path):
    original = dict(schema='NNUE-closedterminal1024-conversion-seal-v1', status='registered',
                    first=10, deadline=610, operator_end_epoch=1000, seed_marker=20262906)
    provenance = {'inputs': original}
    dref = put(tmp_path / 'dataset.json', {'rows': ['fixture']})
    pref = put(tmp_path / 'provenance.json', provenance)
    spec = put(tmp_path / 'spec.json', original)
    parent_ref = put(tmp_path / 'parent.json',
                     {'phase': 'teacher-bootstrap', 'updates': 256, 'seed': 20262906})
    result = {'status': 'PASS-own1024-closed-terminal-fullhistory-conversion-not-strength',
              'first': 10, 'deadline': 610, 'finished': 20,
              'dataset_sha256': dref['sha256'], 'provenance_sha256': pref['sha256']}
    result_ref = put(tmp_path / 'result.json', result)
    augmented = dict(original, conversion_spec_file=spec,
                     conversion_result=result_ref, parent_contract=parent_ref)
    candidate = {'path': 'named-teacher.pt', 'sha256': 'teacher'}
    c = {'seed': 20262906, 'raw_collection_inputs': augmented,
         'parent_candidate_path': candidate['path'], 'parent_candidate_sha256': 'teacher',
         'dataset_sha256': dref['sha256'], 'target_provenance_sha256': pref['sha256']}
    return {'dataset': dref, 'target_provenance': pref}, c, provenance, {
        'contract': parent_ref, 'candidate': candidate}


def test_exact_three_additions_preserve_original_converter_input(tmp_path):
    info, c, trace, parent = fixture(tmp_path)
    assert mc_original_inputs(info, c, trace, parent) == trace['inputs']
    assert len(c['raw_collection_inputs']) == len(trace['inputs']) + 3


@pytest.mark.parametrize('mutation', ['extra', 'missing', 'original', 'spec', 'parent',
                                      'result', 'deadline', 'sha'])
def test_mismatched_augmented_identity_rejected(tmp_path, mutation):
    info, c, trace, parent = fixture(tmp_path)
    raw = c['raw_collection_inputs']
    if mutation == 'extra':
        raw['arbitrary_extra'] = 'unregistered'
    elif mutation == 'missing':
        del raw['conversion_spec_file']
    elif mutation == 'original':
        raw['seed_marker'] = 20262905
    elif mutation == 'spec':
        altered = copy.deepcopy(trace['inputs'])
        altered['seed_marker'] = 20262905
        raw['conversion_spec_file'] = put(tmp_path / 'wrong-spec.json', altered)
    elif mutation == 'parent':
        raw['parent_contract'] = put(tmp_path / 'other-parent.json',
                                    {'phase': 'teacher-bootstrap', 'updates': 256,
                                     'seed': 20262906})
    elif mutation in ('result', 'deadline'):
        result = json.loads((tmp_path / 'result.json').read_text())
        result['dataset_sha256' if mutation == 'result' else 'deadline'] = (
            'wrong' if mutation == 'result' else 611)
        raw['conversion_result'] = put(tmp_path / 'result.json', result)
    else:
        raw['conversion_spec_file']['sha256'] = '0' * 64
    with pytest.raises(ValueError):
        mc_original_inputs(info, c, trace, parent)

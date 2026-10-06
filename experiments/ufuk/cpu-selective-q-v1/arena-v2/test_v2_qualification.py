"""Actual four-input metadata acceptance and fail-closed binding regressions."""

import copy
import json
from pathlib import Path

import pytest
from native_admission import validate_fit_qualifications

HERE = Path(__file__).parent
P = json.loads((HERE / 'protocol-DRAFT.json').read_bytes())


def actual(seed=20262905):
    fit = json.loads(Path('/workspace/work/harbichess/continuation-20261005/'
                         f'selective-q-fit-v1-registration/{seed}-registration.json').read_bytes())
    qualified = P['proof_inputs'][str(seed)]
    proof = json.loads(Path(qualified['registration_path']).read_bytes())
    profile = P['profile_qualification_bindings'][str(seed)]
    return fit, proof, qualified, profile


def test_actual_four_input_fit_accepts_identical_two_input_data_proof():
    for seed in [20262905, 20262906]:
        args = actual(seed)
        assert set(args[0]['inputs']) == {
            'dataset', 'split', 'profile_qualification', 'native_qualification'}
        validate_fit_qualifications(*args)
        native = json.loads(Path(f'/dev/shm/harbichess-selective-q-fit-v1/{seed}/native.json')
                            .read_bytes())
        assert native['contract']['inputs'] == args[0]['inputs']
        assert len(native['contract']['inputs']) == 4


@pytest.mark.parametrize('mutation', ['dataset', 'split', 'native_tag', 'profile_sha',
                                     'drop_input', 'profile_registration'])
def test_actual_binding_changes_rejected(mutation):
    fit, proof, qualified, profile = copy.deepcopy(actual())
    if mutation in ['dataset', 'split']:
        fit['inputs'][mutation]['sha256'] = '0' * 64
    elif mutation == 'native_tag':
        fit['inputs']['native_qualification']['path'] = profile['path']
    elif mutation == 'profile_sha':
        fit['inputs']['profile_qualification']['sha256'] = '0' * 64
    elif mutation == 'drop_input':
        del fit['inputs']['profile_qualification']
    else:
        profile['registration_sha256'] = '0' * 64
    with pytest.raises(ValueError):
        validate_fit_qualifications(fit, proof, qualified, profile)

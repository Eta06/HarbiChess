import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import admission  # noqa: E402
import wiring  # noqa: E402


def test_protocol_uses_exact_h0_parents_and_no_teacher_role():
    q = json.loads((HERE / 'protocol-DRAFT.json').read_bytes())
    assert q['target_variant'] == 'forensic-own-v5-h0'
    assert 'teachers' not in q
    wiring.validate_wiring(q)
    for seed in q['match_seeds']:
        key = str(seed)
        contract = admission.read(q['children'][key]['contract'])
        assert q['models'][key]['parent'] == q['parents'][key]['candidate']
        assert contract['parent_candidate'] == q['parents'][key]['candidate']
        assert contract['parent_generation'] == 0
        assert contract['generation'] == 1
        assert contract['teacher_labels_used_in_own_phase'] is False


def test_actual_proof_fit_contracts_bind_current_proof_and_parent():
    q = json.loads((HERE / 'protocol-DRAFT.json').read_bytes())
    for seed in q['match_seeds']:
        info = q['children'][str(seed)]
        proof = admission.read(info['proof_contract'])
        fit = admission.read(info['contract'])
        admission.compare_contracts(proof, fit, info)
        assert proof['parent_admission_seal'] == q['parents'][str(seed)]['admission_seal']
        assert fit['parent_admission_result'] == q['parents'][str(seed)]['admission_result']


def test_teacher256_cannot_be_substituted_for_h0_parent():
    q = json.loads((HERE / 'protocol-DRAFT.json').read_bytes())
    seed = q['match_seeds'][0]
    info = q['children'][str(seed)]
    contract = admission.read(info['contract'])
    forged = dict(q['parents'][str(seed)])
    forged['candidate'] = dict(path='/forbidden/teacher256.pt', sha256='0' * 64)
    with pytest.raises(ValueError, match='H0 ancestry'):
        admission._h0_parent(contract, forged, seed, None, None)


def test_variant_dispatch_does_not_accept_legacy_teacher_path():
    q = json.loads((HERE / 'protocol-DRAFT.json').read_bytes())
    q['target_variant'] = 'forensic-own-v4'
    assert 'forensic-own-v4' not in admission.VARIANTS
    assert 'forensic-own-v5-h0' in admission.VARIANTS

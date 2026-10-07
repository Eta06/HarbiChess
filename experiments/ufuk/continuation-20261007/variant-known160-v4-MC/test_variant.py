"""Pure source/receipt/sampling wiring tests, no model/search/training jobs."""

import copy
import json
from pathlib import Path

import chess
import pytest
from admission import compare_contracts, td_receipt
from alias_pool import own_aliases, state_alias
from teacher_admission import sha


def contract_fixture():
    c = dict(seed=1, execution_mode='proof', original_first_epoch=10,
             original_deadline_epoch=610, contract_build_seal_sha256='a',
             phase='own-learning-tdlambda-v1', dataset_sha256='d',
             target_provenance_sha256='p', parent_candidate_sha256='parent')
    fit = dict(c, execution_mode='fresh-fit', original_first_epoch=620,
               original_deadline_epoch=2400, contract_build_seal_sha256='b',
               tdlambda_proof_result={'path': 'r', 'sha256': 'h'})
    return c, fit


def test_only_explicit_phase_transition_fields():
    proof, fit = contract_fixture()
    info = dict(proof_result=fit['tdlambda_proof_result'])
    compare_contracts(proof, fit, info)
    for field in ('phase', 'seed', 'dataset_sha256', 'target_provenance_sha256'):
        bad = copy.deepcopy(fit)
        bad[field] = 'changed'
        with pytest.raises(ValueError):
            compare_contracts(proof, bad, info)


def fixture_receipt(tmp_path):
    c, _ = contract_fixture()
    ref = dict(path='c.json', sha256='contract')
    commands, payloads = [], []
    steps = [0, 8, 0, 4, 4, 8]
    for index, step in enumerate(steps):
        payload = tmp_path / f'{index}.pt'
        payload.write_bytes(b'SYNTHETIC-NOT-TORCH')
        log = tmp_path / f'{index}.log'
        log.write_text(json.dumps(dict(status='PASS-strict-native-readonly', step=step)))
        commands.append(dict(command=['python', 'train.py', '--dataset', 'data.json',
                                      '--audit-only', '--resume', str(payload),
                                      '--resume-sha256', sha(payload), '--contract', 'c.json',
                                      '--stop', str(step)], returncode=0, finished=30,
                             log_sha256=sha(log)))
        payloads.append(dict(path=str(payload), sha256=sha(payload), step=step,
                             actual_fresh_process=True, log_path=str(log)))
    commands = [dict(command=['whole'], returncode=0, finished=20),
                dict(command=['pause'], returncode=0, finished=21),
                dict(command=['resume'], returncode=0, finished=22), *commands]
    result = dict(status='PASS-tdlambda-fixed-phase-and-fresh-native-loads-not-strength',
                  mode='proof', seed=1, own_updates=8, contract_sha256='contract',
                  dataset_sha256='d', target_provenance_sha256='p', teacher_labels_used=False,
                  raw_zip_identity_claimed=False, weights_only_initializer={'sha256': 'parent'},
                  first=10, deadline=610, finished=40, full_payload_bits_equal=True,
                  commands=commands, native_payloads=payloads)
    reg = dict(schema='NNUE-own-tdlambda-training-orchestration-v1', mode='proof', seed=1,
               first=10, deadline=610, contract=ref, train={'path': 'train.py'},
               dataset={'path': 'data.json', 'sha256': 'd'},
               target_provenance={'sha256': 'p'}, parent_candidate={'sha256': 'parent'})
    return result, reg, ref, c, steps


def test_actual_fresh_log_and_all_command_inventory(tmp_path):
    args = fixture_receipt(tmp_path)
    assert len(td_receipt(*args, 1000)) == 6
    result = copy.deepcopy(args[0])
    result['commands'][0]['returncode'] = 1
    with pytest.raises(ValueError):
        td_receipt(result, *args[1:], 1000)
    log = Path(args[0]['native_payloads'][0]['log_path'])
    log.write_text('{"status":"FAILED"}')
    with pytest.raises(ValueError):
        td_receipt(*args, 1000)


def test_predraw_alias_ownership_including_mirror_and_different_fen_clocks():
    board = chess.Board()
    rows = [dict(root_fen=board.fen(), source_family_id='a'),
            dict(root_fen=board.mirror().fen(), source_family_id='b'),
            dict(root_fen=board.fen().replace(' 0 1', ' 5 8'), source_family_id='c')]
    assert len({state_alias(r) for r in rows}) == 1
    groups, owners = own_aliases({'B01': [rows[1]], 'A02': [rows[0], rows[2]]})
    assert set(groups) == {'A02'} and len(groups['A02']) == 2
    assert set(owners.values()) == {'A02'}


def test_mc_receipt_is_distinct_never_reuses_td_status(tmp_path):
    result, reg, ref, c, steps = fixture_receipt(tmp_path)
    c['phase'] = 'own-closed-terminal-learning-v1'
    with pytest.raises(ValueError):
        td_receipt(result, reg, ref, c, steps, 1000)
    result['status'] = 'PASS-closed-terminal-fixed-phase-and-fresh-native-loads-not-strength'
    reg['schema'] = 'NNUE-own-closed-terminal-training-orchestration-v1'
    assert len(td_receipt(result, reg, ref, c, steps, 1000)) == 6


def test_mc_proof_contract_transition_not_td_proof_relabel():
    proof, fit = contract_fixture()
    proof['phase'] = fit['phase'] = 'own-closed-terminal-learning-v1'
    fit['closed_terminal_proof_result'] = fit.pop('tdlambda_proof_result')
    info = dict(proof_result=fit['closed_terminal_proof_result'])
    compare_contracts(proof, fit, info)
    fit['tdlambda_proof_result'] = fit.pop('closed_terminal_proof_result')
    with pytest.raises((ValueError, KeyError)):
        compare_contracts(proof, fit, info)


def test_profile_value_seal_interface_reexports():
    import admission

    assert callable(admission.child) and callable(admission.import_nnue)
    assert admission.SEARCH_SHA == (
        'de53c14728a67b7772f18b396ac8ef5c35a144d4e4e616fec40099cd461a6670')


def test_missing_mixed_value_is_preseal_error_without_model_read():
    from wiring import validate_wiring

    with pytest.raises(KeyError, match='original_mixed_value'):
        validate_wiring(dict(target_variant='tdlambda-0.5-v1',
                             match_seeds=[20262905, 20262906]))



def test_afterstate_receipt_requires_new_source_type(tmp_path):
    result, reg, ref, c, steps = fixture_receipt(tmp_path)
    c['phase'] = 'own-afterstate-search-q-learning-v1'
    with pytest.raises(ValueError):
        td_receipt(result, reg, ref, c, steps, 1000)
    result['status'] = 'PASS-afterstate-fixed-phase-and-fresh-native-loads-not-strength'
    reg['schema'] = 'NNUE-own-afterstate-training-orchestration-v1'
    assert len(td_receipt(result, reg, ref, c, steps, 1000)) == 6
    # Parent/source/data/counter gates remain identical in the new typed branch.
    result['weights_only_initializer']['sha256'] = 'another-parent'
    with pytest.raises(ValueError):
        td_receipt(result, reg, ref, c, steps, 1000)


def test_afterstate_proof_transition_has_its_own_key():
    proof, fit = contract_fixture()
    proof['phase'] = fit['phase'] = 'own-afterstate-search-q-learning-v1'
    fit['afterstate_proof_result'] = fit.pop('tdlambda_proof_result')
    info = dict(proof_result=fit['afterstate_proof_result'])
    compare_contracts(proof, fit, info)
    fit['closed_terminal_proof_result'] = fit.pop('afterstate_proof_result')
    with pytest.raises((ValueError, KeyError)):
        compare_contracts(proof, fit, info)


def test_afterstate_registry_matches_frozen_source_contract_literals():
    from admission import VARIANTS

    expected = {
        'phase': 'own-afterstate-search-q-learning-v1',
        'native': 'own-kingbucket-nnue16-afterstate-full-native-cpu-v1',
        'status': 'PASS-afterstate-fixed-phase-and-fresh-native-loads-not-strength',
        'registration': 'NNUE-own-afterstate-training-orchestration-v1',
        'proof_key': 'afterstate_proof_result',
        'modules': ('model', 'native', 'contract', 'convert', 'contract_builder'),
    }
    assert VARIANTS['afterstate-search-q-v1'] == expected


def test_ranking_receipt_phase_cannot_reuse_afterstate_q_proof(tmp_path):
    result, reg, ref, c, steps = fixture_receipt(tmp_path)
    c['phase'] = 'own-afterstate-action-ranking-v1'
    with pytest.raises(ValueError):
        td_receipt(result, reg, ref, c, steps, 1000)
    result['status'] = 'PASS-action-ranking-fixed-phase-and-fresh-native-loads-not-strength'
    reg['schema'] = 'NNUE-own-action-ranking-training-orchestration-v1'
    assert len(td_receipt(result, reg, ref, c, steps, 1000)) == 6


def test_ranking_named_proof_contract_is_not_afterstate_or_mc_alias():
    from admission import VARIANTS

    row = VARIANTS['afterstate-action-ranking-v1']
    assert row['proof_key'] == 'action_ranking_proof_result'
    assert row['modules'] == (
        'model', 'ranking', 'native', 'contract', 'convert', 'contract_builder')
    proof, fit = contract_fixture()
    proof['phase'] = fit['phase'] = row['phase']
    fit[row['proof_key']] = fit.pop('tdlambda_proof_result')
    info = dict(proof_result=fit[row['proof_key']])
    compare_contracts(proof, fit, info)
    fit['afterstate_proof_result'] = fit.pop(row['proof_key'])
    with pytest.raises((ValueError, KeyError)):
        compare_contracts(proof, fit, info)


def test_actual_ranking_child_schema_distinct_from_unchanged_teacher_parent():
    from admission import candidate_schema

    assert candidate_schema('afterstate-action-ranking-v1', 'learned') == (
        'own-kingbucket-nnue16-action-ranking-model-v1')
    assert candidate_schema('afterstate-action-ranking-v1', 'parent') == (
        'own-kingbucket-nnue16-model-v1')
    assert candidate_schema('afterstate-search-q-v1', 'learned') == (
        'own-kingbucket-nnue16-model-v1')
    with pytest.raises(ValueError):
        candidate_schema('unregistered-ranking-alias', 'learned')

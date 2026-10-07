"""Synthetic metadata tests only; no real model/search/optimizer operations."""
import copy
from types import SimpleNamespace

import pytest
from admission import ZERO, check_identity, receipt


def example(generation=0):
    ref = dict(path='parent.pt', sha256='parent')
    origin = dict(phase=ZERO, teacher_labels_used=False)
    search = dict(path='search.py', sha256='exact')
    parent = dict(seed=20262905, generation=generation, lineage_origin=origin,
                  search_helper=search, phase=ZERO if generation == 0 else 'own-learning',
                  updates=0 if generation == 0 else 64)
    c = dict(seed=20262905, execution_scope_schema='human-prior-own-execution-contract-v1',
             phase='own-learning', updates=64, generation=generation + 1,
             parent_generation=generation, lineage_origin=origin, search_helper=search,
             parent_candidate=ref, bootstrap_candidate_path='parent.pt',
             bootstrap_candidate_sha256='parent', teacher_labels_used_in_own_phase=False,
             root_source_type='procedural-uniform-legal-walk-v2')
    candidate = dict(path='child.pt', sha256='child')
    q = dict(target_variant='human-randomstarts-own-q-v2', original_search=search,
             models={'20262905': dict(parent=ref, learned=candidate)},
             children={'20262905': dict(candidate=candidate)})
    return q, c, parent, ref


def test_zero_and_previous_own64_are_separate_current_parent_types():
    for generation in (0, 1, 2):
        q, c, parent, ref = example(generation)
        check_identity(q, 20262905, c, parent, ref)
        parent['phase'], parent['updates'] = 'teacher-bootstrap', 256
        with pytest.raises(ValueError):
            check_identity(q, 20262905, c, parent, ref)


def test_generation_search_parent_seed_and_target_tampering():
    q, c, parent, ref = example()
    for key, value in [('generation', 2), ('seed', 20262906),
                       ('parent_candidate', dict(path='substitute', sha256='parent')),
                       ('search_helper', dict(path='other', sha256='exact')),
                       ('teacher_labels_used_in_own_phase', True)]:
        bad = copy.deepcopy(c)
        bad[key] = value
        with pytest.raises(ValueError):
            check_identity(q, 20262905, bad, parent, ref)


def test_teacher_parent_cannot_be_silently_replaced_with_zero_endpoint():
    q, c, parent, ref = example(1)
    q['models']['20262905']['parent'] = dict(path='zero.pt', sha256='zero')
    with pytest.raises(ValueError):
        check_identity(q, 20262905, c, parent, ref)


def receipt_fixture():
    contract = dict(seed=20262905, generation=1, dataset_sha256='data',
                    parent_candidate={'path': 'parent', 'sha256': 'p'},
                    original_first_epoch=10, original_deadline_epoch=610)
    cref = {'path': 'contract', 'sha256': 'c'}
    steps = [0, 8, 0, 4, 4, 8]
    rows = [{'step': x} for x in steps]
    commands = [dict(command=['optimizer'], returncode=0, finished=20) for _ in range(3)]
    commands += [dict(command=['python', 'train', '--audit-only', '--contract', 'contract',
                              '--dataset', 'data'], returncode=0, finished=20) for _ in rows]
    result = dict(status='PASS-own-NNUE-fixed-phase-and-fresh-native-loads-not-strength',
                  mode='proof', own_updates=8, generation=1, contract_sha256='c',
                  raw_zip_identity_claimed=False, teacher_labels_used=False,
                  weights_only_initializer=dict(contract['parent_candidate'],
                                                contract_sha256='parentcontract'), first=10,
                  finished=40, deadline=610, full_payload_bits_equal=True,
                  commands=commands, native_payloads=rows)
    reg = dict(schema='human-randomstarts-own-training-orchestration-v2', mode='proof',
               seed=20262905, first=10, deadline=610, contract=cref,
               dataset={'path': 'data', 'sha256': 'data'}, train={'path': 'train'},
               parent_candidate=dict(contract['parent_candidate'],
                                     contract_sha256='parentcontract'))
    return result, reg, cref, contract, steps


def test_actual_receipt_native_validator_is_mandatory_and_clock_inclusive():
    args = receipt_fixture()
    calls = []
    bridge = SimpleNamespace(native_rows=lambda r, steps: calls.append(steps))
    assert len(receipt(*args, 1000, bridge, 'parentcontract')) == 6
    assert calls == [[0, 8, 0, 4, 4, 8]]
    args[0]['finished'] = 611
    with pytest.raises(ValueError):
        receipt(*args, 1000, bridge, 'parentcontract')


def test_synthetic_or_different_actual_training_helper_cannot_supply_proof():
    args = receipt_fixture()
    bridge = SimpleNamespace(native_rows=lambda *_: None)
    args[0]['status'] = 'PASS-human-prior-synthetic-parent-bridge-not-collection'
    with pytest.raises(ValueError):
        receipt(*args, 1000, bridge, 'parentcontract')
    args = receipt_fixture()
    args[0]['commands'][-1]['command'][1] = 'substituted-train'
    with pytest.raises(ValueError):
        receipt(*args, 1000, bridge, 'parentcontract')


def test_actual_registry_full_parent_contract_binding_cannot_be_dropped_or_tampered():
    args = receipt_fixture()
    bridge = SimpleNamespace(native_rows=lambda *_: None)
    receipt(*args, 1000, bridge, 'parentcontract')
    args[0]['weights_only_initializer'].pop('contract_sha256')
    with pytest.raises(ValueError):
        receipt(*args, 1000, bridge, 'parentcontract')
    args = receipt_fixture()
    args[1]['parent_candidate']['contract_sha256'] = 'other-contract'
    with pytest.raises(ValueError):
        receipt(*args, 1000, bridge, 'parentcontract')


def test_teacher_selected_root_source_cannot_masquerade_as_procedural_bank():
    q, c, parent, ref = example()
    for field in [None, 'teacher-selected-ownq-train-roots-v2']:
        bad = copy.deepcopy(c)
        if field is None:
            bad.pop('root_source_type')
        else:
            bad['root_source_type'] = field
        with pytest.raises(ValueError):
            check_identity(q, 20262905, bad, parent, ref)


def test_procedural_source_inventory_and_collection_factory_are_actual(tmp_path):
    import hashlib
    import json

    from admission import procedural_sources

    def ref(path):
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    code = {}
    for name in ['root_bank', 'prepare_bank', 'metadata_factory']:
        path = tmp_path / (name + '.py')
        path.write_text('# Synthetic source inventory fixture only\n')
        code[name] = ref(path)
    inv = tmp_path / 'source-inventory.json'
    inv.write_text(json.dumps({'files': {n + '.py': {'sha256': r['sha256']}
                                      for n, r in code.items()}}))
    reg = tmp_path / 'collection.json'
    reg.write_text(json.dumps(dict(schema='human-randomstarts-own-collection-registration-v2',
        producer_source_sha256={'metadata_factory.py': code['metadata_factory']['sha256']})))
    bank = dict(path='synthetic-no-bank-generated', sha256='bank')
    q = dict(procedural_source_inventory=ref(inv),
             procedural_helpers={n:code[n] for n in ['prepare_bank', 'metadata_factory']},
             procedural_bank_receipts={'20262905':bank})
    info = dict(human_helpers={'root_bank':code['root_bank']})
    trace = dict(schema='human-randomstarts-own1024-data-provenance-v2',
        inputs=dict(schema='human-randomstarts-own1024-dataset-conversion-seal-v2',
                    procedural_bank_receipt=bank, registration=ref(reg)))
    assert procedural_sources(q, info, {'seed':20262905}, trace)['schema'].endswith('-v2')
    bad = copy.deepcopy(trace)
    bad['inputs']['procedural_bank_receipt'] = dict(path='old-teacher-selection',sha256='bank')
    with pytest.raises(ValueError):
        procedural_sources(q, info, {'seed':20262905}, bad)
    code_path = tmp_path / 'metadata_factory.py'
    code_path.write_text('# Changed after source freeze\n')
    with pytest.raises(ValueError):
        procedural_sources(q, info, {'seed':20262905}, trace)

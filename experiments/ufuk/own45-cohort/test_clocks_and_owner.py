"""Execute actual pure guards/builders; no jobs, inference or synthetic GPU claims."""
import ast
import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]

def production_path(filename):
    folder = 'ownsearch-method4' if filename.startswith('ownv1') else 'search-acting-method5'
    return ROOT / folder / filename
END = 1791170400


def function(filename, name, namespace=None):
    tree = ast.parse(production_path(filename).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = {'END': END, **(namespace or {})}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<actual-pure-guard>', 'exec'), scope)
    return scope[name]


@pytest.mark.parametrize('prefix', ['ownv1', 'own5'])
def test_actual_clock_guard_earliest_future_clock_and_global_reserve(prefix):
    guard = function(prefix + '_training_controller.py', 'validate_training_clock')
    reg = {'earliest_training_epoch': 100, 'whole_training_seconds_per_seed': 10800,
           'whole_audit_seconds_from_originalfirstclock': 12600,
           'posttraining_reserve_seconds': 10020}
    guard(reg, 101, 10901, 102)
    for first, deadline, now in [(99, 10899, 102), (103, 10903, 102),
                                 (101, 10902, 102), (101, 10901, 10901)]:
        with pytest.raises(AssertionError):
            guard(reg, first, deadline, now)
    first = END - 22620
    with pytest.raises(AssertionError):
        guard({**reg, 'earliest_training_epoch': first}, first, first + 10800, first)
    with pytest.raises(AssertionError):
        guard({**reg, 'earliest_training_epoch': True}, 101, 10901, 102)


def test_method5_reserve_explicitly_covers_method4_waiting_same_original_clock():
    guard = function('own5_training_controller.py', 'validate_training_clock')
    assert 6000 + 16620 == 12600 + 10020
    reg = {'earliest_training_epoch': 100, 'whole_training_seconds_per_seed': 3600,
           'whole_audit_seconds_from_originalfirstclock': 6000,
           'posttraining_reserve_seconds': 16620}
    guard(reg, 101, 3701, 102)
    first = END - 22620
    with pytest.raises(AssertionError):
        guard({**reg, 'earliest_training_epoch': first}, first, first + 3600, first)


def test_actual_method4_manifest_preserves_four_paths_source_E_and_clock(tmp_path):
    helpers = ('ownv1_audit_support.py', 'ownv1_full_audit.py', 'ownv1_fresh_cli_replay.py')
    for name in helpers:
        (tmp_path / name).write_bytes(name.encode())
    registration_file = tmp_path / 'registration.json'
    registration_file.write_bytes(b'synthetic fixture; not a protocol')

    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    seed = 20261425
    reg = {'status': 'frozen-before-formal-execution', 'qualification_ledger_slot': 4,
           'source_commit': 'a278bba67bce962cb9294f0d24040e02e9acf1f4',
           'fixed_epochs': 24, 'neural_audit_epochs': [1, 2, 6, 12, 18, 24],
           'whole_training_seconds_per_seed': 10800,
           'whole_audit_seconds_from_originalfirstclock': 12600,
           'configs': {str(seed): {'seed': seed}}}
    inputs = {name: {'path': '/' + name, 'sha256': 'a' * 64}
              for name in ('initial_weights', 'book', 'experiment_config')}
    inputs['protocol'] = {'path': str(registration_file), 'sha256': sha(registration_file)}
    build = function('ownv1_owner_config_factory.py', 'audit_manifest',
                     {'SEEDS': (20261425, 20261426), 'SOURCE': reg['source_commit'],
                      'sha': sha, 'Path': Path, 'AUDIT_HELPERS': helpers})
    result = build(reg, inputs, seed, 101,
                   {'registration': str(registration_file), 'repo': '/clean-a278',
                    'run': '/fixed-seed/run'}, tmp_path)
    assert result['inputs'] == inputs
    assert result['qualification_ledger_slot'] == 4 and result['fixed_epochs'] == 24
    assert result['original_training_started_epoch'] == 101
    assert result['original_training_deadline_epoch'] == 10901
    assert result['whole_audit_seconds'] == 12600
    assert set(result['helper_sha256']) == set(helpers)
    with pytest.raises(AssertionError):
        build({**reg, 'source_commit': '4515a7c0dda3b4f9615c2fc78a47c872ab14699d'},
              inputs, seed, 101, {'registration': str(registration_file)}, tmp_path)


@pytest.mark.parametrize('prefix', ['ownv1', 'own5'])
def test_actual_owner_common_clock_SHA_future_and_everyseed_match(prefix, tmp_path):
    import json
    from types import SimpleNamespace

    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    clock = tmp_path / 'clock.json'
    clock.write_text(json.dumps({'schema': 'own45-common-original-firstclock-v1',
        'slots': [4, 5], 'original_training_started_epoch': 101}))
    tree = ast.parse(production_path(prefix + '_owner_config_factory.py').read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    index = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'paths' for t in n.targets))
    code = compile(ast.Module(body=main.body[index + 1:index + 8], type_ignores=[]),
                   '<actual-owner-common-clock-guards>', 'exec')
    scope = {'Path': Path, 'sha': sha, 'json': json, 'time': SimpleNamespace(time=lambda: 102),
        'paths': {'common_original_firstclock_receipt': str(clock),
                  'common_original_firstclock_receipt_sha256': sha(clock)}}
    exec(code, scope)
    assert scope['common_first'] == 101
    loop = next(n for n in main.body if isinstance(n, ast.For)
                and isinstance(n.iter, ast.Name) and n.iter.id == 'SEEDS')
    equality = next(n for n in loop.body if isinstance(n, ast.Assert)
                    and isinstance(n.test, ast.Compare)
                    and isinstance(n.test.left, ast.Name) and n.test.left.id == 'first')
    check = compile(ast.Module(body=[equality], type_ignores=[]), '<actual-perseed-clock>', 'exec')
    exec(check, {**scope, 'first': 101})
    with pytest.raises(AssertionError):
        exec(check, {**scope, 'first': 102})
    clock.write_text(json.dumps({'schema': 'own45-common-original-firstclock-v1',
        'slots': [4, 5], 'original_training_started_epoch': 103}))
    with pytest.raises(AssertionError):
        exec(code, scope)
    scope['paths']['common_original_firstclock_receipt_sha256'] = sha(clock)
    with pytest.raises(AssertionError):
        exec(code, scope)

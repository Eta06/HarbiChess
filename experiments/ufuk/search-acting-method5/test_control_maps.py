"""No models: verify real gate function and actual call signatures/config identities."""
import ast
import copy
import inspect
from pathlib import Path

import own5_strength_config as config
import pytest

ROOT = Path(__file__).parent


def extract_gate():
    tree = ast.parse((ROOT / 'own5_all_gates.py').read_text())
    fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef)
              and x.name == 'require_strength_gates')
    namespace = {'SEEDS': config.SEEDS}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<actual-strength-gates>', 'exec'),
         namespace)
    return namespace['require_strength_gates']


def strength():
    return {'replicated_strength_pass': True, 'seed_results': [
        {'seed': seed, 'strength_pass': True, 'gates': {'all': True},
         'direct': {'mean': .61, 'ci_adjusted_98_75': [.51, .7]},
         'sf_paired_delta': {'mean': .11, 'ci_adjusted_98_75': [.01, .2]},
         'final_sf_mean': .25, 'caps': {'direct': .05, 'final_sf': .05, 'initial_sf': .05},
         'direct_adversarial_caps': {'ci_adjusted_98_75': [.51, .7]},
         'sf_delta_adversarial_caps': {'ci_adjusted_98_75': [.01, .2]}}
        for seed in config.SEEDS]}


def test_exact_original_strength_gates_bothseeds_and_strict_boundary():
    gate = extract_gate()
    gate(strength())
    for key, field, boundary in [('direct', 'mean', .6),
        ('direct', 'ci_adjusted_98_75', [.5, .7]),
        ('sf_paired_delta', 'mean', .1),
        ('sf_paired_delta', 'ci_adjusted_98_75', [0, .2])]:
        d = copy.deepcopy(strength())
        d['seed_results'][1][key][field] = boundary
        with pytest.raises(AssertionError):
            gate(d)
    d = strength()
    d['seed_results'][0]['caps']['direct'] = .05001
    with pytest.raises(AssertionError):
        gate(d)


def test_explicit4515_seeds_books_E_no_v1_native_alias():
    q = {'schema': 'ufuk-own5-strength-bindings-v1', 'status': 'frozen-before-formal-execution',
        'qualification_ledger_slot': 5, 'MAX_families': 8, 'seeds': list(config.SEEDS),
        'source_commit': config.SOURCE, 'fixed_epochs': 8,
        'books_sha256': {str(k): v for k, v in config.BOOKS.items()},
        'books_provenance_sha256': 'c' * 64, 'native_schema': 'torch-search-acting-native-cuda-v2',
        'games_per_arm': 96, 'arm_wall_seconds': 3600,
        'runtime_scope': 'same-A100-CPU-one-thread-Torch2.11-all-six-arms'}
    config.validate_config(q)
    for key, value in [('qualification_ledger_slot', 4),
        ('source_commit', 'a278bba67bce962cb9294f0d24040e02e9acf1f4'),
        ('native_schema', 'torch-ownsearch-native-cuda-v1'), ('fixed_epochs', 7)]:
        with pytest.raises(AssertionError):
            config.validate_config({**q, key: value})


def test_all_actual_post_launch_calls_bind_real_signature_no_extra_forbidden_tokens():
    tree = ast.parse((ROOT / 'own5_posttraining.py').read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'launch')
    names = [a.arg for a in fn.args.args]
    defaults = len(fn.args.defaults)
    signature = inspect.Signature([inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD,
        default=None if i >= len(names) - defaults else inspect.Parameter.empty)
        for i, name in enumerate(names)])
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == 'launch']
    assert len(calls) == 7
    for call in calls:
        signature.bind(*[None] * len(call.args), **{k.arg: None for k in call.keywords})


def test_strength_gate_AST_identical_to_frozen_method4():
    original = Path('/workspace/HarbiChess/experiments/ufuk/ownsearch-method4/ownv1_all_gates.py')
    trees = [ast.parse(p.read_text()) for p in (original, ROOT / 'own5_all_gates.py')]
    functions = [next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                      and n.name == 'require_strength_gates') for tree in trees]
    assert ast.dump(functions[0], include_attributes=False) == \
           ast.dump(functions[1], include_attributes=False)


def test_training_evidence_requires_actual_v2_E1_A100_retained_updates_not_infra_label():
    import own5_formal_config_factory as factory

    profile = {'source_commit': config.SOURCE,
        'schema': 'owned900-search-acting-v2-fullshape-development-v1',
        'status': 'pass-one-search-acting-v2-development-epoch-and-fullchronological-audit',
        'finished_epoch': 1, 'absolute_deadline_epoch': 2}
    qualification = {'source_commit': config.SOURCE,
        'schema': 'ufuk-search-acting-v2-E1-fullchronological-audit-result-v1',
        'status': 'pass-actualCUDA-search-acting-v2-E1-all-data-FIRST8-LAST8-'
                  'original128-masks-raw-packets-and-mutations',
        'actual_device_name': 'NVIDIA A100-SXM4-40GB', 'torch_version': '2.11.0+cu130',
        'finished_epoch': 1, 'absolute_deadline_epoch': 2,
        'new_selfplay_transitions_generated': 0, 'optimizer_updates_performed_by_qualification': 0,
        'targeted_actual_data_mutations_rejected': {'raw': True, 'terminal': True,
                                                  'search': True, 'frozen': True},
        'audit_report': {'epoch': 1, 'neural_witness_K': 8, 'raw_actor_replayed': 32768,
                         'optimizer_committed': 1, 'raw_actor_packet_roots_verified': 18}}
    factory.check_training_evidence(profile, qualification)
    for field, value in [('epoch', 32), ('neural_witness_K', 9), ('optimizer_committed', 0)]:
        bad = copy.deepcopy(qualification)
        bad['audit_report'][field] = value
        with pytest.raises(AssertionError):
            factory.check_training_evidence(profile, bad)
    with pytest.raises(AssertionError):
        factory.check_training_evidence(profile, {**qualification, 'actual_device_name': 'CPU'})

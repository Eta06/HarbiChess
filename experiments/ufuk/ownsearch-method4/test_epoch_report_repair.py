"""No models/inference: execute real report assignment after real wavefront iterator."""
import ast
import copy
from pathlib import Path

import ownv1_formal_config_factory as factory
import pytest


def test_real_audit_loop_preserves_native_epoch_while_visiting_33_wavefront_groups():
    tree = ast.parse(Path(__file__).with_name('ownv1_full_audit.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'audit_epoch')
    loop = next(n for n in fn.body if isinstance(n, ast.For) and isinstance(n.iter, ast.Call)
                and isinstance(n.iter.func, ast.Name) and n.iter.func.id == 'enumerate'
                and isinstance(n.iter.args[0], ast.Name) and n.iter.args[0].id == 'batches')
    report = next(n for n in fn.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'report' for t in n.targets))
    epoch_expr = next(k.value for k in report.value.keywords if k.arg == 'epoch')
    isolated = copy.deepcopy(loop)
    isolated.body = [ast.Pass()]
    module = ast.fix_missing_locations(ast.Module(body=[isolated,
         ast.Assign(targets=[ast.Name(id='reported_epoch', ctx=ast.Store())],
                    value=copy.deepcopy(epoch_expr))], type_ignores=[]))
    namespace = {'index': 1, 'batches': [[j] for j in range(33)]}
    exec(compile(module, '<actual-auditor-loop/report-epoch>', 'exec'), namespace)
    assert namespace['reported_epoch'] == 1
    assert namespace['wavefront_index'] == 32


def test_factory_accepts_valid_E1_and_rejects_preserved_epoch32_receipt():
    profile = {'source_commit': factory.SOURCE,
               'status': 'pass-one-development-epoch-and-readonly-audit',
               'finished_epoch': 1, 'absolute_deadline_epoch': 2}
    qualification = {'source_commit': factory.SOURCE,
        'status': 'pass-actualCUDA-E1-full-data-original-groups-raw-packets-and-targeted-mutations',
        'audit_report': {'epoch': 1, 'raw_actor_replayed': 32768,
                        'optimizer_committed': 12, 'raw_actor_packet_roots_verified': 18},
        'targeted_actual_data_mutations_rejected': {'packet': True},
        'optimizer_updates_performed_by_qualification': 0}
    factory.check_training_evidence(profile, qualification)
    qualification['audit_report']['epoch'] = 32
    with pytest.raises(AssertionError):
        factory.check_training_evidence(profile, qualification)

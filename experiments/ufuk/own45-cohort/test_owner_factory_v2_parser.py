"""Execute actual production parser AST only; no files, source changes, or jobs."""

import argparse
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
FACTORIES = [ROOT / 'ownsearch-method4/owner_config_factory_v2.py',
             ROOT / 'search-acting-method5/owner_config_factory_v2.py']
INPUTS = ('registration', 'three-input-manifest', 'qualification-config', 'paths-config')


def actual_parser(path):
    module = ast.parse(path.read_text(), filename=str(path))
    main = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    prefix = []
    for statement in main.body:
        if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call):
            call = statement.value.func
            if isinstance(call, ast.Attribute) and call.attr == 'parse_args':
                break
        prefix.append(statement)
    else:
        raise AssertionError('Actual production parse_args statement missing')
    namespace = {'argparse': argparse, 'Path': Path}
    executable = ast.fix_missing_locations(ast.Module(body=prefix, type_ignores=[]))
    exec(compile(executable, str(path), 'exec'), namespace)
    return namespace['p']


def legitimate_argv():
    args = []
    for name in INPUTS:
        args.extend(['--' + name, '/inputs/' + name + '.json', '--' + name + '-sha256',
                     'digest-argument-parser-only-no-files-loaded'])
    return [*args, '--helpers', '/helpers', '--output', '/absent-new-output']


@pytest.mark.parametrize('factory', FACTORIES)
def test_actual_parser_accepts_legitimate_owner_argv(factory):
    parser = actual_parser(factory)
    value = parser.parse_args(legitimate_argv())
    assert value.output == Path('/absent-new-output')
    assert value.helpers == Path('/helpers')
    assert not hasattr(value, 'output_sha256') and not hasattr(value, 'helpers_sha256')


@pytest.mark.parametrize('factory', FACTORIES)
@pytest.mark.parametrize('missing', INPUTS)
def test_each_actual_input_sha_remains_required(factory, missing):
    argv = legitimate_argv()
    position = argv.index('--' + missing + '-sha256')
    del argv[position:position + 2]
    with pytest.raises(SystemExit) as exc:
        actual_parser(factory).parse_args(argv)
    assert exc.value.code == 2


@pytest.mark.parametrize('factory', FACTORIES)
@pytest.mark.parametrize('forbidden', ['output-sha256', 'helpers-sha256'])
def test_directory_hash_flags_are_rejected(factory, forbidden):
    with pytest.raises(SystemExit) as exc:
        actual_parser(factory).parse_args([*legitimate_argv(), '--' + forbidden, 'unused'])
    assert exc.value.code == 2


def test_actual_launcher_supplies_all_required_owner_flags():
    path = ROOT / 'own45-cohort/launch_cohort.py'
    tree = ast.parse(path.read_text())
    commands = [n for n in ast.walk(tree) if isinstance(n, ast.List)
                and any(isinstance(x, ast.Constant) and x.value == '--paths-config'
                        for x in n.elts)]
    assert len(commands) == 1
    flags = {x.value for x in commands[0].elts if isinstance(x, ast.Constant)
             and isinstance(x.value, str) and x.value.startswith('--')}
    expected = {'--' + n for n in INPUTS} | {'--' + n + '-sha256' for n in INPUTS}
    assert flags == expected | {'--helpers', '--output'}

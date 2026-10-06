"""No real searches or fits; routing and independent charge-corruption checks."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import search
from audit_support import validate_extensions
from support import load

HERE = Path(__file__).parent
P = json.loads((HERE / 'protocol-DRAFT.json').read_bytes())


def test_zero_threshold_and_fixed_six_arms():
    d = P['selective_helpers']['model.py']
    model = load(d['path'], d['sha256'], 'test_selective_model')
    zero = json.loads((HERE / 'newzero.json').read_bytes())
    weights = model.load_model(zero)
    assert model.probability(weights, list(range(16))) == .5
    assert not model.probability(weights, list(range(16))) > zero['threshold']
    assert len(P['tasks']) == 6 and P['total_games'] == 192
    assert ['newzero', 'SF512'] in P['tasks'] and ['prior', 'SF512'] in P['tasks']
    assert ['learned', 'newzero'] in P['tasks']


def test_original_control_does_not_instantiate_selective(monkeypatch):
    class Original:
        def __init__(self, evaluator, **kwargs):
            self.evaluator, self.kwargs = evaluator, kwargs

    calls = []

    def fake_load(path, sha, name):
        calls.append(path)
        assert path == search.ORIGINAL_PATH
        return SimpleNamespace(BudgetSearch=Original)

    monkeypatch.setattr(search, 'load', fake_load)
    def evaluator(_):
        return 0.0
    engine = search.BudgetSearch(evaluator, nodes=512)
    assert type(engine) is Original and engine.evaluator is evaluator
    assert calls == [search.ORIGINAL_PATH]
    assert engine.controller_kind == 'original-prior-or-E8-search'


def packet():
    return dict(nodes=512, controller_kind='new-selective-Q-learned-or-zero',
                q_extension_receipts=[dict(nodes=64, status='completed', checked=True)])


def test_charged_receipts_for_both_selective_roles():
    for role in ['learned', 'newzero']:
        validate_extensions(packet(), role)


@pytest.mark.parametrize('mutation', ['cap', 'sum', 'status', 'boolean', 'controller'])
def test_extension_corruption_rejected(mutation):
    row = copy.deepcopy(packet())
    if mutation == 'cap':
        row['q_extension_receipts'][0]['nodes'] = 65
    elif mutation == 'sum':
        row['nodes'] = 63
    elif mutation == 'status':
        row['q_extension_receipts'][0]['status'] = 'madeup'
    elif mutation == 'boolean':
        row['q_extension_receipts'][0]['checked'] = 1
    else:
        row['controller_kind'] = 'original-prior-or-E8-search'
    with pytest.raises(ValueError):
        validate_extensions(row, 'learned')


def test_old_prior_cannot_borrow_selective_extensions():
    row = packet()
    row['controller_kind'] = 'original-prior-or-E8-search'
    with pytest.raises(ValueError):
        validate_extensions(row, 'prior')

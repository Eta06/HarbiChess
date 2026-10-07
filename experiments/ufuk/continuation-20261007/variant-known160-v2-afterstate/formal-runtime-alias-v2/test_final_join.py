"""Pure fail-closed join tests; no actual models, results, or games."""

import copy

import pytest
from bindings import SEEDS
from final_join import validate_bootstrap


def fixture():
    names = [
        "E8_direct_gt_060", "E8_direct_lower_gt_050", "SF_gain_E8_gt_010",
        "SF_gain_E8_lower_gt_0", "parent_direct_gt_060", "parent_direct_lower_gt_050",
        "SF_gain_parent_gt_0", "SF_gain_parent_lower_gt_0", "final_SF_ge_025", "caps_le_005",
    ]
    return {str(seed): dict(seed=seed, root_blocks=48, games=480, bootstrap_replicates=50000,
                           confidence_two_sided=.9984375, lower_tail=.00078125,
                           analyses=[dict(adverse_caps=adverse, intervals=[{}] * 4,
                                          gates=dict.fromkeys(names, True), passed=True)
                                     for adverse in [False, True]]) for seed in SEEDS}


def test_exact_all_seed_reports_required():
    assert validate_bootstrap(fixture())
    with pytest.raises(ValueError):
        validate_bootstrap({})
    x = fixture()
    del x[str(SEEDS[0])]
    with pytest.raises(ValueError):
        validate_bootstrap(x)


def test_adverse_gates_not_omitted_or_boolean_forged():
    x = fixture()
    x[str(SEEDS[0])]['analyses'][1]['gates']['caps_le_005'] = False
    x[str(SEEDS[0])]['analyses'][1]['passed'] = False
    assert not validate_bootstrap(x)
    for kind in ('drop', 'change', 'vacuous'):
        y = copy.deepcopy(fixture())
        row = y[str(SEEDS[0])]
        if kind == 'drop':
            del row['analyses'][1]['gates']['caps_le_005']
        elif kind == 'change':
            row['bootstrap_replicates'] = 49999
        else:
            row['analyses'] = []
        with pytest.raises(ValueError):
            validate_bootstrap(y)

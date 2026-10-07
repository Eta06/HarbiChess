import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "scope_control", Path(__file__).with_name("control.py")
)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def test_exact_new_scope():
    c.validate_strength_scope(copy.deepcopy(c.STRENGTH_SCOPE))
    assert c.STRENGTH_SCOPE["formal_games"] == 7 * 48 * 2 * 2
    assert c.STRENGTH_SCOPE["development_games"] == 7 * 8 * 2 * 2
    assert (
        c.STRENGTH_SCOPE["formal_lower_bounds"] * c.STRENGTH_SCOPE["formal_per_bound_alpha"]
        == 0.00625
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("required_current_parent", "diagnostic"),
        ("formal_games", 960),
        ("formal_lower_bounds", 8),
        ("formal_e_threshold", 1280),
        ("endpoint_generation", 2),
    ],
)
def test_scope_mutations(key, value):
    x = copy.deepcopy(c.STRENGTH_SCOPE)
    x[key] = value
    with pytest.raises(ValueError):
        c.validate_strength_scope(x)


def test_no_learning_code_diff():
    here = Path(__file__).parent
    old = here.parent / "tdleaf-generational-root-control-v1"
    for name in (
        "learn.py",
        "train_entry.py",
        "generation_entry.py",
        "collect_entry.py",
        "seal_plan.py",
    ):
        assert (here / name).read_bytes() == (old / name).read_bytes()

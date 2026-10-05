import sys
from pathlib import Path

import pytest
from train import assert_checkout_imports, load_common_trainer, sha

ROOT = Path("/workspace/HarbiChess")
COMMON = ROOT / "experiments/ufuk/cpu-fresh-learning-v1/train.py"
FEATURES = ROOT / "experiments/ufuk/cpu-fresh-learning-v1/features.py"
COMMON_SHA = "49445cbd5059770b3ccd433be9f21a4773786885a85d2d586005e10e2359a2c6"
FEATURE_SHA = "2fe5556e7564b2a91fc948fcd77a9214289933b7ae44a7a47bec5f00ea00c645"


def test_actual_main_common_helper_retains_exact_features_module():
    assert sha(COMMON) == COMMON_SHA
    assert sha(FEATURES) == FEATURE_SHA
    prior = sys.modules.get("features")
    common = load_common_trainer(COMMON, COMMON_SHA, FEATURES, FEATURE_SHA)
    assert Path(common.sha.__code__.co_filename).resolve() == FEATURES.resolve()
    assert sys.modules.get("features") is prior
    assert callable(common.prepare_fresh)


def test_harbi_imports_are_bound_to_clean_checkout_source():
    assert_checkout_imports(ROOT)


def test_actor_deadline_and_new_training_deadline_are_separately_bound():
    from train import validate_seed_clocks

    actor_deadline = 1791235580.0397933
    training_deadline = actor_deadline + 3600.0
    actor = {"original_deadline_epoch": actor_deadline}
    contract = {
        "actor_deadline_epoch": actor_deadline,
        "original_deadline_epoch": training_deadline,
    }
    validate_seed_clocks(actor, contract, training_deadline)
    with pytest.raises(ValueError, match="clocks differ"):
        validate_seed_clocks(actor, contract, actor_deadline)
    changed_actor = {"original_deadline_epoch": actor_deadline + 1.0}
    with pytest.raises(ValueError, match="clocks differ"):
        validate_seed_clocks(changed_actor, contract, training_deadline)

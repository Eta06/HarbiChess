import copy
import random

import pytest
from native import SCHEMA, decode, digest, encode, validate


def fixture():
    config = {"max_actions": 32768, "seed": 20262905, "deadline": "UNREGISTERED"}
    model = {"schema": "classical-own-linear-value-v1", "theta": [0.0] * 18}
    state = dict(
        schema=SCHEMA,
        config=config,
        config_sha256=digest(config),
        model=model,
        frozen_epoch_model_sha256=digest(model),
        rng=random.Random(3).getstate(),
        actions=0,
        games=[],
        active=None,
        epoch_index=0,
        journal_parent_sha256=None,
        training_native_sha256=None,
    )
    return config, state


def test_exact_roundtrip_bytes_and_rng_including_gauss_cache():
    config, state = fixture()
    r = random.Random(3)
    r.gauss(0.0, 1.0)
    state["rng"] = r.getstate()
    blob = encode(state, config)
    resumed = decode(blob, config)
    assert encode(resumed, config) == blob
    assert validate(resumed, config).gauss(0.0, 1.0) == r.gauss(0.0, 1.0)
    assert validate(resumed, config).getrandbits(64) == validate(state, config).getrandbits(64)


def test_wrong_oldschema_missing_rng_changed_model_config_rejected():
    c, s = fixture()
    for mutation in ("schema", "rng", "model", "config"):
        bad = copy.deepcopy(s)
        if mutation == "schema":
            bad["schema"] = "fresh-qsearch-selfplay-journal-v2"
        elif mutation == "rng":
            del bad["rng"]
        elif mutation == "model":
            bad["model"]["theta"][0] = 0.1
        else:
            bad["config"]["seed"] = 9
        with pytest.raises(ValueError):
            validate(bad, c)

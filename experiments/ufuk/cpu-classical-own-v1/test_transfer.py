"""Semantic tamper rejection, not a mirrored implementation test."""

import copy

import pytest
from transfer import validate_transfer


def fixture():
    proof = dict(
        seed=20262905,
        max_actions=9,
        epoch_id="actual-qual",
        original_first_epoch=100.0,
        original_deadline_epoch=700.0,
        roots=["initial"],
        excluded_training_position_keys=["protected"],
        model_sha256="m",
        helper_sha256="h",
        extra_provenance="unchanged",
    )
    production = copy.deepcopy(proof)
    production.update(
        max_actions=16384,
        epoch_id="actual-collection",
        original_first_epoch=800.0,
        original_deadline_epoch=8000.0,
    )
    transfer = {
        "proof_config_sha256": "sha",
        "allowed_difference_values": {
            k: {"proof": proof[k], "production": production[k]}
            for k in ("max_actions", "epoch_id", "original_first_epoch", "original_deadline_epoch")
        },
    }
    q = dict(config_sha256="sha", first=100.0, deadline=700.0, finished_epoch=600.0)
    return proof, production, transfer, q


def test_valid_only_four_literal_changes_same_seed():
    p, c, t, q = fixture()
    assert validate_transfer(p, c, t, q, "sha")


@pytest.mark.parametrize(
    "field",
    [
        "roots",
        "seed",
        "excluded_training_position_keys",
        "model_sha256",
        "helper_sha256",
        "extra_provenance",
    ],
)
def test_any_nonallowed_difference_fails_even_declared(field):
    p, c, t, q = fixture()
    c[field] = "tampered"
    t["allowed_difference_values"][field] = {"proof": p[field], "production": c[field]}
    with pytest.raises(ValueError, match="unauthorized"):
        validate_transfer(p, c, t, q, "sha")


def test_changed_sha_extra_field_and_unregisteredclock_rejected():
    p, c, t, q = fixture()
    with pytest.raises(ValueError, match="SHA"):
        validate_transfer(p, c, t, q, "other-sha")
    c["new_provenance"] = "different"
    with pytest.raises(ValueError, match="extra/missing"):
        validate_transfer(p, c, t, q, "sha")
    p, c, t, q = fixture()
    c["original_deadline_epoch"] += 1
    with pytest.raises(ValueError, match="literal"):
        validate_transfer(p, c, t, q, "sha")
    p, c, t, q = fixture()
    q["finished_epoch"] = 701.0
    with pytest.raises(ValueError, match="clock"):
        validate_transfer(p, c, t, q, "sha")

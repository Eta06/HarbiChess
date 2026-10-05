"""Corruption and immutable model transport controls; no SSH or inference."""

import base64
import copy
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).with_name("owner8_mlx_transport.py")
spec = importlib.util.spec_from_file_location("owner8_transport_tested", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def packet():
    bodies = {str(s): f"eligible-native-model-{s}".encode() for s in (20261825, 20261826)}
    return {
        "manifest": {
            "source_commit": mod.SOURCE,
            "qualification_ledger_slot": 8,
            "fixed_epochs": 8,
            "seeds": [
                {
                    "seed": s,
                    "candidate": (
                        f"/content/harbichess-runs/visited-loss-method8-seed-{s}/run/"
                        "checkpoints/epoch-00000008/model.safetensors"
                    ),
                    "candidate_sha256": mod.sha(bodies[str(s)]),
                }
                for s in (20261825, 20261826)
            ],
        },
        "models": {s: base64.b64encode(b).decode() for s, b in bodies.items()},
    }


def test_eligible_packet_and_exact_model_bytes():
    p = packet()
    m, rows = mod.validate_packet(p)
    assert m == p["manifest"]
    assert [body for _, body in rows] == [
        b"eligible-native-model-20261825",
        b"eligible-native-model-20261826",
    ]


@pytest.mark.parametrize(
    "change", ("source", "slot", "epoch", "seed", "path", "hash", "body", "extra")
)
def test_bad_source_or_model_binding_rejected(change):
    p = copy.deepcopy(packet())
    m = p["manifest"]
    if change == "source":
        m["source_commit"] = "0" * 40
    elif change == "slot":
        m["qualification_ledger_slot"] = 6
    elif change == "epoch":
        m["fixed_epochs"] = 7
    elif change == "seed":
        m["seeds"].reverse()
    elif change == "path":
        m["seeds"][0]["candidate"] = "/content/wrong-model.safetensors"
    elif change == "hash":
        m["seeds"][0]["candidate_sha256"] = "0" * 64
    elif change == "body":
        p["models"]["20261825"] = base64.b64encode(b"corrupt").decode()
    elif change == "extra":
        p["models"]["20261727"] = p["models"]["20261825"]
    with pytest.raises((AssertionError, ValueError, KeyError)):
        mod.validate_packet(p)


def test_existing_model_collision_is_preserved(tmp_path):
    p = tmp_path / "model.safetensors"
    mod.once(p, b"original")
    mod.once(p, b"original")
    with pytest.raises(AssertionError):
        mod.once(p, b"replacement")
    assert p.read_bytes() == b"original"


def test_model_symlink_rejected_without_modifying_target(tmp_path):
    target = tmp_path / "old-model"
    target.write_bytes(b"original")
    link = tmp_path / "model.safetensors"
    link.symlink_to(target)
    with pytest.raises(AssertionError):
        mod.once(link, b"replacement")
    assert target.read_bytes() == b"original"

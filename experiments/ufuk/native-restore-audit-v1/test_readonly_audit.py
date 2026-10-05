"""In-memory synthetic tiny state only; no full journals, training or disk payload."""

import copy
import hashlib
import random
import time

import numpy as np
import pytest
import strict_native_readonly_audit as audit
import torch


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen_policy = torch.nn.Parameter(torch.tensor([0.0, -0.0, 2.0]), requires_grad=False)
        self.head = torch.nn.Linear(2, 3)


@pytest.fixture
def tiny():
    with torch.random.fork_rng(devices=[]):
        model = Tiny()
    optimizer = torch.optim.AdamW(
        model.head.parameters(), lr=0.01, weight_decay=0.0001, foreach=False
    )
    contract = {
        "schema": "cpu-own-outcome-linear-training-native-v2",
        "source_commit": audit.SOURCE,
        "initial_e8_sha256": audit.E8,
        "device": "cpu",
        "torch_version": str(torch.__version__),
        "max_steps": 1,
        "original_deadline_epoch": time.time() - 100,
        "new_teacher_labels": False,
        "new_selfplay_moves_generated": 0,
    }
    manifest = {
        "schema": contract["schema"],
        "contract": contract,
        "accepted": 1,
        "artifacts": {"model.safetensors": "a" * 64, "training.pt": "b" * 64},
    }
    # Handcrafted Adam fixture: zero optimizer updates executed, synthetic committed counter1.
    for parameter in model.head.parameters():
        optimizer.state[parameter] = {
            "step": torch.tensor(1.0),
            "exp_avg": torch.zeros_like(parameter),
            "exp_avg_sq": torch.zeros_like(parameter),
        }
    state = copy.deepcopy(model.state_dict())
    portable = copy.deepcopy(state)
    train, val = ((0,),), ((1,),)
    prepared = (
        torch.zeros(2, 2),
        torch.tensor([0, 2]),
        train,
        val,
        [{"synthetic": True}],
        "d" * 64,
    )
    numpy = np.random.RandomState(5).get_state()
    saved = {
        "schema": manifest["schema"],
        "contract": copy.deepcopy(contract),
        "model": state,
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "accepted": 1,
        "attempted": 1,
        "trainable_names": ["head.weight", "head.bias"],
        "history": [{"step": 1, "loss": 1.0, "preclip_norm": 0.0}],
        "dataset_sha256": prepared[-1],
        "dataset_receipts": prepared[-2],
        "split_sha256": hashlib.sha256(
            audit.canonical({"train": train, "validation": val})
        ).hexdigest(),
        "python_rng": random.Random(1).getstate(),
        "sampler_rng": random.Random(2).getstate(),
        "torch_rng": torch.Generator().manual_seed(3).get_state(),
        "numpy_rng": {
            "kind": numpy[0],
            "keys": torch.tensor(numpy[1].astype(np.int64)),
            "position": numpy[2],
            "has_gauss": numpy[3],
            "cached_gauss": numpy[4],
        },
    }
    return manifest, contract, saved, model, portable, optimizer, prepared


def validate(fixture):
    return audit.validate_loaded_state(*fixture, guard=lambda: None)


def test_expired_training_contract_preserved_readonly_all_state_restorable_no_global_RNG_draws(
    tiny,
):
    original = copy.deepcopy(tiny[1])
    torch_rng, python_rng = torch.get_rng_state().clone(), random.getstate()
    result = validate(tiny)
    assert result["original_training_deadline_expired_at_audit"] is True
    assert result["original_training_contract"] == original
    assert tiny[1] == original
    assert result["optimizer_steps_or_generation_executed"] == 0
    assert result["online_actor_resume_claimed"] is False
    assert audit.tensor_bits_equal(torch_rng, torch.get_rng_state())
    assert python_rng == random.getstate()


@pytest.mark.parametrize(
    "kind",
    [
        "frozen-zero-sign",
        "portable",
        "Adam-step",
        "Adam-moment",
        "dataset",
        "split",
        "mask",
        "history",
        "numpy-keys",
    ],
)
def test_corruption_fails_closed_without_any_optimizer_step(tiny, kind):
    saved, portable = tiny[2], tiny[4]
    if kind == "frozen-zero-sign":
        saved["model"]["frozen_policy"][1] = 0.0
        portable["frozen_policy"][1] = 0.0
    elif kind == "portable":
        portable["head.weight"][0, 0] += 1
    elif kind == "Adam-step":
        saved["optimizer"]["state"][0]["step"] = torch.tensor(2.0)
    elif kind == "Adam-moment":
        saved["optimizer"]["state"][0]["exp_avg_sq"][0, 0] = -1
    elif kind == "dataset":
        saved["dataset_sha256"] = "0" * 64
    elif kind == "split":
        saved["split_sha256"] = "0" * 64
    elif kind == "mask":
        saved["trainable_names"].reverse()
    elif kind == "history":
        saved["history"][0]["step"] = 2
    else:
        saved["numpy_rng"]["keys"][0] = -1
    with pytest.raises(ValueError):
        validate(tiny)


def test_separate_expired_or_extended_audit_clock_fails_and_past_train_clock_is_not_reused():
    now = time.time()
    with pytest.raises(ValueError):
        audit.AuditClock(now - 1000, now - 100)
    with pytest.raises(ValueError):
        audit.AuditClock(now, now + 901)
    audit.AuditClock(now, now + 10)()


def test_original_contract_new_training_deadline_not_silently_substituted(tiny):
    changed = copy.deepcopy(tiny[1])
    changed["original_deadline_epoch"] = time.time() + 100
    with pytest.raises(ValueError, match="original-native-contract"):
        audit.validate_contract(tiny[0], changed)


def test_real_three_small_native_files_require_external_original_full_SHA_manifest():
    from pathlib import Path

    native = Path("/workspace/work/harbichess/cpu-outcome-value-v1-actual/fits/") / (
        "20262305-linear/checkpoints/step-00002048"
    )
    manifest = Path("/workspace/work/harbichess/native-delta-backup-proposal/") / (
        "20262305-linear/manifest.json"
    )
    sealed = audit.check_sealed_files(native, manifest, audit.sha(manifest))
    assert set(sealed["files"]) == set(audit.FILES)
    with pytest.raises(ValueError, match="externally-frozen"):
        audit.check_sealed_files(native, manifest, "0" * 64)
    other = Path("/workspace/work/harbichess/cpu-outcome-value-v1-actual/fits/") / (
        "20262306-linear/checkpoints/step-00002048"
    )
    with pytest.raises(ValueError, match="original-bytes"):
        audit.check_sealed_files(other, manifest, audit.sha(manifest))


def test_exact_original_feature_helper_replays_tiny_complete_and_UNKNOWN_games():
    import gzip
    import json
    import tempfile
    from pathlib import Path

    import chess

    helper_path = Path("/workspace/HarbiChess/experiments/ufuk/cpu-outcome-value-v1/features.py")
    native_path = Path("/workspace/work/harbichess/cpu-outcome-value-v1-actual/fits/") / (
        "20262305-linear/checkpoints/step-00002048/checkpoint.json"
    )
    original_manifest = json.loads(native_path.read_text())
    assert audit.sha(helper_path) == original_manifest["contract"]["feature_helper_sha256"]
    helper = audit.import_file(helper_path, "_exact_original_features_for_tiny_synthetic_test")
    sources = []
    for wants_validation in (False, True):
        sources.append(
            next(
                f"tiny-proof-source-{i}"
                for i in range(100)
                if (helper.partition(f"tiny-proof-source-{i}") == 0) == wants_validation
            )
        )
    actions = []
    for source, sequence in [
        (sources[0], ["f2f3", "e7e5", "g2g4", "d8h4"]),
        (sources[1], ["f2f3", "e7e5", "g2g4", "d8h4"]),
        ("synthetic-unknown", ["f2f3"]),
    ]:
        history = []
        for index, move in enumerate(sequence):
            before = {"root_fen": chess.STARTING_FEN, "moves": list(history)}
            history.append(move)
            known_final = len(sequence) == 4 and index == 3
            transition = {
                "source_id": source,
                "game_index": 0,
                "slot": 0,
                "pre": before,
                "post": {"root_fen": chess.STARTING_FEN, "moves": list(history)},
                "action": move,
                "rollout_cutoff": len(sequence) == 1,
                "terminal_result": "0-1" if known_final else None,
                "terminal_termination": "checkmate" if known_final else None,
            }
            actions.append({"transition": transition})
    counts = {
        "complete_games": 2,
        "normal_cap_games": 1,
        "epoch_truncated_games": 0,
        "excluded_actions": 1,
        "win_games": 0,
        "draw_games": 0,
        "loss_games": 2,
    }
    record = {
        "schema": "torch-fresh-fullgame-ppo-v1",
        "epoch": 1,
        "previous_sample_chain_sha256": "0" * 64,
        "target_counts": counts,
        "fresh_transitions": 9,
        "collection": {
            "schema": "full-history-frozen-policy-epoch-v1",
            "actions": actions,
            "truncations": [],
        },
    }
    record["sample_chain_sha256"] = hashlib.sha256(bytes(32) + audit.canonical(record)).hexdigest()
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
        journal = Path(directory) / "SYNTHETIC-tiny-journal.json.gz"
        journal.write_bytes(gzip.compress(audit.canonical(record)))
        x, y, train, validation, receipts, dataset = helper.prepare([journal], lambda: None)
    assert x.shape == (8, 20)
    assert y.tolist() == [2, 0, 2, 0, 2, 0, 2, 0]  # Mover POV flips after every ply.
    assert train == ((0, 1, 2, 3),) and validation == ((4, 5, 6, 7),)
    assert receipts[0]["excluded_actions"] == 1 and receipts[0]["known_rows"] == 8
    assert dataset == hashlib.sha256(x.numpy().tobytes() + y.numpy().tobytes()).hexdigest()

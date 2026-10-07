"""Synthetic source/wiring checks only; no real parent inference, collection or fit."""

import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import collector  # noqa: E402
import generation_plan as plan  # noqa: E402
import native  # noqa: E402
from parent_bridge import canonical, sha  # noqa: E402


def put(p, x):
    p.write_bytes(canonical(x))
    return dict(path=str(p), sha256=sha(p))


def fixture(tmp):
    banks = {}
    seeds = {}
    for s in plan.SEEDS:
        banks[str(s)] = {}
        seeds[str(s)] = {}
        for g in range(1, 4):
            bankseed = s + 10000 * g
            seeds[str(s)][str(g)] = bankseed
            banks[str(s)][str(g)] = put(
                tmp / f"{s}-{g}.json",
                dict(seed=bankseed, protected_aliases={"path": "fixture", "sha256": "x"}),
            )
    x = dict(
        schema=plan.SCHEMA,
        status="registered-before-generation1",
        recipe=plan.RECIPE,
        seeds=list(plan.SEEDS),
        first=1,
        operator_end_epoch=plan.END,
        banks=banks,
        bank_seeds=seeds,
        protected_aliases={"path": "fixture", "sha256": "x"},
    )
    return x


def test_fixed_plan_and_unauthorized_endpoint_rejected(tmp_path):
    x = fixture(tmp_path)
    r = put(tmp_path / "plan.json", x)
    assert plan.validate_plan(r, plan.SEEDS[0]) == x
    x["recipe"] = dict(x["recipe"], endpoint_generation=2)
    with pytest.raises(ValueError):
        plan.validate_plan(put(tmp_path / "bad.json", x), plan.SEEDS[0])


def test_same_seed_bank_and_generation_mutations_rejected(tmp_path):
    x = fixture(tmp_path)
    r = put(tmp_path / "plan.json", x)
    reg = dict(
        seed=plan.SEEDS[0],
        generation=1,
        generation_plan=r,
        procedural_bank_receipt=x["banks"][str(plan.SEEDS[0])]["1"],
        protected_aliases=x["protected_aliases"],
        operator_end_epoch=plan.END,
        original_first_epoch=2,
    )
    assert plan.require_generation(reg) == x
    for update in [
        {"generation": 4},
        {"seed": plan.SEEDS[1]},
        {"operator_end_epoch": plan.END + 1},
    ]:
        with pytest.raises(ValueError):
            plan.require_generation(dict(reg, **update))


def test_native_phase_is_new_and_generation_rng_fail_closed():
    assert collector.ROWS == 2048 and collector.ROOTS == 512
    assert native.SCHEMA == "procedural-generational-tdleaf-own-nnue16-native-cpu-v1"
    for generation, rng in [(1, 1), (4, 4000013)]:
        c = dict(
            phase=native.PHASE,
            updates=128,
            math=native.MATH,
            feature_schema=native.FEATURE_SCHEMA,
            seed=10,
            generation=generation,
            rng_seed=rng,
        )
        with pytest.raises(ValueError):
            native.Learner(c)


def test_cli_imports_fresh_interpreters():
    for name in [
        "metadata_factory.py",
        "admit_parent.py",
        "run_collection.py",
        "convert_cli.py",
        "contracts.py",
        "prepare_training.py",
        "prove.py",
        "train.py",
        "audit_collection_six.py",
    ]:
        p = subprocess.run(
            [sys.executable, str(HERE / name), "--help"],
            capture_output=True,
            text=True,
            timeout=20,
            env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        assert p.returncode == 0, (name, p.stderr)


def test_search_and_model_bytes_preserved():
    assert (
        sha(HERE / "search_original.py")
        == "de53c14728a67b7772f18b396ac8ef5c35a144d4e4e616fec40099cd461a6670"
    )
    assert (
        sha(HERE / "search_pv.py")
        == "a6fa35675fd4eb9c0c346a893bd44a6507b06bbff107f689a3f67a7dcc79015d"
    )
    assert (
        sha(HERE / "model.py") == "c04914bc5a560ad51de81c6d0aaeb3fae4d04b45c07cd8e39bbd31f655d36565"
    )


def test_weights_only_bridge_resets_adam_and_uses_distinct_generation_rng():
    import torch
    from model import NNUE16

    weights = NNUE16().state_dict()
    states = []
    for generation in (1, 2):
        contract = dict(
            phase=native.PHASE,
            updates=128,
            math=native.MATH,
            feature_schema=native.FEATURE_SCHEMA,
            seed=20262905,
            generation=generation,
            rng_seed=20262905 + 1000003 * generation,
            bootstrap_candidate_sha256="synthetic-fixture",
        )
        learner = native.Learner(contract, weights_initializer=weights)
        state = learner.native()
        assert learner.step == 0
        assert learner.optimizer.state_dict()["state"] == {}
        assert native.bits_equal(state["model"], weights)
        assert native.bits_equal(state["baseline"], weights)
        states.append(state)
    assert states[0]["sampler_rng"] != states[1]["sampler_rng"]
    assert not torch.equal(states[0]["torch_cpu_rng"], states[1]["torch_cpu_rng"])


def test_previous_generations_cannot_be_selected(tmp_path):
    for g in (1, 2):
        c = put(tmp_path / f"c{g}.json", dict(generation=g, updates=128, phase=native.PHASE))
        with pytest.raises(ValueError):
            plan.final_candidate({"contract": c})


def test_source_only_previous_alias_skip_keeps_complete_pool():
    import chess

    rows = []
    for i, move in enumerate(["e2e4", "d2d4", "c2c4"]):
        b = chess.Board()
        b.push_uci(move)
        rows.append(
            dict(
                root_id=f"fixture{i}",
                root_fen=chess.STARTING_FEN,
                prefix_uci=[move],
                role="TRAIN",
                final_alias=collector.alias(b),
            )
        )
    pool = dict(schema=collector.POOL, selection_status="pass", train_only=True, rows=rows)
    chosen, _ = collector.starts(pool, 20262905, set(), n=2, pool_size=3)
    excluded = chosen[0]["final_alias"]
    skipped = dict(pool, previous_start_aliases=[excluded])
    later, _ = collector.starts(skipped, 20262905, set(), n=2, pool_size=3)
    assert skipped["rows"] == rows
    assert excluded not in {r["final_alias"] for r in later}
    assert len(later) == 2
    with pytest.raises(ValueError):
        collector.starts(
            dict(pool, previous_start_aliases=sorted(r["final_alias"] for r in rows)),
            20262905,
            set(),
            n=2,
            pool_size=3,
        )

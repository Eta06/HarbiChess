"""Synthetic search fixture only, never real search qualification or strength."""

import json
from types import SimpleNamespace

import chess
from journal_v3 import Actor, digest, read, replay, save
from learner import Learner
from value import ClassicalValue


def config(root=None):
    return dict(
        nodes=512,
        qdepth=2,
        max_depth=8,
        exploration=0.05,
        total_ply_cap=400,
        max_actions=9,
        actors=1,
        original_deadline_epoch=1791270000.0,
        epoch_id="synthetic-test-NOT-qualified",
        seed=3,
        model_sha256="1" * 64,
        source_commit="2" * 40,
        search_helper_sha256="3" * 64,
        value_helper_sha256="4" * 64,
        producer_sha256="5" * 64,
        runtime_helper_sha256="6" * 64,
        prior_target="frozen-human-prior-scalar-mover-v1",
        evaluator_identity="classical-own-linear-value-v1",
        roots=[
            root
            or dict(source_id="test-mate", root_fen="7k/5Q2/6K1/8/8/8/8/8 w - - 0 1", prefix=[])
        ],
    )


class FixtureSearch:
    def search(self, b):
        legal = sorted(b.legal_moves, key=lambda m: m.uci())
        move = legal[0]
        for m in legal:
            b.push(m)
            mate = b.is_checkmate()
            b.pop()
            if mate:
                move = m
                break
        return SimpleNamespace(
            move=move, root_actions=len(legal), nodes=1, evaluations=1, completed_depth=1, value=0.5
        )


def test_whole9_pause4_resume9_exact_gzip_rng_and_terminal(tmp_path):
    c = config()
    prior = ClassicalValue().nonterminal
    whole = Actor(c, FixtureSearch, prior)
    whole.advance(9)
    save(tmp_path / "whole.gz", whole.state)
    pause = Actor(c, FixtureSearch, prior)
    pause.advance(4)
    save(tmp_path / "pause.gz", pause.state)
    resumed = Actor(c, FixtureSearch, prior, read(tmp_path / "pause.gz"))
    resumed.advance(9)
    save(tmp_path / "resumed.gz", resumed.state)
    assert (tmp_path / "whole.gz").read_bytes() == (tmp_path / "resumed.gz").read_bytes()
    packets = replay(resumed.state, c)
    assert packets and all(p[3] in ("1-0", "0-1", "1/2-1/2") for p in packets)


def test_total_ply_unknown_cap_no_outcome_rows_and_active_tail():
    c = config(
        dict(
            source_id="test-cap",
            root_fen=chess.STARTING_FEN.replace("w KQkq - 0 1", "b KQkq - 0 200"),
            prefix=[],
        )
    )
    c["max_actions"] = 1
    a = Actor(c, FixtureSearch, ClassicalValue().nonterminal)
    a.advance(1)
    assert a.state["games"][0]["result"] == "UNKNOWN" and replay(a.state, c) == []
    c = config(dict(source_id="test-active", root_fen=chess.STARTING_FEN, prefix=[]))
    c["max_actions"] = 1
    a = Actor(c, FixtureSearch, ClassicalValue().nonterminal)
    a.advance(1)
    assert a.state["active"] is not None and replay(a.state, c) == []


def test_synthetic_adam_whole8_pause4_resume8_fulljson_rng_exact_and_changed():
    groups = {"a": [((1.0,) * 18, 0.1, 1.0, 0.5)], "b": [((-0.5,) * 18, -0.2, -1.0, -0.7)]}
    contract = {"updates": 8, "schema": "synthetic-offline-not-data-qualified"}
    whole = Learner(3, contract)
    whole.advance(groups, 8)
    expected = json.loads(json.dumps(whole.native()))
    part = Learner(3, contract)
    part.advance(groups, 4)
    restored = Learner(3, contract, json.loads(json.dumps(part.native())))
    restored.advance(groups, 8)
    assert digest(restored.native()) == digest(expected)
    assert any(restored.theta) and restored.step == 8


def test_fresh_process_synthetic_whole_split_actor_and_adam(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    driver = Path(__file__).with_name("synthetic_fresh_resume.py")
    for mode, final in [("actor", 9), ("learner", 8)]:
        w = tmp_path / (mode + "-whole")
        p = tmp_path / (mode + "-pause")
        r = tmp_path / (mode + "-resume")
        base = [sys.executable, str(driver), "--mode", mode]
        for target, out, resume in [(final, w, None), (4, p, None), (final, r, p)]:
            cmd = [*base, "--stop", str(target), "--output", str(out)]
            if resume:
                cmd += ["--resume", str(resume)]
            subprocess.run(cmd, check=True, timeout=10)
        assert w.read_bytes() == r.read_bytes()


def test_full_prior_handoff_dedup_protected_and_unknown_filter(tmp_path):
    import pytest
    from learner import prepare

    c = config()
    a = Actor(c, FixtureSearch, ClassicalValue().nonterminal)
    a.advance(9)
    path = tmp_path / "data.gz"
    save(path, a.state)
    # Default production minimums may NEVER silently admit this tiny synthetic corpus.
    with pytest.raises(ValueError, match="admission failed"):
        prepare(path, c, [])
    protected = [" ".join(c["roots"][0]["root_fen"].split()[:4])]
    with pytest.raises(ValueError, match="admission failed"):
        prepare(path, c, protected, min_games=0, min_rows=0)
    # Real stored prior corruption must fail before sampling/training.
    state = json.loads(json.dumps(a.state))
    state["games"][0]["moves"][0]["human_prior_scalar"] += 0.01
    corrupted = tmp_path / "changed.gz"
    save(corrupted, state)
    with pytest.raises(ValueError, match="stored human prior"):
        prepare(corrupted, c, [], min_games=0, min_rows=0)

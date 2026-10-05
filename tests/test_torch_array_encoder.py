import os

import chess
import numpy as np
import pytest
import torch

from harbichess.chess.encoding import BoardEncoder
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove
from harbichess.training.torch_array_encoder import (
    TorchArrayBoardEncoder as ArrayBoardEncoder,
)

CASES = [
    (chess.STARTING_FEN, "e2e4 a7a6 e4e5 d7d5 e5d6"),
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1 e8c8"),
    ("7k/P7/8/8/8/8/8/7K w - - 0 1", "a7a8n"),
    ("7k/8/8/8/8/8/8/R6K w - - 99 1", "a1a2 h8g8"),
    (chess.STARTING_FEN, "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8"),
]


@pytest.mark.parametrize("fen,moves", CASES)
def test_all104_float32_storage_both_perspectives_history_and_rules(fen, moves):
    rules = PythonChessRules()
    old, new = BoardEncoder(rules), ArrayBoardEncoder(rules)
    state = rules.initial_state(fen)
    states = [state]
    for uci in moves.split():
        states.append(rules.apply(states[-1], ChessMove(uci)))
    for state in states:
        reference, candidate = old.encode(state), new.encode(state)
        expected = np.asarray(reference.values, dtype=np.float32)
        assert expected.tobytes() == candidate.values.tobytes()
        assert candidate.shape == reference.shape and candidate.schema_version == 1
        with pytest.raises(ValueError):
            candidate.values.setflags(write=True)
        with pytest.raises(ValueError):
            candidate.values[0] = 3.0
        unchanged = np.asarray(old.encode(state).values, dtype=np.float32)
        assert unchanged.tobytes() == expected.tobytes()


def test_real_e8_masked_cpu_outputs_storage_and_rng_unchanged():
    import copy
    from pathlib import Path

    import torch

    from harbichess.backends.torch_network import load_weights
    from harbichess.chess.actions import legal_action_indices
    from harbichess.training.torch_fullgame_ppo import make_torch_epoch_inference

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        weights = Path(
            os.environ.get(
                "HARBICHESS_TEST_E8",
                "/workspace/HarbiChess/artifacts/ufuk-a100-mirror-20261004/"
                "harbichess-inputs/initial-e8.safetensors",
            )
        )
        network = load_weights(weights)
        base = copy.deepcopy(network)
        before = {
            k: v.detach().reshape(-1).view(torch.uint8).clone()
            for k, v in network.state_dict().items()
        }
        rules = PythonChessRules()
        white = rules.initial_state()
        black = rules.apply(white, ChessMove("e2e4"))
        states = (white, black)
        legal = tuple(legal_action_indices(rules.inspect(s)) for s in states)
        rng_before = torch.get_rng_state().clone()
        old = make_torch_epoch_inference(network, base, BoardEncoder(rules), device="cpu")
        new = make_torch_epoch_inference(network, base, ArrayBoardEncoder(rules), device="cpu")
        assert old(states, legal) == new(states, legal)
        assert torch.equal(torch.get_rng_state(), rng_before)
        assert all(
            torch.equal(before[k], v.detach().reshape(-1).view(torch.uint8))
            for k, v in network.state_dict().items()
        )
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="Actual CUDA unavailable locally; root must run this branch on A100",
            ),
        ),
    ],
)
def test_private_array_learner_counterfactual_and_fresh_resume_all_payloads(tmp_path, device):
    import os
    import subprocess
    import sys
    from pathlib import Path

    from test_search_acting_stage import PROCESS, SOURCE, WEIGHTS, fixture

    config_path, _ = fixture(tmp_path)
    import json

    data = json.loads(config_path.read_text())
    data["device"] = device
    config_path.write_text(json.dumps(data) + "\n")
    import gzip
    import json

    import torch

    from harbichess.backends.torch_network import load_weights
    from harbichess.training.torch_search_acting_learner import TorchSearchActingLearner

    candidate = Path(__file__).resolve().parent.parent / "src"
    parent428 = Path(
        os.environ.get(
            "HARBICHESS_TEST_SOURCE428_SRC",
            "/workspace/work/harbichess/terminal-certificate-merged428-stage/baseline428/src",
        )
    )
    assert parent428.is_dir() and parent428.resolve() != candidate.resolve()
    config_path, inputs = fixture(tmp_path)
    raw = json.loads(config_path.read_text())
    raw["actors"]["max_additional_plies"] = 1
    raw["epoch_steps"] = 1
    raw["search"]["block_plies"] = 1
    raw["device"] = device
    config_path.write_text(json.dumps(raw) + "\n")
    book_path = tmp_path / "book.json"
    book = json.loads(book_path.read_text())
    for row in book["splits"]["train"]:
        row["opening"] = dict(root_fen=chess.STARTING_FEN, moves=[])
    book_path.write_text(json.dumps(book, sort_keys=True) + "\n")

    jobs = [
        (parent428, tmp_path / "parent428", "whole", "8" * 40),
        (candidate, tmp_path / "candidate", "whole", SOURCE),
        (candidate, tmp_path / "split", "pause", SOURCE),
        (candidate, tmp_path / "split", "resume", SOURCE),
        (candidate, tmp_path / "candidate", "strictload", SOURCE),
        (candidate, tmp_path / "split", "strictload", SOURCE),
    ]
    worker = PROCESS.replace(
        "root.mkdir(exist_ok=True)",
        "root.mkdir(exist_ok=True)\n"
        "expected='TorchArrayBoardEncoder'\n"
        "assert type(l.encoder).__name__==expected\n"
        "if l.epoch==0: l.checkpoint(root/'native0')",
    )
    for source_path, output, mode, source_commit in jobs:
        env = dict(
            os.environ,
            PYTHONPATH=str(source_path),
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
        )
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                worker,
                str(output),
                str(config_path),
                str(WEIGHTS),
                source_commit,
                mode,
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stderr

    def assert_tree_equal(left, right):
        if isinstance(left, torch.Tensor):
            assert isinstance(right, torch.Tensor) and torch.equal(left, right)
        elif isinstance(left, dict):
            assert isinstance(right, dict) and left.keys() == right.keys()
            for key in left:
                if key == "sample_chain_sha256":
                    assert len(left[key]) == len(right[key]) == 64
                    continue
                assert_tree_equal(left[key], right[key])
        elif isinstance(left, tuple | list):
            assert type(left) is type(right) and len(left) == len(right)
            for a, b in zip(left, right, strict=True):
                assert_tree_equal(a, b)
        else:
            assert left == right

    # Control and candidate both use source428's private array backend. Every
    # root is a fresh opening with no one-ply mate; all learned/optimizer/RNG
    # state must therefore match while journal format changes are explicit.
    for epoch in (0, 1, 2):
        parent = tmp_path / f"parent428/native{epoch}"
        candidate_native = tmp_path / f"candidate/native{epoch}"
        split = tmp_path / f"split/native{epoch}"
        for name in ("model.safetensors", "base.safetensors", "behavior.safetensors"):
            left = load_weights(parent / name).state_dict()
            right = load_weights(candidate_native / name).state_dict()
            assert all(
                torch.equal(left[k].view(torch.uint8), right[k].view(torch.uint8)) for k in left
            )
            assert (candidate_native / name).read_bytes() == (split / name).read_bytes()
        assert_tree_equal(
            torch.load(parent / "training.pt", map_location="cpu", weights_only=True),
            torch.load(candidate_native / "training.pt", map_location="cpu", weights_only=True),
        )
        assert_tree_equal(
            torch.load(candidate_native / "training.pt", map_location="cpu", weights_only=True),
            torch.load(split / "training.pt", map_location="cpu", weights_only=True),
        )
        parent_actor = json.loads((parent / "actor.json").read_text())
        candidate_actor = json.loads((candidate_native / "actor.json").read_text())
        parent_actor.pop("sample_chain_sha256", None)
        candidate_actor.pop("sample_chain_sha256", None)
        assert parent_actor == candidate_actor
        if epoch == 0:
            continue
        parent_record = json.loads(
            gzip.decompress((tmp_path / f"parent428/journal{epoch}.gz").read_bytes())
        )
        candidate_record = json.loads(
            gzip.decompress((tmp_path / f"candidate/journal{epoch}.gz").read_bytes())
        )
        split_record = json.loads(
            gzip.decompress((tmp_path / f"split/journal{epoch}.gz").read_bytes())
        )
        assert parent_record["own_search"]["schema"] == "pre-action-masked-search-behavior-v2"
        assert candidate_record["own_search"]["schema"] == "pre-action-masked-search-behavior-v4"
        assert parent_record["own_search"]["targets"] != candidate_record["own_search"]["targets"]
        for record in (candidate_record, split_record):
            assert all(root["certified_mates"] == [] for root in record["own_search"]["roots"])
            for root in record["own_search"]["roots"]:
                assert root["certified_losing_actions"] == []
                assert root["search_policy"] == root["raw_search_policy"]
                assert root["selected_action"] == root["raw_selected_action"]
                assert root["selected_action_reason"] == "raw-search-selection"
                assert root["loss_shield_status"] == "no-visited-loss-certificate"
                assert root["loss_shield_epsilon"] == 1e-12
        for record in (parent_record, candidate_record, split_record):
            record.pop("sample_chain_sha256", None)
            record.pop("previous_sample_chain_sha256", None)
            record.pop("schema", None)
            record["own_search"].pop("schema", None)
            record["own_search"].pop("targets", None)
            record["own_search"].pop("behavior", None)
            for root in record["own_search"]["roots"]:
                root.pop("certified_mates", None)
                for v4_field in (
                    "raw_search_policy",
                    "raw_selected_action",
                    "selected_action_reason",
                    "visited_loss_checked_actions",
                    "certified_losing_actions",
                    "loss_shield_status",
                    "loss_shield_epsilon",
                ):
                    root.pop(v4_field, None)
        assert parent_record == candidate_record == split_record

    # The candidate must refuse a source428 native rather than silently resume
    # old-format collection/search history as if it were a method7 checkpoint.
    parent_native = tmp_path / "parent428/native2"
    manifest_sha = (
        __import__("hashlib").sha256((parent_native / "checkpoint.json").read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="source/config/runtime mismatch"):
        TorchSearchActingLearner.resume(
            parent_native,
            config=__import__("test_search_acting_stage").config(config_path),
            input_paths=inputs,
            source_commit=SOURCE,
        )
    assert (
        __import__("hashlib").sha256((parent_native / "checkpoint.json").read_bytes()).hexdigest()
        == manifest_sha
    )

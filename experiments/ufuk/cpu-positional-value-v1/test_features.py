"""Exact existing-network feature parity; no training or strength games."""

import gzip
import hashlib
import json
import random

import chess
import numpy as np
import torch
from features import invariants, partition, prepare

from harbichess.training.torch_array_encoder import TorchArrayBoardEncoder


def test_invariants_exact_existing_encoder_reduction_all_halfmove_values():
    torch.set_num_threads(1)
    encoder = TorchArrayBoardEncoder()
    board = chess.Board()
    rng = random.Random(20262355)
    for i in range(101):
        if board.is_game_over() or i % 32 == 0:
            board = chess.Board()
        board.push(rng.choice(list(board.legal_moves)))
        board.halfmove_clock = i
        for position in (board, board.mirror()):
            encoded = encoder.encode_board(position)
            dense = torch.tensor(np.asarray(encoded.values).reshape(1, 8, 8, 104))
            expected = torch.cat(
                (
                    dense[:, :, :, :12].reshape(1, 768),
                    dense[:, :, :, 101].reshape(1, 64),
                    dense[:, :, :, 96:].mean((1, 2)),
                ),
                dim=1,
            )[0].numpy()
            assert invariants(position).tobytes() == expected.tobytes()


def test_repetition_and_en_passant_metadata_exact():
    encoder = TorchArrayBoardEncoder()
    for moves in (("e2e4",), ("g1f3", "g8f6", "f3g1", "f6g8") * 2):
        board = chess.Board()
        for move in moves:
            board.push_uci(move)
        encoded = encoder.encode_board(board)
        x = torch.tensor(np.asarray(encoded.values).reshape(1, 8, 8, 104))
        expected = torch.cat(
            (
                x[:, :, :, :12].reshape(1, 768),
                x[:, :, :, 101].reshape(1, 64),
                x[:, :, :, 96:].mean((1, 2)),
            ),
            dim=1,
        )[0].numpy()
        assert invariants(board).tobytes() == expected.tobytes()


def journal_fixture(path):
    train = next(f"source{i}" for i in range(100) if partition(f"source{i}") != 0)
    val = next(f"source{i}" for i in range(100) if partition(f"source{i}") == 0)
    actions = []
    for game, source, fen, ucis, cutoff in (
        (0, train, "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1", ["f7g7"], False),
        (1, val, "6k1/5Q2/6K1/8/8/8/8/8 b - - 0 1", ["g8h8", "f7g7"], False),
        (2, train, chess.STARTING_FEN, ["e2e4"], True),
    ):
        board = chess.Board(fen)
        history = []
        for uci in ucis:
            pre = {"root_fen": fen, "moves": list(history)}
            board.push_uci(uci)
            history.append(uci)
            outcome = board.outcome(claim_draw=True)
            actions.append(
                {
                    "transition": {
                        "source_id": source,
                        "game_index": game,
                        "slot": 0,
                        "pre": pre,
                        "post": {"root_fen": fen, "moves": list(history)},
                        "action": uci,
                        "rollout_cutoff": cutoff,
                        "terminal_result": outcome.result() if outcome else None,
                        "terminal_termination": outcome.termination.name.lower()
                        if outcome
                        else None,
                    }
                }
            )
    data = {
        "schema": "torch-fresh-fullgame-ppo-v1",
        "epoch": 1,
        "previous_sample_chain_sha256": hashlib.sha256(b"").hexdigest(),
        "fresh_transitions": 4,
        "collection": {
            "schema": "full-history-frozen-policy-epoch-v1",
            "actions": actions,
            "truncations": [],
        },
        "target_counts": {
            "complete_games": 2,
            "win_games": 1,
            "draw_games": 0,
            "loss_games": 1,
            "normal_cap_games": 1,
            "epoch_truncated_games": 0,
            "excluded_actions": 1,
        },
    }
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    data["sample_chain_sha256"] = hashlib.sha256(
        bytes.fromhex(data["previous_sample_chain_sha256"]) + canonical
    ).hexdigest()
    path.write_bytes(gzip.compress(json.dumps(data).encode()))


def test_known_both_mover_labels_and_unknown_exclusion(tmp_path):
    path = tmp_path / "journal.json.gz"
    journal_fixture(path)
    x, y, train, val, receipts, _ = prepare([path], lambda: None)
    assert x.shape == (3, 840) and y.tolist() == [0, 2, 0]
    assert train == ((0,),) and val == ((1, 2),)
    assert receipts[0]["legal_plies"] == 4 and receipts[0]["excluded_actions"] == 1


def test_old_journal_checksum_tamper_rejected(tmp_path):
    import pytest

    path = tmp_path / "journal.json.gz"
    journal_fixture(path)
    data = json.loads(gzip.decompress(path.read_bytes()))
    data["collection"]["actions"][0]["transition"]["action"] = "f7f8"
    path.write_bytes(gzip.compress(json.dumps(data).encode()))
    with pytest.raises(ValueError, match="chain checksum"):
        prepare([path], lambda: None)

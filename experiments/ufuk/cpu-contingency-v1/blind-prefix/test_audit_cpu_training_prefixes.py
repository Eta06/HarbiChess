from pathlib import Path

import chess
import pytest

import audit_cpu_training_prefixes as audit


def no_timeout():
    return None


def test_declares_both_full_eight_epoch_formal_runs():
    specs = audit.declared_registry(Path("/formal25"), Path("/formal26"))
    assert specs[-2]["expected_epochs"] == list(range(1, 9))
    assert specs[-1]["expected_epochs"] == list(range(1, 9))


def test_missing_formal_journals_never_pass_full_registry():
    missing = ["cpu-formal-seed20261925:epoch-8"]
    assert audit.full_coverage_status(missing, []) == "incomplete-formal-registry"


def test_missing_mandatory_tiny_or_development_prefix_never_passes():
    missing = ["cpu-tiny-whole-seed20261955:epoch-2"]
    assert audit.full_coverage_status(missing, []) == "incomplete-current-development-coverage"


def test_found_overlap_fails_even_with_missing_formal_journals():
    overlap = [{"run": "cpu-tiny-whole-seed20261955", "key": "x"}]
    assert audit.full_coverage_status(["cpu-formal-seed20261925:epoch-1"], overlap) == "fail-training-prefix-overlap"


def test_current_subset_stays_incomplete_until_dev_e1_is_present():
    missing = ["cpu-development-E1-seed20261925:epoch-1"]
    assert audit.current_subset_status(missing, []) == "incomplete-current-development-coverage"
    assert audit.current_subset_status([], []) == "pass-zero-overlap-current-tiny-and-E1"


def test_position_key_matches_four_fen_fields():
    board = chess.Board()
    assert audit.pos_key(board) == " ".join(board.fen().split()[:4])


def test_full_history_replay_rejects_illegal_move():
    with pytest.raises(ValueError, match="illegal full-history move"):
        audit.replay({"root_fen": chess.STARTING_FEN, "moves": ["e2e5"]}, no_timeout)


def test_full_history_replay_accepts_legal_line():
    board = audit.replay(
        {"root_fen": chess.STARTING_FEN, "moves": ["e2e4", "c7c5", "g1f3"]},
        no_timeout,
    )
    assert board.fen() == "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"

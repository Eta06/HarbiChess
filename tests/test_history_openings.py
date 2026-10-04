import hashlib
import io
import random

import chess
import chess.pgn
import pytest

from harbichess.training.history_openings import complete_records, freeze


def record(identifier, seed, **headers):
    game = chess.pgn.Game()
    game.headers.update(
        Site=f"https://lichess.org/{identifier}",
        Date="2026.09.01",
        WhiteElo="2200",
        BlackElo="2100",
        TimeControl="180+0",
        Result="1-0",
    )
    game.headers.update(headers)
    board, node, rng = chess.Board(), game, random.Random(seed)
    for _ in range(48):
        if board.outcome(claim_draw=True):
            break
        move = rng.choice(sorted(board.legal_moves, key=lambda m: m.uci()))
        board.push(move)
        node = node.add_variation(move)
    return (
        game.accept(chess.pgn.StringExporter(headers=True, variations=False, comments=False))
        + "\n\n"
    )


def test_true_history_roots_split_exclusions_rating_and_truncation(tmp_path):
    pgn = tmp_path / "prefix.pgn"
    text = "".join(record(f"game{i:04d}", i + 1) for i in range(8))
    text += record("lowElo00", 30, WhiteElo="1000")
    text += record("fastGame", 31, TimeControl="30+0")
    text += record("custom00", 32, SetUp="1", FEN=chess.STARTING_FEN)
    text += record("game0000", 1)
    text += '[Event "Rated Blitz game"]\n[Site "https://lichess.org/partial0"]\n1. e4 '
    pgn.write_text(text)
    assert len(list(complete_records(io.StringIO(text)))) == 12
    args = dict(
        pgn=pgn,
        excluded_keys=set(),
        source_receipts={"source": "fixture"},
        seed=42,
        train_families=3,
        validation_families=2,
    )
    book = freeze(output=tmp_path / "book.json", **args)
    roots = [row for rows in book["splits"].values() for row in rows]
    assert len({row["source_game"] for row in roots}) == 5
    assert book["selection_counts"]["rating_or_clock_filter"] == 2
    assert book["selection_counts"]["nonstandard_initial_state"] == 1
    assert book["selection_counts"]["duplicate_game"] == 1
    for row in roots:
        board = chess.Board()
        for move in row["opening"]["moves"]:
            board.push_uci(move)
        assert board.fen() == row["opening"]["fen"]
        assert len(board.move_stack) == row["root_ply"]
        assert board.outcome(claim_draw=True) is None
    forbidden = " ".join(roots[0]["opening"]["fen"].split()[:4])
    filtered = freeze(output=tmp_path / "filtered.json", **{**args, "excluded_keys": {forbidden}})
    assert all(
        " ".join(row["opening"]["fen"].split()[:4]) != forbidden
        for rows in filtered["splits"].values()
        for row in rows
    )
    again = freeze(output=tmp_path / "again.json", **args)
    assert book["splits"] == again["splits"]
    with pytest.raises(FileExistsError):
        freeze(output=tmp_path / "book.json", **args)
    with pytest.raises(ValueError, match="insufficient"):
        freeze(output=tmp_path / "insufficient.json", **{**args, "train_families": 20})
    assert not (tmp_path / "insufficient.json").exists()


def test_missing_result_cannot_publish_even_if_parser_accepts(tmp_path):
    pgn = tmp_path / "prefix.pgn"
    text = record("badGame0", 1).rstrip().removesuffix("1-0")
    pgn.write_text(text + '\n\n[Event "next incomplete"]\n')
    with pytest.raises(ValueError, match="insufficient"):
        freeze(
            pgn,
            tmp_path / "refused.json",
            excluded_keys=set(),
            source_receipts={},
            train_families=1,
            validation_families=1,
        )
    assert not (tmp_path / "refused.json").exists()


def test_arena_excludes_entire_source_games_even_with_another_root_seed(tmp_path):
    pgn = tmp_path / "prefix.pgn"
    pgn.write_text(
        "".join(record(f"game{i:04d}", i + 1) for i in range(8))
        + '[Event "partial"]\n1. e4 '
    )
    arguments = dict(pgn=pgn, excluded_keys=set(), source_receipts={},
                     train_families=2, validation_families=2, seed=42)
    original = freeze(output=tmp_path / "training-book.json", **arguments)
    excluded = {row["source_game"] for rows in original["splits"].values() for row in rows}
    assert "excluded_source_games" not in original and "requested_split_counts" not in original
    arena = freeze(output=tmp_path / "arena.json", **{**arguments, "seed": 43},
                   excluded_games=excluded, split_counts={"arena": 2})
    assert set(arena["splits"]) == {"arena"} and len(arena["splits"]["arena"]) == 2
    assert not {row["source_game"] for row in arena["splits"]["arena"]} & excluded
    assert arena["selection_counts"]["excluded_source_game"] == len(excluded)
    assert arena["excluded_source_games_sha256"] == hashlib.sha256(
        ("\n".join(sorted(excluded)) + "\n").encode()
    ).hexdigest()
    assert arena["requested_split_counts"] == {"arena": 2}
    for row in arena["splits"]["arena"]:
        board = chess.Board()
        for move in row["opening"]["moves"]:
            board.push_uci(move)
        assert board.fen() == row["opening"]["fen"]
    with pytest.raises(ValueError, match="insufficient"):
        freeze(output=tmp_path / "all-excluded.json", **arguments,
               excluded_games={f"https://lichess.org/game{i:04d}" for i in range(8)},
               split_counts={"arena": 2})
    assert not (tmp_path / "all-excluded.json").exists()


@pytest.mark.parametrize("counts", [{}, {"unknown": 2}, {"arena": 0}, {"arena": True}])
def test_invalid_explicit_root_splits_refused_before_reading(tmp_path, counts):
    with pytest.raises(ValueError, match="split counts"):
        freeze(tmp_path / "not-opened.pgn", tmp_path / "refused.json", excluded_keys=set(),
               source_receipts={}, split_counts=counts)
    assert not (tmp_path / "refused.json").exists()

"""Synthetic human-PGN wire and mocked randomness only; no real book selection."""

import io
import random

import chess
import chess.pgn
import pytest
from exposure import alias
from select_books import candidates, draw


def synthetic_game(seed, eco):
    board, game = chess.Board(), chess.pgn.Game()
    game.headers.update(Event="Synthetic fixture",
                        Site=f"https://lichess.org/fixture{seed}", ECO=eco)
    node, rng = game, random.Random(seed)
    for _ in range(20):
        if board.is_game_over():
            break
        move = rng.choice(list(board.legal_moves))
        node = node.add_variation(move)
        board.push(move)
    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
    return game.accept(exporter) + "\n\n"


def test_pool_last_record_excluded_fixed16_root_and_protected_alias():
    text = "".join(synthetic_game(seed, f"A{seed:02d}") for seed in range(3))
    groups, counts = candidates(text, set())
    assert counts["records"] == 2 and set(groups) == {"A00", "A01"}
    assert all(len(r["opening"]["moves"]) == 16 for rows in groups.values() for r in rows)
    blocked = alias(chess.Board(groups["A00"][0]["root_fen"]))
    filtered, counts = candidates(text, {blocked})
    assert "A00" not in filtered and counts["excluded"] == 1
    # Result headers do not enter root selection; no candidate score/ratings filters.
    changed = text.replace('[Result "*"]', '[Result "1-0"]')
    changed_groups = candidates(changed, set())[0]
    assert {k: [{a: b for a, b in r.items() if a != "human_source_record_sha256"}
                for r in rows] for k, rows in changed_groups.items()} == {
                    k: [{a: b for a, b in r.items() if a != "human_source_record_sha256"}
                        for r in rows] for k, rows in groups.items()}


def test_independent_stratum_uniform_draw_counts_probabilities_no_refill():
    groups = {}
    for i in range(96):
        game = chess.pgn.read_game(io.StringIO(synthetic_game(i + 100, f"B{i % 100:02d}")))
        b = chess.Board()
        moves = []
        for move in list(game.mainline_moves())[:16]:
            b.push(move)
            moves.append(move.uci())
        groups[f"E{i:02d}"] = [dict(root_id=f"r{i}", source_family_id=f"g{i}", eco=f"E{i:02d}",
                                       root_fen=b.fen(), opening=dict(moves=moves))]
    books, draws, strata = draw(groups, randbelow=lambda _n: 0)
    assert len(strata) == 96 and len(draws) == 96
    assert all(x["probability"] == 1 and x["population_size"] == 1 for x in draws)
    assert all(len(x["splits"]["arena"]) == 48 for x in books.values())
    del groups["E00"]
    with pytest.raises(ValueError, match="no refill"):
        draw(groups, randbelow=lambda _n: 0)

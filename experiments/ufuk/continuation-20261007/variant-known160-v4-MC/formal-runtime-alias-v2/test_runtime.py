"""Pure full-history/count/supplement bindings; no inference/search/model reads."""

import copy
import json
import random
from pathlib import Path

import chess
import pytest
from audit_formal import game_packet, validate_progress
from bindings import (
    SEEDS,
    sha,
    validate_book,
    validate_pair,
    validate_supplement,
    validate_value_wiring,
)


def book(offset):
    rows = []
    for i in range(48):
        rng = random.Random(offset + i)
        b = chess.Board()
        for _ in range(12):
            b.push(rng.choice(list(b.legal_moves)))
        rows.append(dict(root_id=f"r{offset+i}", source_family_id=f"s{offset+i}",
                         eco=f"A{offset+i:02d}", root_fen=b.fen(),
                         opening=dict(moves=[m.uci() for m in b.move_stack])))
    return {"splits": {"arena": rows}}


def test_exact48_pair_source_eco_counts_and_fullhistory_root():
    books = {str(SEEDS[0]): book(0), str(SEEDS[1]): book(48)}
    assert len(validate_pair(books)) == 96
    corrupted = copy.deepcopy(books)
    corrupted[str(SEEDS[1])]["splits"]["arena"][0]["source_family_id"] = "s0"
    with pytest.raises(ValueError):
        validate_pair(corrupted)
    corrupted = book(0)
    corrupted["splits"]["arena"][0]["root_fen"] = chess.STARTING_FEN
    with pytest.raises(ValueError):
        validate_book(corrupted)


def test_actual_registered_supplement_wire_and_markdown_pins(tmp_path):
    refs = []
    for name in ("draft.json", "approval.json", "prereg.md"):
        p = tmp_path / name
        p.write_text("{}" if name.endswith("json") else "# Frozen experiment\n")
        refs.append(dict(path=str(p), sha256=sha(p)))
    packet = dict(schema="ONE-confirmation-additional-fixed-n-betting-preregistration-v1",
                  status="REGISTERED-prospective-additional-requirement-before-any-new-strength-outcome",
                  original_confirmation_DRAFT=refs[0], user_v2_approval=refs[1],
                  current_experiment_preregistration=refs[2],
                  original_confirmation_registration_sha256=None, new_code_sha256="code",
                  fraction_grid=[f"{i}/10" for i in range(1, 10)] + ["1"],
                  exact_new_union_alpha="1/160", data_driven_bets_allowed=False,
                  root_blocks_fixed_n=48, new_required_bounds=8, exact_alpha_each="1/1280",
                  optional_stopping_allowed=False, original_bounds_unchanged_required=True)
    p = tmp_path / "supplement.json"
    p.write_text(json.dumps(packet))
    assert validate_supplement(dict(path=str(p), sha256=sha(p)), *refs, "code") == packet
    packet["original_confirmation_registration_sha256"] = "fake-prior-actual-registration"
    p.write_text(json.dumps(packet))
    with pytest.raises(ValueError):
        validate_supplement(dict(path=str(p), sha256=sha(p)), *refs, "code")


def test_empty_sf_nodes_only_when_no_sf_move_and_strict_played_depth():
    moves = ["f2f3", "e7e5", "g2g4", "d8h4"]
    game = dict(opening=moves[:3], moves=moves, candidate_color="black", plies=4,
                termination="checkmate", score=1., move_wall_seconds=[0.01],
                stockfish_nodes_by_move=[], search_by_move=[dict(ply=4, candidate=True,
                selected_move="d8h4", root_actions=30, legal_root_actions=30,
                nodes=31, evaluations=10, completed_depth=1, value=1.01, wall_seconds=0.01)])
    board = chess.Board()
    for u in moves[:3]:
        board.push_uci(u)
    game["search_by_move"][0].update(root_actions=board.legal_moves.count(),
                                     legal_root_actions=board.legal_moves.count(),
                                     nodes=board.legal_moves.count()+1)
    assert game_packet(game, moves[:3], "SF512")["SF_nodes"] == 0
    game["search_by_move"][0]["completed_depth"] = 0
    with pytest.raises(ValueError):
        game_packet(game, moves[:3], "SF512")


def test_search_identical_scope_metadata_only():
    here = Path(__file__).parent
    assert sha(here / "search.py") == (
        "de53c14728a67b7772f18b396ac8ef5c35a144d4e4e616fec40099cd461a6670")


def test_complete_nested_e8_mixed_adapter_metadata_required_without_model_reads():
    ref = dict(path="helper.py", sha256="a" * 64)
    q = {k: dict(ref)
         for k in ("original_mixed_value", "E8_value_helper", "prior_helper", "binary")}
    q.update(nnue_directory="same-directory",
             nnue_helpers={k: dict(ref) for k in ("model.py", "native.py", "evaluator.py")},
             models={str(s): {"learned": dict(path="own.pt"), "parent": dict(path="parent.pt"),
                              "e8": dict(path="e8.safetensors", sha256=
                              "e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03")}
                     for s in SEEDS})
    q["teachers"] = {s: dict(candidate=q["models"][s]["parent"]) for s in q["models"]}
    q["children"] = {s: dict(candidate=q["models"][s]["learned"]) for s in q["models"]}
    validate_value_wiring(q)
    del q["original_mixed_value"]
    with pytest.raises(ValueError, match="original_mixed_value"):
        validate_value_wiring(q)


def test_progress_reconciliation_all_moves_gameend_and_tail(tmp_path):
    game = dict(opening_pair=0, candidate_color="white", opening=["e2e4"],
                moves=["e2e4", "e7e5"])
    events = [dict(type="game_start", pair=0, color="white"),
              dict(type="move", ply=2, uci="e7e5"), dict(type="game_end", game=game)]
    path = tmp_path / "progress.jsonl"
    path.write_text("".join(json.dumps(x) + "\n" for x in events))
    validate_progress(path, [game])
    events[1]["uci"] = "e7e6"
    path.write_text("".join(json.dumps(x) + "\n" for x in events))
    with pytest.raises(ValueError):
        validate_progress(path, [game])

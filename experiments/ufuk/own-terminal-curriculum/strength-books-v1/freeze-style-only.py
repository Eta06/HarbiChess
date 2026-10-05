"""Freeze two virgin method6 strength-root books; never query models or engines."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import time
from pathlib import Path

import chess
import chess.pgn
from harbichess.training.history_openings import complete_records, file_hash, freeze

ROOT = Path("/workspace/HarbiChess")
CONSUMER = Path("/workspace/work/harbichess/search-acting-method5-books/consumer-635413")
WORK = Path("/workspace/work/harbichess/family6-strength-books")
METHOD4 = Path("/workspace/work/harbichess/ownsearch-method4-books")
PGN = ROOT / "artifacts/ufuk-broad-history-input-20261004/lichess-standard-2026-09-prefix-32MiB.pgn"
CAPACITY = ROOT / "artifacts/ufuk-broad-capacity-strength-roots-20261004"
FORMAL2_BOOK = ROOT / "artifacts/ufuk-a100-mirror-20261004/harbichess-inputs/book.json"
FIRST_METHOD6 = WORK / "20261705/opening-splits.json"


def add_book(path: Path, games: set[str], keys: set[str]) -> None:
    obj = json.loads(path.read_text())
    for rows in obj.get("splits", {}).values():
        for row in rows:
            if row.get("source_game"):
                games.add(row["source_game"])
            opening = row.get("opening", {})
            if opening.get("fen"):
                keys.add(" ".join(opening["fen"].split()[:4]))
            if opening.get("moves"):
                board = chess.Board(opening.get("root_fen", chess.STARTING_FEN))
                for uci in opening["moves"]:
                    board.push_uci(uci)
                    keys.add(" ".join(board.fen().split()[:4]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(20261705, 20261706), required=True)
    args = parser.parse_args()
    seed = args.seed
    train_seed = 20261625 if seed == 20261705 else 20261626
    out = WORK / str(seed)
    if seed == 20261706 and not FIRST_METHOD6.is_file():
        raise FileNotFoundError("first method6 book must be frozen before the second")
    started = time.monotonic()
    if out.exists():
        raise FileExistsError(out)
    _checkout = Path(
        subprocess.check_output(
            ["git", "-C", str(CONSUMER), "rev-parse", "--show-toplevel"], text=True
        ).strip()
    )
    source_commit = subprocess.check_output(
        ["git", "-C", str(CONSUMER), "rev-parse", "HEAD"], text=True
    ).strip()
    source_status = subprocess.check_output(
        ["git", "-C", str(CONSUMER), "status", "--porcelain"], text=True
    )
    prereg_path = WORK / "PREREGISTRATION.json"
    prereg = json.loads(prereg_path.read_text())
    all40_path = WORK / "main40-exclusions-v2/exclusions.json"
    all40_sha = file_hash(all40_path)
    expected = prereg["all_main40_train_exclusions"]
    if all40_sha != expected["sha256"]:
        raise ValueError("complete main40 exclusion SHA mismatch before selection")
    all40 = json.loads(all40_path.read_text())
    expected_epochs = {(seed, epoch) for seed in (20261205, 20261206) for epoch in range(1, 41)}
    rows = all40["journal_receipts"]
    if (
        all40["status"] != "pass-all80-exact-TRAIN-fullhistory-journals"
        or len(rows) != 80
        or {(row["seed"], row["epoch"]) for row in rows} != expected_epochs
        or any(
            row["status"] != "pass-exact-legal-fullhistory-TRAIN-prefixes"
            or row["fresh_transitions"] != 32768
            for row in rows
        )
        or all40["fresh_transitions_examined"] != 2621440
        or len(all40["all_observed_own_prefix_position_keys"]) != 2581543
        or len(set(all40["underlying_original_TRAIN_source_ids"])) != 4096
    ):
        raise ValueError("complete main40 exact TRAIN inventory mismatch")
    if source_commit != prereg["source_pin"]["consumer_repo_commit"] or source_status:
        raise ValueError("selection consumer source pin/cleanliness differs")
    if file_hash(PGN) != prereg["source_pin"]["pgn_sha256"]:
        raise ValueError("fixed PGN SHA mismatch")
    keys = set(json.loads((CAPACITY / "excluded-position-keys.json").read_text()))
    games = set(json.loads((CAPACITY / "excluded-source-games.json").read_text()))
    keys.update(all40["all_observed_own_prefix_position_keys"])
    games.update(all40["underlying_original_TRAIN_source_ids"])
    del all40
    receipts = {
        "scope": (
            "Roots only; no model/engine queries, games, optimizer, or outcome inspection. "
            "Preserves capacity/formal2/confirmation3 and all method4 source IDs "
            "and root/fullprefix exclusions."
        ),
        "selection_seed": seed,
        "training_seed": train_seed,
        "qualification_ledger_slot": 6,
        "consumer_repo_commit": source_commit,
        "consumer_repo_clean_at_freeze": not bool(source_status),
        "freeze_script_sha256": file_hash(Path(__file__)),
        "preregistration_sha256": file_hash(WORK / "PREREGISTRATION.json"),
        "history_openings_source_sha256": file_hash(
            CONSUMER / "src/harbichess/training/history_openings.py"
        ),
        "all_main40_exclusion_sha256": all40_sha,
        "all_main40_journals": 80,
        "all_main40_fresh_transitions": 2621440,
        "all_main40_unique_position_keys": 2581543,
        "all_main40_source_ids": 4096,
        "excluded_inputs": [],
        "exclusion_unions": [],
    }
    excluded_files = [
        CAPACITY / "excluded-position-keys.json",
        CAPACITY / "excluded-source-games.json",
        CAPACITY / "freeze-result.json",
        CAPACITY / "independent-root-audit.json",
        all40_path,
    ]
    books = list((ROOT / "docs/research").glob("*opening-splits*.json"))
    books += [Path("/workspace/work/harbichess/a100-strength-second-book/opening-splits.json")]
    books += list(
        Path("/workspace/work/harbichess/a100-mc-strength-books").glob("*/opening-splits.json")
    )
    books += list(
        Path("/workspace/work/harbichess/a100-mc-confirmation3-books").glob("*/opening-splits.json")
    )
    books += [FORMAL2_BOOK]
    books += list(
        Path("/workspace/work/harbichess/search-acting-method5-books").glob("*/opening-splits.json")
    )
    extra_path = Path(
        "/workspace/work/harbichess/own-terminal-curriculum-stage/training-exclusions.json"
    )
    extra = json.loads(extra_path.read_text())
    keys.update(extra["all_observed_own_prefix_position_keys"])
    games.update(extra["underlying_original_TRAIN_source_ids"])
    excluded_files.append(extra_path)
    for old_seed in (20261605, 20261606):
        base = Path("/workspace/work/harbichess/search-acting-method5-books") / str(old_seed)
        keys.update(json.loads((base / "excluded-position-keys.json").read_text()))
        games.update(json.loads((base / "excluded-source-games.json").read_text()))
        excluded_files += [
            base / "excluded-position-keys.json",
            base / "excluded-source-games.json",
        ]
    method4_exclusions = [
        METHOD4 / str(method4_seed) / name
        for method4_seed in (20261505, 20261506)
        for name in ("excluded-position-keys.json", "excluded-source-games.json")
    ]
    for path in method4_exclusions:
        if not path.is_file():
            raise FileNotFoundError(f"method4 exclusion receipt missing: {path.name}")
        if path.name == "excluded-position-keys.json":
            keys.update(json.loads(path.read_text()))
        else:
            games.update(json.loads(path.read_text()))
        excluded_files.append(path)
    books += [METHOD4 / "20261505/opening-splits.json", METHOD4 / "20261506/opening-splits.json"]
    if seed == 20261706:
        books += [FIRST_METHOD6]
    unique_books = sorted({p.resolve() for p in books if p.is_file()})
    for path in excluded_files:
        kind = (
            "inherited-capacity-exclusion"
            if path.name.startswith("excluded-")
            else "inherited-capacity-evidence"
        )
        receipts["excluded_inputs"].append(
            {"kind": kind, "path": str(path), "sha256": file_hash(path)}
        )
    for path in unique_books:
        receipts["excluded_inputs"].append(
            {"kind": "prior-or-earlier-opening-book", "path": str(path), "sha256": file_hash(path)}
        )
        add_book(path, games, keys)
    receipts["exclusion_unions"] = [
        {
            "kind": "source_ids",
            "count": len(games),
            "sha256": hashlib.sha256(("\n".join(sorted(games)) + "\n").encode()).hexdigest(),
        },
        {
            "kind": "fullprefix_position_keys",
            "count": len(keys),
            "sha256": hashlib.sha256(("\n".join(sorted(keys)) + "\n").encode()).hexdigest(),
        },
    ]
    out.mkdir(parents=True, exist_ok=False)
    (out / "excluded-position-keys.json").write_text(
        json.dumps(sorted(keys), separators=(",", ":")) + "\n"
    )
    (out / "excluded-source-games.json").write_text(
        json.dumps(sorted(games), separators=(",", ":")) + "\n"
    )
    remaining = 600.0 - (time.monotonic() - started)
    if remaining <= 0:
        raise TimeoutError("600-second selection budget elapsed before source scan")
    book_path = out / "opening-splits.json"
    book = freeze(
        PGN,
        book_path,
        excluded_keys=keys,
        excluded_games=games,
        source_receipts=receipts,
        seed=seed,
        split_counts={"arena": 48},
        wall_seconds=remaining,
    )
    if time.monotonic() - started > 600:
        raise TimeoutError("selection exceeded original600-second ceiling")
    selected = {row["source_record"]: row for row in book["splits"]["arena"]}
    seen_ids = set()
    seen_keys = set()
    audited = []
    with PGN.open(encoding="utf8") as stream:
        for ordinal, record in enumerate(complete_records(stream)):
            if time.monotonic() - started > 900:
                raise TimeoutError("selection plus replay exceeded original900-second ceiling")
            if ordinal not in selected:
                continue
            row = selected[ordinal]
            game = chess.pgn.read_game(io.StringIO(record))
            assert game is not None and not game.errors
            assert hashlib.sha256(record.encode()).hexdigest() == row["source_record_sha256"]
            assert game.headers.get("Site") == row["source_game"]
            assert row["source_game"] not in games | seen_ids
            assert (
                game.headers.get("Variant", "Standard") == "Standard" and "FEN" not in game.headers
            )
            assert min(int(game.headers["WhiteElo"]), int(game.headers["BlackElo"])) >= 2000
            assert int(game.headers["TimeControl"].split("+")[0]) >= 60
            moves = list(game.mainline_moves())
            assert row["root_ply"] in (16, 24, 32, 48, 64, 96) and row["root_ply"] <= len(moves) - 8
            assert [m.uci() for m in moves[: row["root_ply"]]] == row["opening"]["moves"]
            board = chess.Board()
            for move in moves[: row["root_ply"]]:
                assert move in board.legal_moves
                board.push(move)
            assert board.fen() == row["opening"]["fen"] and board.is_valid()
            assert board.outcome(claim_draw=True) is None
            key = " ".join(board.fen().split()[:4])
            assert key not in keys | seen_keys
            seen_ids.add(row["source_game"])
            seen_keys.add(key)
            audited.append(
                {
                    "source_record": row["source_record"],
                    "source_game": row["source_game"],
                    "record_sha256": row["source_record_sha256"],
                    "root_ply": row["root_ply"],
                    "position_key": key,
                    "full_fen": board.fen(),
                    "full_history_uci": row["opening"]["moves"],
                }
            )
    assert len(audited) == 48 and file_hash(PGN) == book["pgn_sha256"]
    for item in receipts["excluded_inputs"]:
        if file_hash(Path(item["path"])) != item["sha256"]:
            raise ValueError("immutable exclusion input changed during selection/replay")
    if file_hash(prereg_path) != receipts["preregistration_sha256"]:
        raise ValueError("prospective registration changed during selection/replay")
    report = {
        "status": "pass-roots-only-no-games",
        "selection_seed": seed,
        "training_seed": train_seed,
        "qualification_ledger_slot": 6,
        "source_commit_at_freeze": source_commit,
        "source_status_clean_at_freeze": not bool(source_status),
        "script_sha256": file_hash(Path(__file__)),
        "preregistration_sha256": file_hash(WORK / "PREREGISTRATION.json"),
        "history_openings_source_sha256": file_hash(
            CONSUMER / "src/harbichess/training/history_openings.py"
        ),
        "book_sha256": file_hash(book_path),
        "all_main40_exclusion_sha256": all40_sha,
        "all_main40_journals": 80,
        "all_main40_fresh_transitions": 2621440,
        "all_main40_unique_position_keys": 2581543,
        "all_main40_source_ids": 4096,
        "pgn_sha256": book["pgn_sha256"],
        "excluded_inputs": receipts["excluded_inputs"],
        "excluded_source_count": len(games),
        "excluded_source_list_sha256": receipts["exclusion_unions"][0]["sha256"],
        "excluded_position_count": len(keys),
        "excluded_position_list_sha256": receipts["exclusion_unions"][1]["sha256"],
        "excluded_source_json_sha256": file_hash(out / "excluded-source-games.json"),
        "excluded_position_json_sha256": file_hash(out / "excluded-position-keys.json"),
        "independently_replayed_roots": len(audited),
        "audited_roots": audited,
        "selection_seconds": book["wall_seconds"],
        "whole_seconds": time.monotonic() - started,
        "selection_and_replay_ceiling_seconds": 900,
        "model_or_engine_queries": 0,
        "games_played": 0,
        "updates": 0,
    }
    (out / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k != "audited_roots"}, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()

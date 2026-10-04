"""Freeze true-history PGN roots without using engine scores or model quality."""

from __future__ import annotations

import hashlib
import io
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import chess
import chess.pgn

ROOT_PLIES = (16, 24, 32, 48, 64, 96)
HISTORY_BOOK_SCHEMA = 1


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def complete_records(stream):
    """A following Event header proves the previous prefix record was not truncated.

    The final record is conservatively rejected even if apparently complete: the
    input contract is a partial archive, not an independently complete PGN file.
    """
    lines = []
    for line in stream:
        if line.startswith('[Event "') and lines:
            yield "".join(lines)
            lines = []
        if lines or line.startswith('[Event "'):
            lines.append(line)


def freeze(
    pgn: Path,
    output: Path,
    *,
    excluded_keys: set[str],
    source_receipts: dict,
    seed: int = 20261025,
    train_families: int = 4096,
    validation_families: int = 1024,
    wall_seconds: float = 600,
    excluded_games: set[str] | None = None,
    split_counts: dict[str, int] | None = None,
) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if min(train_families, validation_families, wall_seconds) <= 0:
        raise ValueError("positive distinct-game split and selection budgets required")
    counts_by_split = {"train": train_families, "validation": validation_families}
    if split_counts is not None:
        if (
            not split_counts
            or not set(split_counts) <= {"train", "validation", "arena"}
            or any(type(count) is not int or count <= 0 for count in split_counts.values())
        ):
            raise ValueError("named split counts must be positive integers")
        counts_by_split = dict(split_counts)
    forbidden_games = frozenset(excluded_games or ())
    started = time.perf_counter()
    before = file_hash(pgn)
    candidates, seen_games = [], set()
    counts = Counter()
    with pgn.open(encoding="utf8") as stream:
        for ordinal, record in enumerate(complete_records(stream)):
            if time.perf_counter() - started >= wall_seconds:
                raise TimeoutError("fixed PGN selection budget; no book published")
            counts["complete_prefix_records"] += 1
            headers = chess.pgn.read_headers(io.StringIO(record))
            if headers is None:
                counts["invalid_headers"] += 1
                continue
            try:
                strong = min(int(headers["WhiteElo"]), int(headers["BlackElo"])) >= 2000
                clock = int(headers["TimeControl"].split("+")[0]) >= 60
            except (KeyError, ValueError):
                counts["invalid_rating_or_clock"] += 1
                continue
            if not strong or not clock:
                counts["rating_or_clock_filter"] += 1
                continue
            site = headers.get("Site", "")
            if not re.fullmatch(r"https://lichess.org/[A-Za-z0-9]{8}", site):
                counts["invalid_game_identifier"] += 1
                continue
            if site in seen_games:
                counts["duplicate_game"] += 1
                continue
            seen_games.add(site)
            if site in forbidden_games:
                counts["excluded_source_game"] += 1
                continue
            result = headers.get("Result")
            if result not in ("1-0", "0-1", "1/2-1/2") or not record.rstrip().endswith(result):
                counts["incomplete_result"] += 1
                continue
            if (
                "FEN" in headers
                or headers.get("SetUp", "0") != "0"
                or headers.get("Variant", "Standard") != "Standard"
            ):
                counts["nonstandard_initial_state"] += 1
                continue
            game = chess.pgn.read_game(io.StringIO(record))
            if game is None or game.errors:
                counts["invalid_mainline"] += 1
                continue
            moves = list(game.mainline_moves())
            plies = [ply for ply in ROOT_PLIES if ply <= len(moves) - 8]
            if not plies:
                counts["short_game"] += 1
                continue
            selector = random.Random(f"{seed}:{site}")
            ply = selector.choice(plies)
            board = chess.Board()
            for move in moves[:ply]:
                if move not in board.legal_moves:
                    raise ValueError("parsed PGN supplied an illegal true-history move")
                board.push(move)
            if not board.is_valid() or board.outcome(claim_draw=True) is not None:
                counts["terminal_or_drawn_root"] += 1
                continue
            candidates.append(
                {
                    "source_game": site,
                    "source_record": ordinal,
                    "source_record_sha256": hashlib.sha256(record.encode()).hexdigest(),
                    "source_headers": {
                        k: headers[k]
                        for k in ("Date", "WhiteElo", "BlackElo", "TimeControl")
                        if k in headers
                    },
                    "root_ply": ply,
                    "opening": {
                        "moves": [move.uci() for move in moves[:ply]],
                        "fen": board.fen(),
                        "eco": headers.get("ECO", "?"),
                        "name": "True-history CC0 rated game prefix",
                    },
                }
            )
    counts["eligible_candidates_before_root_exclusions"] = len(candidates)
    random.Random(seed).shuffle(candidates)
    roots, chosen = set(excluded_keys), []
    required = sum(counts_by_split.values())
    for candidate in candidates:
        key = " ".join(candidate["opening"]["fen"].split()[:4])
        if key in roots:
            counts["excluded_or_duplicate_root"] += 1
            continue
        roots.add(key)
        chosen.append(candidate)
        if len(chosen) == required:
            break
    if len(chosen) != required:
        raise ValueError(f"insufficient true-history distinct roots: {len(chosen)}/{required}")
    if file_hash(pgn) != before:
        raise ValueError("PGN source changed during immutable root selection")
    splits = {}
    cursor = 0
    for split, count in counts_by_split.items():
        rows = chosen[cursor : cursor + count]
        splits[split] = [{"family": family, **row} for family, row in enumerate(rows)]
        cursor += count
    book = {
        "schema": HISTORY_BOOK_SCHEMA,
        "seed": seed,
        "source": (
            "CC0 standard Lichess true complete game histories; "
            "bounded chronological archive prefix"
        ),
        "pgn_sha256": before,
        "source_receipts": source_receipts,
        "root_plies": list(ROOT_PLIES),
        "minimum_both_ratings": 2000,
        "minimum_base_seconds": 60,
        "remaining_recorded_plies": 8,
        "excluded_position_keys": len(excluded_keys),
        "excluded_keys_sha256": hashlib.sha256(
            ("\n".join(sorted(excluded_keys)) + "\n").encode()
        ).hexdigest(),
        "selection_counts": dict(counts),
        "wall_seconds": time.perf_counter() - started,
        "splits": splits,
        "scope": (
            "Root diversity only; no model/engine/result quality filtering, no teacher labels, "
            "no population representativeness or complete inherited-pretraining holdout. "
            "Last partial-source game conservatively excluded."
        ),
    }
    if excluded_games is not None:
        book["excluded_source_games"] = len(forbidden_games)
        book["excluded_source_games_sha256"] = hashlib.sha256(
            ("\n".join(sorted(forbidden_games)) + "\n").encode()
        ).hexdigest()
    if split_counts is not None:
        book["requested_split_counts"] = counts_by_split
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf8") as stream:
        stream.write(json.dumps(book, indent=2) + "\n")
    return book

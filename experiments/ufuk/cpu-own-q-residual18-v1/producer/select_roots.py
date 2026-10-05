"""Outcome-blind deterministic root selector for a future own-search audit.

Input groups must already be the exact, admitted *training* trajectory split.
This helper deliberately has no journal, engine, or model dependencies.
"""

from __future__ import annotations

import hashlib


def _rank(seed: int, trajectory_id: str, row_id: str) -> bytes:
    payload = f"own-q-reanalysis-v1|{seed}|{trajectory_id}|{row_id}".encode()
    return hashlib.sha256(payload).digest()


def select_roots(groups: dict[str, list[dict]], seed: int, count: int = 1024):
    """Select count pre-action rows, balancing complete trajectories.

    Each row requires a unique stable ``row_id``. Selection uses only identity
    and row order; result, score, q, depth, and outcome fields are ignored.
    It is intentionally strict about insufficient coverage and duplicate IDs.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("count must be positive")
    if len(groups) < 16 or any(not rows for rows in groups.values()):
        raise ValueError("at least 16 nonempty training trajectories required")
    all_ids = [row.get("row_id") for rows in groups.values() for row in rows]
    if any(not isinstance(value, str) or not value for value in all_ids):
        raise ValueError("every pre-action row needs a stable row_id")
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("row_id must be globally unique")
    if sum(map(len, groups.values())) < count:
        raise ValueError("training rows cannot fill the requested sample")

    ordered_games = sorted(
        groups,
        key=lambda game: hashlib.sha256(f"{seed}|{game}".encode()).digest(),
    )
    chosen: list[dict] = []
    # Round-robin game allocation avoids long games dominating the sample.
    ranked = {
        game: sorted(
            groups[game], key=lambda row: _rank(seed, game, row["row_id"])
        )
        for game in ordered_games
    }
    cursors = {game: 0 for game in ordered_games}
    while len(chosen) < count:
        progressed = False
        for game in ordered_games:
            cursor = cursors[game]
            if cursor < len(ranked[game]):
                chosen.append(ranked[game][cursor])
                cursors[game] = cursor + 1
                progressed = True
                if len(chosen) == count:
                    break
        if not progressed:
            raise ValueError("selector exhausted rows unexpectedly")
    return chosen

"""Dormant lambda-return transformer for synthetic/future sealed own-Q rows.

No data loading, chess search, inference, or fitting occurs in this module.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

LAMBDA = 0.5
KNOWN = "completed-own-terminal"
UNKNOWN = {"unknown-ply-cap", "unknown-row-budget-prefix"}
EXCLUDED = "excluded-protected-trajectory"


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite number")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def transform_episode(rows: Iterable[Mapping]) -> dict:
    """Build mover-perspective lambda targets for one final episode packet.

    Known episodes use the actual own terminal outcome. UNKNOWN episodes use
    their last logged pre-action Q as the truncation bootstrap. Protected games
    are excluded wholesale. Input mappings are never mutated.
    """
    lam = LAMBDA
    items = [dict(row) for row in rows]
    if not items:
        return {"status": "empty", "rows": [], "lambda": lam}

    root_ids = {r.get("root_id") for r in items}
    statuses = {r.get("episode_status") for r in items}
    if len(root_ids) != 1 or not isinstance(next(iter(root_ids)), str):
        raise ValueError("episode must have exactly one stable root_id")
    if len(statuses) != 1 or not isinstance(next(iter(statuses)), str):
        raise ValueError("episode rows must share one final status")
    status = next(iter(statuses))

    items.sort(key=lambda r: r.get("local_ply", -1))
    for index, row in enumerate(items):
        if row.get("local_ply") != index:
            raise ValueError("episode local plies must be contiguous from zero")
        if row.get("mover") not in {"white", "black"}:
            raise ValueError("invalid mover perspective")
        if type(row.get("train_eligible")) is not bool:
            raise ValueError("final train_eligible flag is required")
        if type(row.get("selected_action_played")) is not bool:
            raise ValueError("final selected_action_played flag is required")
        if not isinstance(row.get("protected_search_aliases"), list):
            raise ValueError("protected search alias list is required")
        if row.get("selected_action_played") is not True:
            # A protected search row is archived but is not part of a played
            # trajectory; it cannot be used as a TD transition.
            status = EXCLUDED
        if row.get("protected_search_aliases"):
            status = EXCLUDED
        if row.get("train_eligible") is False:
            status = EXCLUDED

    if status == EXCLUDED:
        return {
            "root_id": next(iter(root_ids)),
            "status": "excluded-protected-trajectory",
            "lambda": lam,
            "rows": [],
            "source_rows": len(items),
        }
    if status not in UNKNOWN | {KNOWN}:
        raise ValueError("unsupported final episode status")

    q_white: list[float] = []
    source_ids: list[str] = []
    white_outcome: float | None = None
    for row in items:
        q = _finite_number(row.get("clipped_q_mover"), "clipped_q_mover")
        if not -1.0 <= q <= 1.0:
            raise ValueError("clipped Q must be in [-1, 1]")
        q_white.append(q if row["mover"] == "white" else -q)
        row_id = row.get("root_id") + ":" + str(row["local_ply"])
        source_ids.append(row_id)

        outcome = row.get("own_wdl_mover")
        if status == KNOWN:
            if type(outcome) is not int or outcome not in {-1, 0, 1}:
                raise ValueError("known terminal episode needs exact mover WDL on every row")
            z_white = float(outcome if row["mover"] == "white" else -outcome)
            if white_outcome is not None and z_white != white_outcome:
                raise ValueError("mover WDL labels disagree on the episode result")
            white_outcome = z_white
        elif outcome is not None:
            raise ValueError("UNKNOWN episode cannot carry a WDL label")

    if status == KNOWN:
        assert white_outcome is not None
        returns = [0.0] * len(items)
        returns[-1] = white_outcome
        for i in range(len(items) - 2, -1, -1):
            returns[i] = (1.0 - lam) * q_white[i + 1] + lam * returns[i + 1]
        endpoint = "exact-own-terminal-wdl"
    else:
        # Last available Q belongs to the last recorded PRE-ACTION board. The
        # unseen board after that action is never evaluated or invented here.
        returns = [0.0] * len(items)
        returns[-1] = q_white[-1]
        for i in range(len(items) - 2, -1, -1):
            returns[i] = (1.0 - lam) * q_white[i + 1] + lam * returns[i + 1]
        endpoint = "last-recorded-preaction-q-bootstrap"

    out_rows = []
    for i, (row, target_white) in enumerate(zip(items, returns, strict=True)):
        target_mover = target_white if row["mover"] == "white" else -target_white
        out_rows.append(
            {
                "source_row_id": source_ids[i],
                "local_ply": i,
                "mover": row["mover"],
                "target_white": target_white,
                "target_mover": target_mover,
                "source_q_mover": float(row["clipped_q_mover"]),
                "terminal_wdl_used": status == KNOWN,
                "endpoint": endpoint,
            }
        )
    return {
        "root_id": next(iter(root_ids)),
        "status": "targeted-known-terminal" if status == KNOWN else "targeted-unknown-bootstrap",
        "lambda": lam,
        "source_rows": len(items),
        "rows": out_rows,
    }


def transform_episodes(episodes: Iterable[Iterable[Mapping]]) -> list[dict]:
    """Pure map over final episode groups; preserves episode ordering."""
    return [transform_episode(episode) for episode in episodes]

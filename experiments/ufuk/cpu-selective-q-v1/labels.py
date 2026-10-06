"""FIRST256 frozen root IDs/FIRST4 chronological nominalq2 leaves; exact history."""

import hashlib
import json
import math

import chess
from model import features


def history(board):
    return dict(root_fen=board.root().fen(), prefix_uci=[m.uci() for m in board.move_stack])


def restore(h):
    b = chess.Board(h["root_fen"])
    if not b.is_valid():
        raise ValueError("valid history root")
    for u in h["prefix_uci"]:
        m = chess.Move.from_uci(u)
        if m not in b.legal_moves:
            raise ValueError("illegal history")
        b.push(m)
    return b


def digest(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def collect(rows, base, exhausted, evaluator, guard=lambda: None, root_count=256):
    if len(rows) != 1024 or len({r["row_id"] for r in rows}) != 1024:
        raise ValueError("exact immutable1024 selected TRAIN roots")
    selected = sorted(rows, key=lambda r: r["row_id"])[:root_count]
    output = []
    roots = []

    class Observer(base):
        def quiesce(self, b, a, z, remaining, ply):
            pending = remaining == 0 and len(self.leaves) < 4 and self.terminal(b, ply) is None
            h = history(b) if pending else None
            value = super().quiesce(b, a, z, remaining, ply)
            if pending:
                self.leaves.append(
                    dict(
                        history=h,
                        static=value,
                        features=features(b, value),
                        alpha=a if math.isfinite(a) else None,
                        beta=z if math.isfinite(z) else None,
                        ply=ply,
                    )
                )
            return value

    # Extra search is a full-window bounded reference. It never returns a
    # stand-pat for an unresolved checked horizon; censor instead of false0.
    from search import search_type

    Extra = search_type(base, exhausted)
    for root in selected:
        guard()
        b = restore(root.get("history", root))
        if b.is_game_over(claim_draw=True):
            raise ValueError("nonterminal TRAIN root")
        observer = Observer(evaluator, nodes=512, quiescence_plies=2, max_depth=8, guard=guard)
        observer.leaves = []
        packet = observer.search(b)
        roots.append(
            dict(
                row_id=root["row_id"],
                trajectory_id=root["trajectory_id"],
                history_sha256=digest(history(b)),
                nodes=packet.nodes,
                leaf_count=len(observer.leaves),
            )
        )
        for ordinal, leaf in enumerate(observer.leaves):
            guard()
            at = restore(leaf["history"])
            extra = Extra(evaluator, nodes=64, quiescence_plies=2, max_depth=8, guard=guard)
            try:
                v = extra.extra(at, -math.inf, math.inf, 2, leaf["ply"], 0)
                status = "completed"
                target = int(abs(v - leaf["static"]) > 0.10)
            except exhausted:
                v = None
                status = "censored"
                target = None
            output.append(
                dict(
                    schema="own-selective-q-counterfactual-leaf-v1",
                    root_row_id=root["row_id"],
                    trajectory_id=root["trajectory_id"],
                    leaf_ordinal=ordinal,
                    **leaf,
                    history_sha256=digest(leaf["history"]),
                    reference_mover_value=v,
                    target=target,
                    status=status,
                    reference_bound="exact-within-fixed-depth"
                    if v is not None
                    else "unknown-budget-or-checked-horizon",
                    additional_nodes=extra.nodes,
                )
            )
            guard()
    return dict(
        schema="own-selective-q-leaf-dataset-v1",
        roots=roots,
        leaves=output,
        new_actions=0,
        new_games=0,
    )


def admission(data, training_groups):
    grouped = {}
    for r in data["leaves"]:
        board = restore(r["history"])
        if board.is_game_over(claim_draw=True) or r["features"] != features(board, r["static"]):
            raise ValueError("fullhistory feature packet")
        if not 1 <= r["additional_nodes"] <= 64 or r["leaf_ordinal"] not in range(4):
            raise ValueError("counterfactual exact budget/ordinal")
        if digest(r["history"]) != r["history_sha256"]:
            raise ValueError("history SHA")
        if r["status"] == "censored":
            if r["target"] is not None or r["reference_mover_value"] is not None:
                raise ValueError("censor is not negative")
            continue
        if r["status"] != "completed" or r["target"] != int(
            abs(r["reference_mover_value"] - r["static"]) > 0.10
        ):
            raise ValueError("literal label boundary")
        if r["trajectory_id"] in training_groups:
            grouped.setdefault(r["trajectory_id"], []).append(r)
    positives = sum(r["target"] for rows in grouped.values() for r in rows)
    if len(grouped) < 16 or positives < 64:
        raise ValueError("INCOMPLETE admission16trajectories/64positives")
    return grouped

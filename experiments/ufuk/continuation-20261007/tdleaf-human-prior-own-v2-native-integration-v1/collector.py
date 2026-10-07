"""Versioned current-parent self-play collector that records true search PV leaves."""

from __future__ import annotations

import hashlib
import math
import struct
from pathlib import Path

import chess

POOL = "human-prior-own-train-roots-v2"
PROTECTION_SCOPE = "human-prior-tdleaf-current-root-query-path-v2"
ROWS, ROOTS, PLIES, NODES, QDEPTH, DEPTH = 1024, 128, 16, 8192, 2, 8
CHUNK = 8 * 2**20


def alias(board):
    b = min(board.board_fen(), board.mirror().board_fen()).encode("ascii")
    return int.from_bytes(hashlib.sha256(b).digest()[:8], "little", signed=True)


def replay(root):
    if root.get("role") != "TRAIN" or root.get("root_fen") != chess.STARTING_FEN:
        raise ValueError("full standard-start TRAIN history required")
    if not isinstance(root.get("root_id"), str) or not isinstance(root.get("prefix_uci"), list):
        raise ValueError("root ID/full UCI prefix required")
    board = chess.Board()
    for uci in root["prefix_uci"]:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError("illegal root history")
        board.push(move)
    if not board.is_valid() or board.outcome(claim_draw=True):
        raise ValueError("invalid/terminal root")
    return board


def history_aliases(root):
    """Return ancestry context aliases; these are diagnostics, not exposure claims."""
    board = chess.Board(root["root_fen"])
    values = [alias(board)]
    for token in root["prefix_uci"]:
        board.push_uci(token)
        values.append(alias(board))
    return values


def current_path_aliases(board, moves):
    """Aliases from the current neural root through a legal PV/played path."""
    path = board.copy(stack=True)
    values = [alias(path)]
    for move in moves:
        if move not in path.legal_moves:
            raise ValueError("current-root path must be legal")
        path.push(move)
        values.append(alias(path))
    return values


def starts(pool, seed, protected, n=ROOTS, pool_size=4096):
    if (
        pool.get("schema") != POOL
        or pool.get("selection_status") != "pass"
        or pool.get("train_only") is not True
    ):
        raise ValueError("sealed TRAIN-only pool required")
    rs = pool.get("rows")
    if not isinstance(rs, list) or len(rs) != pool_size:
        raise ValueError("fixed root pool count")
    ids, histories, usable = set(), set(), []
    for row in rs:
        replay(row)
        history_sha = hashlib.sha256(
            (row["root_fen"] + "\n" + " ".join(row["prefix_uci"])).encode()
        ).hexdigest()
        if row["root_id"] in ids or history_sha in histories:
            raise ValueError("duplicate root/history")
        ids.add(row["root_id"])
        histories.add(history_sha)
        # Protection begins at the actual selected root. The unqueried ancestral
        # prefix remains recorded context and is not treated as a model input.
        if alias(replay(row)) not in protected:
            usable.append(row)
    if len(usable) < n:
        raise ValueError("fixed pool exhausted; no validation/autofill")
    chosen = sorted(
        usable,
        key=lambda row: hashlib.sha256(
            f"own-tdleaf-start-v2|{seed}|{row['root_id']}".encode()
        ).digest(),
    )[:n]
    return chosen, hashlib.sha256("\n".join(row["root_id"] for row in chosen).encode()).hexdigest()


class Traced:
    def __init__(self, fn):
        self.fn, self.seen, self.calls = fn, set(), 0

    def reset(self):
        self.seen.clear()
        self.calls = 0

    def __call__(self, board):
        self.seen.add(alias(board))
        self.calls += 1
        value = float(self.fn(board))
        if not math.isfinite(value) or abs(value) > 1:
            raise ValueError("static mover W-L range")
        return value


def _label(rows, status, white_result=None):
    for row in rows:
        row["episode_status"] = status
        row["own_wdl_mover"] = (
            None
            if white_result is None
            else (white_result if row["mover"] == "white" else -white_result)
        )
        row["outcome_status"] = "UNKNOWN" if white_result is None else "known-own-terminal"
        row["train_eligible"] = (
            white_result is not None or status != "excluded-protected-trajectory"
        )


def collect(
    pool,
    *,
    seed,
    protected,
    factory,
    row_limit=ROWS,
    start_limit=ROOTS,
    plies=PLIES,
    pool_size=4096,
    guard=lambda: None,
    on_event=lambda _event: None,
):
    roots, order_sha = starts(pool, seed, protected, start_limit, pool_size)
    train, allrows, games, exposure, ancestry_context = [], [], [], set(), set()
    for ordinal, root in enumerate(roots):
        guard()
        board = replay(root)
        prefix = list(root["prefix_uci"])
        source_prefix_aliases = history_aliases(root)
        ancestry_context.update(source_prefix_aliases)
        ancestry_protected = sorted(set(source_prefix_aliases) & protected)
        rows, was_protected = [], False
        on_event(
            dict(
                type="game_start",
                root_id=root["root_id"],
                root_ordinal=ordinal,
                root_fen=root["root_fen"],
                root_prefix_uci=list(prefix),
                root_alias=alias(board),
                root_prefix_aliases=source_prefix_aliases,
                ancestry_protected_aliases=ancestry_protected,
            )
        )
        traced = Traced(factory.evaluator)
        search = factory.make(traced)
        status, white_result = "unknown-ply-cap", None
        exposure.add(alias(board))
        for ply in range(plies):
            guard()
            outcome = board.outcome(claim_draw=True)
            if outcome:
                white_result = 0 if outcome.winner is None else (1 if outcome.winner else -1)
                status = "completed-own-terminal"
                break
            root_alias = alias(board)
            exposure.add(root_alias)
            if root_alias in protected:
                was_protected, status = True, "excluded-protected-trajectory"
                break
            traced.reset()
            result = search.search(board)
            legal = board.legal_moves.count()
            if result.move is None or result.move not in board.legal_moves:
                raise ValueError("selected best move illegal")
            if (
                result.nodes < result.root_actions + 1
                or result.nodes > NODES
                or result.root_actions != legal
                or result.evaluations != traced.calls
            ):
                raise ValueError("search/root/evaluation accounting")
            q = float(result.value)
            if not math.isfinite(q) or abs(q) > 2:
                raise ValueError("mover Q range")
            protected_hits = sorted(traced.seen & protected)
            pv_path_aliases = current_path_aliases(board, result.pv)
            leaf_board = board.copy(stack=True)
            for move in result.pv:
                if move not in leaf_board.legal_moves:
                    raise ValueError("search returned illegal PV")
                leaf_board.push(move)
            if pv_path_aliases[-1] != alias(leaf_board):
                raise ValueError("PV endpoint alias differs from leaf replay")
            protected_current_path = sorted(set(pv_path_aliases) & protected)
            history = prefix.copy()
            history_sha = hashlib.sha256(
                __import__("json")
                .dumps(
                    {"root_fen": root["root_fen"], "history_uci": history},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                .encode()
            ).hexdigest()
            leaf_outcome = leaf_board.outcome(claim_draw=True)
            if result.leaf.kind == "static":
                leaf_value, terminal_wdl = float(result.leaf.value), None
            elif leaf_outcome is not None:
                leaf_value = None
                terminal_wdl = (
                    0
                    if leaf_outcome.winner is None
                    else (1 if leaf_outcome.winner == leaf_board.turn else -1)
                )
            else:
                raise ValueError("PV leaf terminal flag differs from replay")
            parity = -1.0 if len(result.pv) % 2 else 1.0
            if not math.isclose(q, parity * float(result.leaf.value), rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("root search value/PV leaf parity mismatch")
            row = dict(
                root_id=root["root_id"],
                root_ordinal=ordinal,
                root_fen=root["root_fen"],
                root_prefix_uci=list(root["prefix_uci"]),
                local_ply=ply,
                history_uci=history,
                history_sha256=history_sha,
                fen4=" ".join(board.fen().split()[:4]),
                mover="white" if board.turn else "black",
                root_alias=root_alias,
                search_root_value=q,
                raw_q_mover=q,
                clipped_q_mover=max(-1.0, min(1.0, q)),
                mate_range_score_returned=abs(q) > 1,
                selected_best_uci=result.move.uci(),
                selected_action_uci=result.move.uci(),
                played_action_uci=None
                if protected_hits or protected_current_path
                else result.move.uci(),
                played_action=not (protected_hits or protected_current_path),
                selected_action_played=not (protected_hits or protected_current_path),
                nodes=result.nodes,
                evaluations=result.evaluations,
                completed_depth=result.completed_depth,
                root_actions=result.root_actions,
                actual_eval_calls=traced.calls,
                eval_aliases=sorted(traced.seen),
                protected_search_aliases=protected_hits,
                pv_path_aliases=pv_path_aliases,
                protected_current_path_aliases=protected_current_path,
                behavior_policy_available=False,
                label_source="own-frozen-parent-search",
                pv_uci=[move.uci() for move in result.pv],
                leaf_kind=result.leaf.kind,
                leaf_value_mover=leaf_value,
                leaf_search_value_mover=float(result.leaf.value),
                leaf_terminal_mover_wdl=terminal_wdl,
                leaf_fen=leaf_board.fen(),
                leaf_history_sha256=hashlib.sha256(
                    __import__("json")
                    .dumps(
                        {
                            "root_fen": root["root_fen"],
                            "history_uci": history + [move.uci() for move in result.pv],
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    .encode()
                ).hexdigest(),
                train_eligible=False,
            )
            rows.append(row)
            allrows.append(row)
            exposure.update(traced.seen)
            exposure.update(pv_path_aliases)
            on_event(dict(type="search_row", row=dict(row)))
            if protected_hits or protected_current_path:
                was_protected, status = True, "excluded-protected-trajectory"
                break
            board.push(result.move)
            prefix.append(result.move.uci())
            next_alias = alias(board)
            exposure.add(next_alias)
            if next_alias in protected:
                was_protected, status = True, "excluded-protected-trajectory"
                break
            outcome = board.outcome(claim_draw=True)
            if outcome:
                white_result = 0 if outcome.winner is None else (1 if outcome.winner else -1)
                status = "completed-own-terminal"
                break
            if len(train) + len(rows) >= row_limit:
                status = "unknown-row-budget-prefix"
                break
        final = dict(
            fen4=" ".join(board.fen().split()[:4]),
            alias=alias(board),
            history_uci=prefix.copy(),
            protected=alias(board) in protected,
        )
        exposure.add(final["alias"])
        if final["protected"]:
            was_protected, status = True, "excluded-protected-trajectory"
        if was_protected:
            _label(rows, status)
        else:
            _label(rows, status, white_result)
            train.extend(rows)
        games.append(dict(root_id=root["root_id"], status=status, rows=len(rows)))
        on_event(
            dict(
                type="game_end",
                root_id=root["root_id"],
                status=status,
                final_state=final,
                training_eligible=not was_protected,
                row_labels=[row["own_wdl_mover"] for row in rows],
            )
        )
        if len(train) == row_limit:
            return dict(
                schema="human-prior-tdleaf-labels-v2",
                protection_scope_schema=PROTECTION_SCOPE,
                seed=seed,
                selected_root_order_sha256=order_sha,
                rows=allrows,
                training_row_ids=[row["root_id"] + ":" + str(row["local_ply"]) for row in train],
                games=games,
                exposed_piece_aliases=sorted(exposure | ancestry_context),
                unique_exposure_aliases=len(exposure | ancestry_context),
                neural_exposure_aliases=sorted(exposure),
                unique_neural_exposure_aliases=len(exposure),
                ancestral_context_aliases=sorted(ancestry_context),
                unique_ancestral_context_aliases=len(ancestry_context),
                ancestral_protected_intersections=sorted(ancestry_context & protected),
                starts_considered=ordinal + 1,
                target_source=(
                    "searched-PV-leaf TD(lambda=0.5); own WDL only at completed rule-terminal games"
                ),
                teacher_labels_used=False,
                status="PASS-exact-row-budget",
            )
        if len(train) > row_limit:
            raise ValueError("row budget overshoot")
    raise ValueError("fixed root pool exhausted before exact row budget")


class AliasWriter:
    """Sorted unique int64 input aliases; chunks are individually capped at 8 MiB."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.i = 0
        self.used = 0
        self.f = None

    def append(self, values):
        values = sorted(set(values))
        raw = struct.pack(f"<{len(values)}q", *values) if values else b""
        if len(raw) > CHUNK:
            raise ValueError("one alias row exceeds chunk cap")
        if self.f is None or self.used + len(raw) > CHUNK:
            if self.f:
                self.f.flush()
                self.f.close()
                self.i += 1
            self.name = f"search-aliases-{self.i:04d}.bin"
            self.f = (self.directory / self.name).open("xb")
            self.used = 0
        ref = dict(file=self.name, offset_bytes=self.used, count=len(values))
        self.f.write(raw)
        self.f.flush()
        __import__("os").fsync(self.f.fileno())
        self.used += len(raw)
        return ref

    def close(self):
        if self.f:
            self.f.flush()
            __import__("os").fsync(self.f.fileno())
            self.f.close()
            self.f = None

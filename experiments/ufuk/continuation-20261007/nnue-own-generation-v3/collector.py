"""Deterministic current-parent own-play/Q collector; pure collection API."""

import hashlib
import math
import struct
from pathlib import Path

import chess

POOL = "own-generation-train-roots-v3"
ROWS, ROOTS, PLIES, NODES, QDEPTH, DEPTH = 1024, 128, 16, 8192, 2, 8
CHUNK = 8 * 2**20


def alias(board):
    # Conservative placement equivalence: ignore side/rules/history; mirror-canonicalize.
    b = min(board.board_fen(), board.mirror().board_fen()).encode("ascii")
    return int.from_bytes(hashlib.sha256(b).digest()[:8], "little", signed=True)


def replay(root):
    if root.get("role") != "TRAIN" or root.get("root_fen") != chess.STARTING_FEN:
        raise ValueError("full standard-start TRAIN history required")
    if not isinstance(root.get("root_id"), str) or not isinstance(
        root.get("prefix_uci"), list
    ):
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


def starts(pool, seed, protected, n=ROOTS, pool_size=4096):
    if (
        pool.get("schema") != POOL
        or pool.get("selection_status") != "pass"
        or pool.get("train_only") is not True
    ):
        raise ValueError("sealed TRAIN-only pool required")
    rs = pool.get("rows")
    if (
        not isinstance(rs, list)
        or len(rs) != pool_size
        or not isinstance(pool.get("source_selection_sha256"), str)
        or len(pool["source_selection_sha256"]) != 64
        or not isinstance(pool.get("ancestral_teacher_labels_sha256"), str)
        or len(pool["ancestral_teacher_labels_sha256"]) != 64
    ):
        raise ValueError("fixed root pool count")
    ids, histories, usable = set(), set(), []
    for r in rs:
        b = replay(r)
        h = hashlib.sha256(
            (r["root_fen"] + "\n" + " ".join(r["prefix_uci"])).encode()
        ).hexdigest()
        if (
            r["root_id"] in ids
            or h in histories
            or not isinstance(r.get("trajectory_id"), str)
            or not isinstance(r.get("source_row_id"), str)
        ):
            raise ValueError("duplicate root/history")
        ids.add(r["root_id"])
        histories.add(h)
        if alias(b) not in protected:
            usable.append(r)
    if len(usable) < n:
        raise ValueError("fixed pool exhausted; no validation/autofill")
    chosen = sorted(
        usable,
        key=lambda r: hashlib.sha256(
            f"own-nnue-ownq-start-v1|{seed}|{r['root_id']}".encode()
        ).digest(),
    )[:n]
    return chosen, hashlib.sha256(
        "\n".join(r["root_id"] for r in chosen).encode()
    ).hexdigest()


class Traced:
    def __init__(self, fn):
        self.fn, self.seen, self.calls = fn, set(), 0

    def reset(self):
        self.seen.clear()
        self.calls = 0

    def __call__(self, board):
        self.seen.add(alias(board))
        self.calls += 1
        x = float(self.fn(board))
        if not math.isfinite(x) or abs(x) > 1:
            raise ValueError("static mover W-L range")
        return x


def _label(rows, status, white_result=None):
    for r in rows:
        r["episode_status"] = status
        r["own_wdl_mover"] = (
            None
            if white_result is None
            else (white_result if r["mover"] == "white" else -white_result)
        )
        r["outcome_status"] = (
            "UNKNOWN" if white_result is None else "known-own-terminal"
        )
        r["train_eligible"] = (
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
    train, allrows, games, exposure = [], [], [], set()
    for ordinal, root in enumerate(roots):
        guard()
        board = replay(root)
        prefix = list(root["prefix_uci"])
        rows, was_protected = [], False
        on_event(
            dict(
                type="game_start",
                root_id=root["root_id"],
                root_ordinal=ordinal,
                root_fen=root["root_fen"],
                root_prefix_uci=list(prefix),
                root_alias=alias(board),
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
                white_result = (
                    0 if outcome.winner is None else (1 if outcome.winner else -1)
                )
                status = "completed-own-terminal"
                break
            root_alias = alias(board)
            exposure.add(root_alias)
            if root_alias in protected:
                was_protected = True
                status = "excluded-protected-trajectory"
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
            r = dict(
                root_id=root["root_id"],
                root_ordinal=ordinal,
                root_fen=root["root_fen"],
                root_prefix_uci=list(root["prefix_uci"]),
                local_ply=ply,
                history_uci=prefix.copy(),
                fen4=" ".join(board.fen().split()[:4]),
                mover="white" if board.turn else "black",
                root_alias=root_alias,
                raw_q_mover=q,
                clipped_q_mover=max(-1.0, min(1.0, q)),
                mate_range_score_returned=abs(q) > 1,
                selected_best_uci=result.move.uci(),
                nodes=result.nodes,
                evaluations=result.evaluations,
                completed_depth=result.completed_depth,
                root_actions=result.root_actions,
                actual_eval_calls=traced.calls,
                eval_aliases=sorted(traced.seen),
                protected_search_aliases=protected_hits,
                selected_action_played=not protected_hits,
                behavior_policy_available=False,
                label_source="own-frozen-parent-search",
                train_eligible=False,
            )
            rows.append(r)
            allrows.append(r)
            exposure.update(traced.seen)
            on_event(dict(type="search_row", row=dict(r)))
            if r["protected_search_aliases"]:
                was_protected = True
                status = "excluded-protected-trajectory"
                break
            board.push(result.move)
            prefix.append(result.move.uci())
            next_alias = alias(board)
            exposure.add(next_alias)
            if next_alias in protected:
                was_protected = True
                status = "excluded-protected-trajectory"
                break
            outcome = board.outcome(claim_draw=True)
            if outcome:
                white_result = (
                    0 if outcome.winner is None else (1 if outcome.winner else -1)
                )
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
        if was_protected:
            _label(rows, status)
            games.append(dict(root_id=root["root_id"], status=status, rows=len(rows)))
            on_event(
                dict(
                    type="game_end",
                    root_id=root["root_id"],
                    status=status,
                    final_state=final,
                    training_eligible=False,
                    row_labels=[r["own_wdl_mover"] for r in rows],
                )
            )
            continue
        _label(rows, status, white_result)
        train.extend(rows)
        games.append(dict(root_id=root["root_id"], status=status, rows=len(rows)))
        on_event(
            dict(
                type="game_end",
                root_id=root["root_id"],
                status=status,
                final_state=final,
                training_eligible=True,
                row_labels=[r["own_wdl_mover"] for r in rows],
            )
        )
        if len(train) == row_limit:
            return dict(
                schema="own-nnue-ownq-labels-v3",
                seed=seed,
                selected_root_order_sha256=order_sha,
                rows=allrows,
                training_row_ids=[
                    r["root_id"] + ":" + str(r["local_ply"]) for r in train
                ],
                games=games,
                exposed_piece_aliases=sorted(exposure),
                starts_considered=ordinal + 1,
                target_source="own-search; own WDL only for completed episodes",
                teacher_labels_used=False,
                status="PASS-exact-row-budget",
            )
        if len(train) > row_limit:
            raise ValueError("row budget overshoot")
    raise ValueError("fixed root pool exhausted before exact row budget")


class AliasWriter:
    """Sorted unique int64 search-input aliases; each binary file <=8MiB."""

    def __init__(self, directory):
        self.d = Path(directory)
        self.i = 0
        self.used = 0
        self.f = None
        self.name = None

    def append(self, vals):
        vals = sorted(set(vals))
        raw = struct.pack(f"<{len(vals)}q", *vals) if vals else b""
        if len(raw) > CHUNK:
            raise ValueError("one alias row exceeds chunk cap")
        if self.f is None or self.used + len(raw) > CHUNK:
            if self.f:
                self.f.flush()
                self.f.close()
                self.i += 1
            self.name = f"search-aliases-{self.i:04d}.bin"
            self.f = (self.d / self.name).open("xb")
            self.used = 0
        ref = dict(file=self.name, offset_bytes=self.used, count=len(vals))
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

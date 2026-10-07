"""ROOT-clocked paired HUMAN PRIOR search-only development, no SGD/model labels."""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path

import chess
import chess.engine

ADV = "cfea021c9fa226424774daa07ee7c066f61b198fad61e5d76068a6e4017ca5f9"
OLD = "de53c14728a67b7772f18b396ac8ef5c35a144d4e4e616fec40099cd461a6670"
SF = "0f83d24cc46d2c66c60f16001af5444873bc112b7d028594513426894c12da19"
END = 1791448916.685839
RECORD = struct.Struct("<32sqd")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(262144), b""):
            h.update(b)
    return h.hexdigest()


def canonical(x):
    return json.dumps(
        x, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def pinned(ref):
    p = Path(ref["path"])
    if not p.is_file() or sha(p) != ref["sha256"]:
        raise ValueError("immutable input/source SHA")
    return p


def load(ref, name):
    p = pinned(ref)
    s = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


def replay(root_fen, history):
    if root_fen != chess.STARTING_FEN:
        raise ValueError("full standard-start history required")
    b = chess.Board(root_fen)
    for uci in history:
        move = chess.Move.from_uci(uci)
        if move not in b.legal_moves:
            raise ValueError("illegal fullhistory")
        b.push(move)
    if not b.is_valid() or b.outcome(claim_draw=True):
        raise ValueError("valid nonterminal start")
    return b


def packet(r, board):
    if (
        r.move is None
        or r.move not in board.legal_moves
        or r.root_actions != board.legal_moves.count()
        or not r.root_actions + 1 <= r.nodes <= 512
        or not 0 <= r.evaluations <= r.nodes
        or not 0 <= r.completed_depth <= 8
    ):
        raise ValueError("legal root/charged counters")
    out = dict(
        move=r.move.uci(),
        value_hex=float(r.value).hex(),
        nodes=r.nodes,
        evaluations=r.evaluations,
        completed_depth=r.completed_depth,
        root_actions=r.root_actions,
    )
    for k in (
        "tt_hits",
        "pvs_researches",
        "q_check_extensions",
        "incheck_horizon_returns",
        "budget_exhausted",
        "principal_variation",
        "completed_root_passes",
        "root_width_mode",
        "full_legal_completed_depth",
        "completed_root_coverage",
        "attempted_depth",
        "attempted_root_actions",
        "completed_attempted_root_actions",
        "depth_semantics",
        "root_estimate_semantics",
        "completed_root_estimates",
    ):
        if hasattr(r, k):
            out[k] = getattr(r, k)
    return out


class Trace:
    """All ordered evaluations: full-history/rule-state SHA + piece alias + float64 value."""

    def __init__(self, out, fn):
        self.out = out
        self.fn = fn
        self.chunk = 0
        self.used = 0
        self.f = None
        self.path = None
        self.rows = 0
        self.root = None
        self.cost_ns = 0

    def begin(self, board):
        self.root = board.root().fen()
        self.raw = bytearray()
        self.rows = 0
        self.cost_ns = 0

    def __call__(self, b):
        value = float(self.fn(b))
        first = time.perf_counter_ns()
        state = hashlib.sha256(
            (
                self.root
                + "\n"
                + " ".join(m.uci() for m in b.move_stack)
                + "\n"
                + b.fen()
            ).encode()
        ).digest()
        placement = min(b.board_fen(), b.mirror().board_fen()).encode()
        alias = int.from_bytes(
            hashlib.sha256(placement).digest()[:8], "little", signed=True
        )
        self.raw.extend(RECORD.pack(state, alias, value))
        self.rows += 1
        self.cost_ns += time.perf_counter_ns() - first
        return value

    def finish(self):
        if len(self.raw) > 8 * 2**20:
            raise ValueError("one search trace8MiB")
        if self.f is None or self.used + len(self.raw) > 8 * 2**20:
            self.close()
            self.path = self.out / f"eval-trace-{self.chunk:04d}.bin"
            self.chunk += 1
            self.f = self.path.open("xb")
            self.used = 0
        ref = dict(
            file=self.path.name,
            offset_bytes=self.used,
            count=self.rows,
            bytes=len(self.raw),
            sha256=hashlib.sha256(self.raw).hexdigest(),
            record_schema="full-rule-history-sha256/piece-alias-i64/value-f64-le-v1",
        )
        self.f.write(self.raw)
        self.f.flush()
        os.fsync(self.f.fileno())
        self.used += len(self.raw)
        return ref

    def close(self):
        if self.f:
            self.f.flush()
            self.f.close()
            self.f = None


def clock(reg, now):
    cap = 600 if reg["mode"] == "profile" else 7200
    if (
        reg["schema"] != "progressive-width-human-prior-development-registration-v1"
        or reg["status"] != "registered"
        or reg["mode"] not in ("profile", "arena")
        or type(reg["cpu_core"]) is not int
        or reg["cpu_core"] not in range(os.cpu_count() or 1)
        or reg["advanced_incheck_extensions"] not in (0, 1)
        or not reg["first"]
        <= now
        < reg["deadline"]
        <= min(reg["first"] + cap, reg["operator_end_epoch"], END)
        or reg["old_search"]["sha256"] != OLD
        or reg["advanced_search"]["sha256"] != ADV
        or reg["helper_sha256"] != sha(__file__)
        or reg["search_math"] != dict(nodes=512, qdepth=2, max_depth=8)
        or reg["root_width"] != "progressive"
    ):
        raise ValueError("ROOT new paired-development clock/source/search registration")
    if reg["mode"] == "arena" and reg["advanced_incheck_extensions"] != 1:
        raise ValueError(
            "horizon0 is profile-only optional ablation, no unregistered games"
        )


def prepare_profile_roots(reg):
    pool = json.loads(pinned(reg["train_roots"]).read_bytes())
    if (
        pool["schema"]
        not in ("teacher-selected-ownq-train-roots-v2", "own-generation-train-roots-v3")
        or pool["train_only"] is not True
        or pool["selection_status"] != "pass"
        or len(pool["rows"]) != 4096
    ):
        raise ValueError("fixed whole4096 TRAIN projection")
    rows = pool["rows"][
        :24
    ]  # fixed original order, no score/terminal/outcome filtering
    if len({r["root_id"] for r in rows}) != 24 or any(
        r["role"] != "TRAIN" for r in rows
    ):
        raise ValueError("exact first24 distinct TRAIN IDs")
    return [(r, replay(r["root_fen"], r["prefix_uci"])) for r in rows]


def sf_packet(result, board):
    nodes = result.info.get("nodes")
    if (
        type(nodes) is not int
        or nodes < 1
        or result.move is None
        or result.move not in board.legal_moves
    ):
        raise ValueError("actual SF UCI move/node packet")
    return dict(
        move=result.move.uci(),
        actual_nodes=nodes,
        nominal_nodes=512,
        overrun_nodes=max(0, nodes - 512),
        depth=result.info.get("depth"),
        nps=result.info.get("nps"),
        time=result.info.get("time"),
    )


def execute(reg, reg_path):
    clock(reg, time.time())
    monotonic_deadline = time.monotonic() + reg["deadline"] - time.time()
    os.sched_setaffinity(0, {reg["cpu_core"]})
    core = Path(reg["core_repo"])
    if subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=core, text=True
    ).strip() != reg["core_commit"] or subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=core, text=True
    ):
        raise ValueError("clean original core source closure")
    sys.path.insert(0, str(core / "src"))
    from harbichess.training.cgroup_budget import CgroupMemoryBudget

    memory = CgroupMemoryBudget(15 * 2**30)
    old = load(reg["old_search"], "paired_dev_de53")
    new = load(reg["advanced_search"], "paired_dev_advanced")
    prior = load(reg["prior_helper"], "paired_dev_human18")
    value = prior.ClassicalValue()
    if value.weights != tuple(prior.PRIOR) or any(value.theta):
        raise ValueError("literal unchanged humanprior18")
    out = Path(reg["output"])
    if not out.resolve().is_relative_to("/dev/shm"):
        raise ValueError("RAM-only artifacts")
    out.mkdir(parents=True, exist_ok=False)
    records = []
    engine = None
    trace = Trace(out, value.nonterminal)
    snapshots = {
        str(pinned(ref)): ref["sha256"]
        for ref in (reg["old_search"], reg["advanced_search"], reg["prior_helper"])
    }
    for name in ("book", "train_roots", "stockfish"):
        if name in reg:
            snapshots[str(pinned(reg[name]))] = reg[name]["sha256"]
    snapshots[str(reg_path)] = sha(reg_path)

    def guard():
        clock(reg, time.time())
        if time.monotonic() >= monotonic_deadline:
            raise TimeoutError("original monotonic phase budget expired")
        memory.check()
        if shutil.disk_usage("/workspace").free < 256 * 2**20:
            raise RuntimeError("primary disk floor256MiB")
        if sum(p.stat().st_size for p in out.iterdir() if p.is_file()) > 256 * 2**20:
            raise RuntimeError("RAM256MiB artifact ceiling")

    searches = dict(
        de53=old.BudgetSearch(
            trace, nodes=512, quiescence_plies=2, max_depth=8, guard=guard
        ),
        advanced=new.BudgetSearch(
            trace,
            nodes=512,
            quiescence_plies=2,
            max_depth=8,
            guard=guard,
            incheck_extensions=reg["advanced_incheck_extensions"],
            root_width=reg["root_width"],
        ),
    )
    progress = (out / "progress.jsonl").open("xb")

    def emit(x):
        progress.write(canonical(x) + b"\n")
        progress.flush()
        os.fsync(progress.fileno())

    def search(role, b):
        guard()
        before = b.fen(), tuple(b.move_stack)
        trace.begin(b)
        first = time.perf_counter()
        result = searches[role].search(b)
        wall = time.perf_counter() - first
        r = packet(result, b)
        if result.evaluations != trace.rows or before != (b.fen(), tuple(b.move_stack)):
            raise ValueError("actual eval calls/history invariant")
        r.update(
            role=role,
            wall_seconds=wall,
            trace_build_seconds=trace.cost_ns / 1e9,
            eval_trace=trace.finish(),
            preaction_fen=b.fen(),
            history_uci=[m.uci() for m in b.move_stack],
        )
        emit(dict(type="search", packet=r))
        guard()
        return result, r

    result = dict(
        schema="progressive-width-human-prior-development-result-v1",
        status="FAILED-preserved",
    )
    try:
        guard()
        if reg["mode"] == "profile":
            for ordinal, (root, b) in enumerate(prepare_profile_roots(reg)):
                row = dict(ordinal=ordinal, source_row_id=root["root_id"], packets={})
                for role in (
                    ("de53", "advanced") if ordinal % 2 == 0 else ("advanced", "de53")
                ):
                    _, row["packets"][role] = search(role, b)
                records.append(row)
            ratios = [
                x["packets"]["advanced"]["wall_seconds"]
                / x["packets"]["de53"]["wall_seconds"]
                for x in records
            ]
            result.update(
                status="PASS-24-paired-TRAIN-searches-not-strength",
                paired_rows=records,
                median_wall_ratio=statistics.median(ratios),
                latency_within1p10=statistics.median(ratios) <= 1.10,
                selected_move_changes=sum(
                    x["packets"]["advanced"]["move"] != x["packets"]["de53"]["move"]
                    for x in records
                ),
                alias_trace_every_evaluation=True,
            )
        else:
            # Arena is admitted ONLY from same-source actual frozen profile.
            profile = json.loads(pinned(reg["profile_result"]).read_bytes())
            if (
                profile["status"] != "PASS-24-paired-TRAIN-searches-not-strength"
                or profile["advanced_incheck_extensions"] != 1
                or profile["root_width"] != "progressive"
                or len(profile["paired_rows"]) != 24
                or not profile["first"] <= profile["finished"] <= profile["deadline"]
                or profile["helper_sha256"] != sha(__file__)
                or profile["latency_within1p10"] is not True
                or profile["source_pins"]["advanced_search"] != reg["advanced_search"]
                or profile["source_pins"]["old_search"] != reg["old_search"]
                or profile["source_pins"]["prior_helper"] != reg["prior_helper"]
            ):
                raise ValueError(
                    "actual same-source profile before32 development games"
                )
            if reg["stockfish"]["sha256"] != SF:
                raise ValueError("official fixed SF19 SHA")
            book = json.loads(pinned(reg["book"]).read_bytes())["splits"]["arena"]
            if len(book) != 8:
                raise ValueError("exact known8 book, no root selection")
            openings = [r["opening"]["moves"] for r in book]
            for moves in openings:
                replay(chess.STARTING_FEN, moves)
            engine = chess.engine.SimpleEngine.popen_uci(
                str(pinned(reg["stockfish"])), timeout=15
            )
            engine.configure({"Threads": 1, "Hash": 16})
            for pair, moves in enumerate(openings):
                for color in (chess.WHITE, chess.BLACK):
                    for role in ("de53", "advanced"):
                        guard()
                        engine.configure({"Clear Hash": None})
                        b = replay(chess.STARTING_FEN, moves)
                        game_id = (pair, color, role)
                        searches_by_move = []
                        sf = []
                        first = time.time()
                        emit(
                            dict(
                                type="game_start",
                                role=role,
                                pair=pair,
                                color="white" if color else "black",
                                opening=moves,
                                first_epoch=first,
                            )
                        )
                        while b.outcome(claim_draw=True) is None and b.ply() < 400:
                            guard()
                            if b.turn == color:
                                selected, pk = search(role, b)
                                move = selected.move
                                searches_by_move.append(pk)
                            else:
                                started = time.perf_counter()
                                # Only actual UCI counters consumed, no teacher scores.
                                played = engine.play(
                                    b,
                                    chess.engine.Limit(nodes=512),
                                    game=game_id,
                                    info=chess.engine.INFO_BASIC,
                                )
                                pk = sf_packet(played, b)
                                pk["wall_seconds"] = time.perf_counter() - started
                                pk["preaction_fen"] = b.fen()
                                pk["ply"] = b.ply()
                                sf.append(pk)
                                move = played.move
                                emit(
                                    dict(
                                        type="stockfish",
                                        packet=pk,
                                        role=role,
                                        pair=pair,
                                    )
                                )
                            if move not in b.legal_moves:
                                raise ValueError("illegal actual move")
                            b.push(move)
                            emit(
                                dict(
                                    type="move",
                                    role=role,
                                    pair=pair,
                                    ply=b.ply(),
                                    uci=move.uci(),
                                )
                            )
                        outcome = b.outcome(claim_draw=True)
                        known = outcome is not None
                        score = (
                            None
                            if not known
                            else 0.5
                            if outcome.winner is None
                            else float(outcome.winner == color)
                        )
                        row = dict(
                            role=role,
                            pair=pair,
                            color="white" if color else "black",
                            opening=moves,
                            moves=[m.uci() for m in b.move_stack],
                            score=score,
                            outcome_status="KNOWN" if known else "UNKNOWN",
                            termination=outcome.termination.name
                            if known
                            else "total-ply-cap400",
                            searches=searches_by_move,
                            stockfish=sf,
                            first_epoch=first,
                            finished_epoch=time.time(),
                        )
                        records.append(row)
                        emit(dict(type="game_end", game=row))
            result.update(
                status="PASS-32-fixed-development-games-search-only-not-selflearning",
                games=records,
                summary={
                    role: dict(
                        games=16,
                        unknown=sum(
                            x["score"] is None for x in records if x["role"] == role
                        ),
                        score_drawfilled=sum(
                            0.5 if x["score"] is None else x["score"]
                            for x in records
                            if x["role"] == role
                        )
                        / 16,
                        score_unknown_pessimistic=sum(
                            x["score"] or 0 for x in records if x["role"] == role
                        )
                        / 16,
                    )
                    for role in ("de53", "advanced")
                },
            )
        guard()
        for path, digest in snapshots.items():
            if sha(path) != digest:
                raise ValueError("immutable source/input changed during phase")
    except BaseException as exc:
        result["error"] = repr(exc)
        raise
    finally:
        trace.close()
        progress.close()
        if engine is not None:
            try:
                engine.quit()
            except Exception as exc:
                engine.close()
                result["close_error"] = type(exc).__name__
                result["status"] = "FAILED-preserved"
        if time.time() >= reg["deadline"]:
            result["status"] = "FAILED-preserved"
            result["finish_error"] = "original phase expired"
        result.update(
            mode=reg["mode"],
            root_width=reg["root_width"],
            first=reg["first"],
            deadline=reg["deadline"],
            finished=time.time(),
            source_pins={
                k: reg[k] for k in ("old_search", "advanced_search", "prior_helper")
            },
            registration_sha256=sha(reg_path),
            helper_sha256=sha(__file__),
            GPU_used=False,
            cpu_core=reg["cpu_core"],
            actual_cpu_affinity=sorted(os.sched_getaffinity(0)),
            advanced_incheck_extensions=reg["advanced_incheck_extensions"],
            selflearning_claim=False,
            trace_inventory=[
                dict(file=p.name, bytes=p.stat().st_size, sha256=sha(p))
                for p in sorted(out.glob("eval-trace-*.bin"))
            ],
            budget_definition=(
                "charged own recursive nodes inclq/check/TT; SF nominal512 actual overruns logged"
            ),
            completed_depth_note="progressive depth is DECLARED root-subset depth; full_legal_completed_depth separate; static-only=0; de53 labels fallback1",
        )
        result["finished"] = time.time()
        if (
            result["finished"] >= reg["deadline"]
            or time.monotonic() >= monotonic_deadline
        ):
            result["status"] = "FAILED-preserved"
            result["finish_error"] = "original phase expired including closure/readback"
        with (out / "result.json").open("xb") as f:
            f.write(canonical(result) + b"\n")
        if result["status"] == "FAILED-preserved" and "error" not in result:
            raise RuntimeError(
                result.get("finish_error", result.get("close_error", "phase failed"))
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registration", type=Path, required=True)
    a = p.parse_args()
    execute(json.loads(a.registration.read_bytes()), a.registration)

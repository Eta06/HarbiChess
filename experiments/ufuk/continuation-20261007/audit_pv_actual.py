"""ROOT-only integrity and six actual replays; never a strength gate."""

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import chess


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(262144), b""):
            h.update(b)
    return h.hexdigest()


def ref(p):
    return {"path": str(p), "sha256": sha(p)}


def pinned(r):
    p = Path(r["path"])
    if sha(p) != r["sha256"]:
        raise ValueError("input SHA changed")
    return p


def load(r, name):
    s = importlib.util.spec_from_file_location(name, pinned(r))
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


def execute(reg, registration):
    first = reg["first"]
    end = reg["deadline"]
    if not first <= time.time() < end <= min(first + 900, 1791448916.685839):
        raise ValueError("new ROOT actual audit clock")
    if reg["helper_sha256"] != sha(__file__):
        raise ValueError("ROOT helper pin")
    os.sched_setaffinity(0, {2})
    dev = load(reg["develop"], "root_pv_develop")
    registrations = {
        k: json.loads(pinned(reg[k + "_registration"]).read_bytes())
        for k in ("profile", "arena")
    }
    results = {
        k: json.loads(pinned(reg[k + "_result"]).read_bytes())
        for k in ("profile", "arena")
    }
    for k, r in registrations.items():
        j = results[k]
        if (
            j["registration_sha256"] != reg[k + "_registration"]["sha256"]
            or j["helper_sha256"] != reg["develop"]["sha256"]
            or not r["first"]
            == j["first"]
            <= j["finished"]
            <= j["deadline"]
            == r["deadline"]
            or j["advanced_incheck_extensions"] != 1
        ):
            raise ValueError("actual closed source and original clock")
        for name in ("old_search", "advanced_search", "prior_helper"):
            pinned(r[name])
            if r[name] != registrations["arena"][name]:
                raise ValueError("source roles changed")
    out = Path(reg["output"])
    out.mkdir(parents=True, exist_ok=False)

    def guard():
        if not first <= time.time() < end:
            raise TimeoutError("original audit clock")

    def board(history):
        b = chess.Board()
        for u in history:
            m = chess.Move.from_uci(u)
            if m not in b.legal_moves or b.outcome(claim_draw=True):
                raise ValueError("illegal history or move after terminal")
            b.push(m)
        return b

    trace_refs = []
    ordered = {"profile": [], "arena": []}
    for row in results["profile"]["paired_rows"]:
        roles = (
            ("de53", "advanced") if row["ordinal"] % 2 == 0 else ("advanced", "de53")
        )
        ordered["profile"].extend(row["packets"][r] for r in roles)
    for g in results["arena"]["games"]:
        ordered["arena"].extend(g["searches"])
    for k, packets in ordered.items():
        base = pinned(reg[k + "_result"]).parent
        offsets = {}
        inventory = {x["file"]: x for x in results[k]["trace_inventory"]}
        for name, r in inventory.items():
            p = base / name
            if p.stat().st_size != r["bytes"] or sha(p) != r["sha256"]:
                raise ValueError("all trace chunk bytes")
            trace_refs.append(ref(p))
        for p in packets:
            guard()
            b = board(p["history_uci"])
            if b.fen() != p["preaction_fen"] or b.outcome(claim_draw=True):
                raise ValueError("packet rule history")
            if (
                p["move"] not in [m.uci() for m in b.legal_moves]
                or p["root_actions"] != b.legal_moves.count()
                or not 1 + p["root_actions"] <= p["nodes"] <= 512
                or not 0 <= p["evaluations"] <= p["nodes"]
                or not 0 <= p["completed_depth"] <= 8
            ):
                raise ValueError("charged search counters and legal action")
            t = p["eval_trace"]
            name = t["file"]
            if t["count"] != p["evaluations"] or t["bytes"] != 48 * t["count"]:
                raise ValueError("all ordered evaluations counted")
            if t["offset_bytes"] != offsets.get(name, 0):
                raise ValueError("trace gaps or duplicates")
            with (base / name).open("rb") as f:
                f.seek(t["offset_bytes"])
                raw = f.read(t["bytes"])
            if len(raw) != t["bytes"] or hashlib.sha256(raw).hexdigest() != t["sha256"]:
                raise ValueError("every original trace segment bytes")
            offsets[name] = t["offset_bytes"] + t["bytes"]
        if offsets != {n: x["bytes"] for n, x in inventory.items()}:
            raise ValueError("all chunks completely accounted")
    games = results["arena"]["games"]
    expected = [
        (p, c, r)
        for p in range(8)
        for c in ("white", "black")
        for r in ("de53", "advanced")
    ]
    if [(g["pair"], g["color"], g["role"]) for g in games] != expected:
        raise ValueError("exact32 paired original order")
    book = json.loads(pinned(registrations["arena"]["book"]).read_bytes())["splits"][
        "arena"
    ]
    plies = sf_nodes = max_overrun = 0
    for g in games:
        guard()
        if (
            g["opening"] != book[g["pair"]]["opening"]["moves"]
            or g["moves"][: len(g["opening"])] != g["opening"]
        ):
            raise ValueError("known book exact prefix")
        b = board(g["opening"])
        color = g["color"] == "white"
        si = fi = 0
        for u in g["moves"][len(g["opening"]) :]:
            if b.outcome(claim_draw=True) or b.ply() >= 400:
                raise ValueError("postterminal or cap move")
            p = g["searches"][si] if b.turn == color else g["stockfish"][fi]
            if p["preaction_fen"] != b.fen() or p["move"] != u:
                raise ValueError("actual played search/SF packet")
            if b.turn == color:
                if (
                    p["history_uci"] != [m.uci() for m in b.move_stack]
                    or p["role"] != g["role"]
                ):
                    raise ValueError("full own history binding")
                si += 1
            else:
                if (
                    p["ply"] != b.ply()
                    or p["actual_nodes"] < 1
                    or p["nominal_nodes"] != 512
                    or p["overrun_nodes"] != max(0, p["actual_nodes"] - 512)
                ):
                    raise ValueError("actual UCI node accounting")
                sf_nodes += p["actual_nodes"]
                max_overrun = max(max_overrun, p["overrun_nodes"])
                fi += 1
            m = chess.Move.from_uci(u)
            if m not in b.legal_moves:
                raise ValueError("illegal game move")
            b.push(m)
            plies += 1
        if si != len(g["searches"]) or fi != len(g["stockfish"]):
            raise ValueError("unused/missing move packets")
        o = b.outcome(claim_draw=True)
        expected_score = (
            None if o is None else 0.5 if o.winner is None else float(o.winner == color)
        )
        if (
            g["score"] != expected_score
            or g["outcome_status"] != ("UNKNOWN" if o is None else "KNOWN")
            or g["termination"]
            != ("total-ply-cap400" if o is None else o.termination.name)
        ):
            raise ValueError("actual terminal/censored result")
        if o is None and b.ply() != 400:
            raise ValueError("unknown is exact cap, never fabricated draw")
    prior = load(
        registrations["arena"]["prior_helper"], "root_pv_prior"
    ).ClassicalValue()
    modules = {
        "de53": load(registrations["arena"]["old_search"], "root_pv_old"),
        "advanced": load(registrations["arena"]["advanced_search"], "root_pv_new"),
    }
    selected = [
        ("profile", results["profile"]["paired_rows"][i]["packets"][r])
        for i in (0, 1)
        for r in ("de53", "advanced")
    ]
    selected += [
        (
            "arena",
            next(
                g
                for g in games
                if g["pair"] == 0 and g["color"] == "white" and g["role"] == r
            )["searches"][0],
        )
        for r in ("de53", "advanced")
    ]
    replays = []
    for k, p in selected:
        guard()
        b = board(p["history_uci"])
        before = b.fen(), tuple(b.move_stack)
        trace = dev.Trace(out, prior.nonterminal)
        trace.begin(b)
        opts = dict(nodes=512, quiescence_plies=2, max_depth=8, guard=guard)
        if p["role"] == "advanced":
            opts["incheck_extensions"] = 1
        result = modules[p["role"]].BudgetSearch(trace, **opts).search(b)
        actual = dev.packet(result, b)
        if any(actual[x] != p[x] for x in actual) or before != (
            b.fen(),
            tuple(b.move_stack),
        ):
            raise ValueError("six actual semantic search bit mismatch")
        t = p["eval_trace"]
        src = pinned(reg[k + "_result"]).parent / t["file"]
        with src.open("rb") as f:
            f.seek(t["offset_bytes"])
            raw = f.read(t["bytes"])
        if bytes(trace.raw) != raw or trace.rows != p["evaluations"]:
            raise ValueError("six actual complete ordered eval trace mismatch")
        replays.append(
            dict(
                mode=k,
                role=p["role"],
                history=p["history_uci"],
                packet=actual,
                trace_sha256=hashlib.sha256(raw).hexdigest(),
                match=True,
            )
        )
    summary = {
        r: {
            "games": 16,
            "unknown": sum(g["score"] is None for g in games if g["role"] == r),
            "score_drawfilled": sum(
                0.5 if g["score"] is None else g["score"]
                for g in games
                if g["role"] == r
            )
            / 16,
        }
        for r in ("de53", "advanced")
    }
    if any(
        summary[r][x] != results["arena"]["summary"][r][x]
        for r in summary
        for x in summary[r]
    ):
        raise ValueError("all role summaries recomputed")
    guard()
    record = dict(
        schema="ROOT-pv-search-only-independent-actual-audit-v1",
        status="PASS-integrity-and-six-actual-not-strength",
        first=first,
        deadline=end,
        finished=time.time(),
        registration=ref(registration),
        helper=ref(__file__),
        closed_games=32,
        checked_played_plies=plies,
        actual_stockfish_nodes=sf_nodes,
        maximum_stockfish_node_overrun=max_overrun,
        complete_ordered_trace_refs=trace_refs,
        six_actual_replays=replays,
        summary=summary,
        observed_search_score_gain=summary["advanced"]["score_drawfilled"]
        - summary["de53"]["score_drawfilled"],
        selflearning_success=False,
        inputs=reg,
    )
    (out / "result.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                x: record[x]
                for x in (
                    "status",
                    "closed_games",
                    "checked_played_plies",
                    "observed_search_score_gain",
                )
            }
        )
    )


if __name__ == "__main__":
    path = Path(sys.argv[1])
    execute(json.loads(path.read_bytes()), path)

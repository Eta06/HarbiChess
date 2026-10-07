"""Strict own-Q/fullhistory/terminal trace converter; no forward, engine, SGD or teacher."""

import argparse
import hashlib
import importlib.util
import json
import math
import struct
import time
from pathlib import Path

import chess


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pinned(row):
    p = Path(row["path"])
    if sha(p) != row["sha256"]:
        raise ValueError("sealed input/source SHA")
    return p


def pins_tree(value):
    """Validate every explicit path/SHA leaf, including raw sidecar maps."""
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            pinned(value)
        for child in value.values():
            pins_tree(child)
    elif isinstance(value, list):
        for child in value:
            pins_tree(child)


def module(row, name):
    spec = importlib.util.spec_from_file_location(name, pinned(row))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def board_at(row):
    if row["root_fen"] != chess.STARTING_FEN:
        raise ValueError("standard fullhistory start")
    b = chess.Board(row["root_fen"])
    for uci in row["history_uci"]:
        move = chess.Move.from_uci(uci)
        if not b.is_legal(move):
            raise ValueError("illegal fullhistory")
        b.push(move)
    if b.outcome(claim_draw=True) is not None:
        raise ValueError("training root must be nonterminal")
    return b


def alias(b):
    return int.from_bytes(
        hashlib.sha256(min(b.board_fen(), b.mirror().board_fen()).encode()).digest()[:8],
        "little",
        signed=True,
    )


def verify_row(row, features, prior):
    b = board_at(row)
    if (
        row["fen4"] != " ".join(b.fen().split()[:4])
        or row["mover"] != ("white" if b.turn else "black")
        or row["root_alias"] != alias(b)
        or not b.is_legal(chess.Move.from_uci(row["selected_best_uci"]))
    ):
        raise ValueError("mover/root/selected-action legality")
    q = row["raw_q_mover"]
    if (
        not math.isfinite(q)
        or abs(q) > 2
        or row["clipped_q_mover"] != max(-1.0, min(1.0, q))
        or row["mate_range_score_returned"] != (abs(q) > 1)
        or row["behavior_policy_available"] is not False
        or row["label_source"] != "own-frozen-parent-search"
    ):
        raise ValueError("own Q proxy/clip/provenance; not calibrated WDL")
    if (
        type(row["root_actions"]) is not int
        or row["root_actions"] != b.legal_moves.count()
        or type(row["nodes"]) is not int
        or not row["root_actions"] + 1 <= row["nodes"] <= 8192
        or type(row["evaluations"]) is not int
        or row["evaluations"] != row["actual_eval_calls"]
        or not 0 <= row["evaluations"] <= row["nodes"]
        or type(row["completed_depth"]) is not int
        or not 0 <= row["completed_depth"] <= 8
    ):
        raise ValueError("completed root search/global actual counters")
    prior_logit = (
        sum(w * x for w, x in zip(prior.PRIOR, prior.features(b), strict=True)) / prior.SCALE
    )
    return dict(
        indices=features.board_indices(b),
        prior_logit=prior_logit,
        target=float(row["clipped_q_mover"]),
    )


def convert(spec, guard=lambda: None):
    if spec["schema"] != "human-randomstarts-own1024-dataset-conversion-seal-v2":
        raise ValueError("own seal required")
    reg = json.loads(pinned(spec["registration"]).read_bytes())
    receipt = json.loads(pinned(spec["receipt"]).read_bytes())
    if (
        reg["schema"] != "human-randomstarts-own-collection-registration-v2"
        or reg["status"] != "registered"
        or receipt["schema"] != "human-randomstarts-own-collection-receipt-v2"
        or receipt["status"] != "PASS-exact-row-budget"
        or receipt["train_rows"] != 1024
        or receipt["registration_sha256"] != spec["registration"]["sha256"]
        or receipt["parent_candidate_sha256"] != reg["parent_candidate"]["sha256"]
        or receipt["teacher_labels_used"] is not False
        or receipt["search"] != dict(nodes=8192, qdepth=2, max_depth=8)
        or receipt["root_pool_sha256"] != reg["root_pool"]["sha256"]
        or receipt["protected_aliases_sha256"] != reg["protected_aliases"]["sha256"]
        or not reg["original_first_epoch"]
        <= receipt["finished_epoch"]
        <= receipt["original_deadline_epoch"]
        == reg["original_deadline_epoch"]
        <= reg["operator_end_epoch"]
    ):
        raise ValueError("exact closed current-parent own-search1024 lineage")
    for entry in [
        reg["parent_candidate"],
        reg["root_pool"],
        reg["protected_aliases"],
        reg["search_helper"],
    ]:
        pinned(entry)
    for name, digest in reg["producer_source_sha256"].items():
        pinned(dict(path=str(Path(spec["producer_directory"]) / name), sha256=digest))
    if receipt["generation_helper_sha256"] != reg["generation_helper_sha256"]:
        raise ValueError("generation source closure receipt differs")
    for path, digest in reg["generation_helper_sha256"].items():
        pinned(dict(path=path, sha256=digest))
    protected_raw = pinned(reg["protected_aliases"]).read_bytes()
    if len(protected_raw) % 8:
        raise ValueError("int64 protected aliases")
    protected_values = list(struct.unpack(f"<{len(protected_raw) // 8}q", protected_raw))
    if protected_values != sorted(set(protected_values)):
        raise ValueError("canonical protected aliases")
    protected = set(protected_values)
    if (
        spec["features"]["sha256"] != reg["parent_helpers"]["model_sha256"]
        or Path(spec["features"]["path"]).resolve()
        != (Path(reg["parent_helpers"]["directory"]) / "model.py").resolve()
    ):
        raise ValueError("exact frozen parent feature origin")
    if (
        receipt["events_sha256"] != spec["events"]["sha256"]
        or receipt["operator_end_epoch"] != reg["operator_end_epoch"]
        or receipt["producer_source_sha256"] != reg["producer_source_sha256"]
        or receipt["parent_helpers"] != reg["parent_helpers"]
    ):
        raise ValueError("complete producer/event/parent receipt bindings")
    event_path = pinned(spec["events"])
    if (
        event_path.stat().st_size != receipt["events_bytes"]
        or event_path.stat().st_size > 16 * 2**20
    ):
        raise ValueError("event bytes/cap")
    chunks, spans = {}, {}
    if len(spec["alias_chunks"]) > 32:
        raise ValueError("sidecar count cap")
    inventory = []
    for name, ref in spec["alias_chunks"].items():
        if (
            Path(name).name != name
            or not name.startswith("search-aliases-")
            or not name.endswith(".bin")
        ):
            raise ValueError("fixed sidecar basename")
        raw = pinned(ref).read_bytes()
        if len(raw) > 8 * 2**20 or len(raw) % 8:
            raise ValueError("sidecar int64/8MiB cap")
        chunks[name], spans[name] = raw, []
        inventory.append(dict(file=name, bytes=len(raw), sha256=ref["sha256"]))
    if sorted(inventory, key=lambda x: x["file"]) != sorted(
        receipt["alias_chunks"], key=lambda x: x["file"]
    ):
        raise ValueError("exact full sidecar receipt inventory")
    collector = module(
        dict(
            path=str(Path(spec["producer_directory"]) / "collector.py"),
            sha256=reg["producer_source_sha256"]["collector.py"],
        ),
        "own_converter_collector",
    )
    pool = json.loads(pinned(reg["root_pool"]).read_bytes())
    pool_ref = reg["root_pool"]
    selection_ref = dict(path=pool_ref["selection_path"], sha256=pool_ref["selection_sha256"])
    from root_bank import verify

    if spec["procedural_bank_receipt"] != reg["procedural_bank_receipt"]:
        raise ValueError("explicit procedural source seal")
    original_selection, bank = verify(reg["procedural_bank_receipt"], reg["protected_aliases"])
    if (
        bank["selection"] != selection_ref
        or pool["source_selection_sha256"] != selection_ref["sha256"]
        or receipt["procedural_selection_path"] != selection_ref["path"]
        or receipt["procedural_selection_sha256"] != selection_ref["sha256"]
        or pool["procedural_receipt_sha256"] != reg["procedural_bank_receipt"]["sha256"]
        or receipt["procedural_bank_receipt"] != reg["procedural_bank_receipt"]
    ):
        raise ValueError("immutable rule-only procedural TRAIN source")
    from parent_bridge import require_collection_parent, validate_admission_result

    admitted_seal, admitted_contract = validate_admission_result(
        reg["parent_admission_seal"], reg["parent_admission_result"]
    )
    require_collection_parent(reg, admitted_seal, admitted_contract)
    if (
        reg["generation"] != receipt["generation"]
        or reg["parent_admission_result"] != receipt["parent_admission_result"]
        or reg["parent_admission_seal"] != receipt["parent_admission_seal"]
    ):
        raise ValueError("same admitted current own-parent and generation")
    if len(original_selection["rows"]) != 4096 or len(pool["rows"]) != 4096:
        raise ValueError("whole original4096 pool")
    for source, projected in zip(original_selection["rows"], pool["rows"], strict=True):
        if projected != source:
            raise ValueError("exact complete procedural source row projection")
    chosen, order_sha = collector.starts(pool, reg["seed"], protected)
    if order_sha != receipt["selected_root_order_sha256"]:
        raise ValueError("fixed outcome-blind root order")
    feature = module(spec["features"], "own_conversion_features")
    prior = module(spec["prior"], "own_conversion_prior")
    if spec["prior"]["sha256"] != reg["parent_helpers"]["prior_sha256"]:
        raise ValueError("original authoritative parent prior")
    events = [json.loads(line) for line in pinned(spec["events"]).read_bytes().splitlines()]
    starts = {}
    rows = {}
    ends = {}
    aliases = set()
    selected = {}
    active = None
    exposure = set()
    for event in events:
        guard()
        kind = event["type"]
        key = event.get("root_id")
        if kind == "game_start":
            if key in starts or active is not None:
                raise ValueError("duplicate trajectory start")
            ordinal = len(starts)
            root = chosen[ordinal]
            if (
                event["root_ordinal"] != ordinal
                or key != root["root_id"]
                or event["root_fen"] != root["root_fen"]
                or event["root_prefix_uci"] != root["prefix_uci"]
            ):
                raise ValueError("exact chosen root/full prefix")
            start_board = collector.replay(root)
            if event["root_alias"] != alias(start_board):
                raise ValueError("actual starting alias")
            exposure.add(alias(start_board))
            active = key
            starts[key] = event
            rows[key] = []
        elif kind == "search_row":
            row = event["row"]
            key = row["root_id"]
            if key not in starts or key in ends or active != key:
                raise ValueError("chronological active game")
            sequence = rows[key]
            start = starts[key]
            if any(x["selected_action_played"] is not True for x in sequence):
                raise ValueError("no action after discarded protected search row")
            prefix = start["root_prefix_uci"] + [x["selected_best_uci"] for x in sequence]
            if (
                row["local_ply"] != len(sequence)
                or row["history_uci"] != prefix
                or row["root_prefix_uci"] != start["root_prefix_uci"]
                or row["root_fen"] != start["root_fen"]
                or row["root_ordinal"] != start["root_ordinal"]
            ):
                raise ValueError("exact chronological complete-history row chain")
            ref = row["search_alias_ref"]
            file = ref["file"]
            if file not in spec["alias_chunks"]:
                raise ValueError("unsealed alias chunk")
            raw = chunks[file]
            off = ref["offset_bytes"]
            n = ref["count"]
            if (
                type(off) is not int
                or type(n) is not int
                or off < 0
                or n < 0
                or off % 8
                or off + 8 * n > len(raw)
            ):
                raise ValueError("exact alias offset/count")
            spans[file].append((off, off + 8 * n))
            vals = list(struct.unpack(f"<{n}q", raw[off : off + 8 * n]))
            if vals != sorted(set(vals)) or n > row["actual_eval_calls"]:
                raise ValueError("sorted unique evaluator traces")
            if (
                row["protected_search_aliases"] != sorted(set(vals) & protected)
                or type(row["selected_action_played"]) is not bool
                or row["selected_action_played"] != (not row["protected_search_aliases"])
            ):
                raise ValueError("exact protected search witnesses/action played")
            aliases.update(vals)
            exposure.update(vals)
            exposure.add(row["root_alias"])
            row["_aliases"] = vals
            sequence.append(row)
        elif kind == "game_end":
            if key not in starts or key in ends or active != key:
                raise ValueError("one closed episode each")
            ends[key] = event
            active = None
        else:
            raise ValueError("failed/unknown producer event is not closed data")
    for name, ranges in spans.items():
        cursor = 0
        for begin, finish in ranges:
            if begin != cursor:
                raise ValueError("sidecar span overlap/gap/reorder")
            cursor = finish
        if cursor != len(chunks[name]):
            raise ValueError("unreferenced sidecar suffix")
    if set(starts) != set(ends):
        raise ValueError("partial games cannot be sealed")
    trace = []
    for key, sequence in rows.items():
        guard()
        end = ends[key]
        hit = any(r["root_alias"] in protected or set(r["_aliases"]) & protected for r in sequence)
        if sequence:
            final = board_at(sequence[-1])
            if sequence[-1]["selected_action_played"]:
                final.push_uci(sequence[-1]["selected_best_uci"])
        else:
            final = chess.Board(starts[key]["root_fen"])
            for u in starts[key]["root_prefix_uci"]:
                final.push_uci(u)
        expected_final = dict(
            fen4=" ".join(final.fen().split()[:4]),
            alias=alias(final),
            history_uci=[m.uci() for m in final.move_stack],
            protected=alias(final) in protected,
        )
        if end["final_state"] != expected_final:
            raise ValueError("full exact final-board/history packet")
        exposure.add(alias(final))
        hit = hit or alias(final) in protected
        outcome = final.outcome(claim_draw=True)
        status = end["status"]
        eligible = end["training_eligible"]
        if len(sequence) > 16 or type(eligible) is not bool:
            raise ValueError("fixed ply cap/episode eligibility")
        if hit != (status == "excluded-protected-trajectory"):
            raise ValueError("protected status must match actual path/evaluator witnesses")
        if hit:
            if eligible:
                raise ValueError("whole trajectory protected path/branch leakage")
            labels = [None] * len(sequence)
        elif status == "completed-own-terminal":
            if outcome is None or not eligible:
                raise ValueError("actual own terminal required")
            white = 0 if outcome.winner is None else (1 if outcome.winner else -1)
            labels = [white if r["mover"] == "white" else -white for r in sequence]
        elif status in ["unknown-ply-cap", "unknown-row-budget-prefix"]:
            if status == "unknown-ply-cap" and len(sequence) != 16:
                raise ValueError("actual fixed16 cap")
            if status == "unknown-row-budget-prefix" and key != list(ends)[-1]:
                raise ValueError("only final game may reach fixed row budget")
            if outcome is not None or not eligible:
                raise ValueError("UNKNOWN cap is not terminal draw")
            labels = [None] * len(sequence)
        else:
            raise ValueError("explicit censor/terminal status")
        if end["row_labels"] != labels:
            raise ValueError("terminal alternating mover WDL")
        for r, z in zip(sequence, labels, strict=True):
            parsed = verify_row(r, feature, prior)
            rid = key + ":" + str(r["local_ply"])
            if eligible:
                selected[rid] = parsed
            trace.append(
                dict(
                    row_id=rid,
                    history_sha256=hashlib.sha256(
                        canonical(dict(root_fen=r["root_fen"], prefix_uci=r["history_uci"]))
                    ).hexdigest(),
                    raw_q_mover=r["raw_q_mover"],
                    target=r["clipped_q_mover"],
                    own_wdl_mover=z,
                    outcome_status="UNKNOWN" if z is None else "known-own-terminal",
                    episode_status=status,
                    train_eligible=eligible,
                )
            )
    expected_games = [
        dict(root_id=k, status=ends[k]["status"], rows=len(v)) for k, v in rows.items()
    ]
    if (
        receipt["starts_considered"] != len(starts)
        or receipt["games"] != expected_games
        or receipt["exposed_board_alias_count"] != len(exposure)
    ):
        raise ValueError("all episode/exposure counts reconcile")
    ids = receipt["training_row_ids"]
    if (
        len(ids) != 1024
        or len(set(ids)) != 1024
        or ids != list(selected)
        or receipt["periodic_independent_search_rows"]
        != [ids[i] for i in (0, 204, 409, 614, 819, 1023)]
        or receipt["all_actor_rows"] != sum(map(len, rows.values()))
    ):
        raise ValueError("exact all-row reconciliation/1024 eligible rows")
    data = (
        canonical(
            dict(
                schema="own-kingbucket-sparse-training-data-v1",
                phase="own-learning",
                rows=[selected[i] for i in ids],
            )
        )
        + b"\n"
    )
    provenance = dict(
        schema="human-randomstarts-own1024-data-provenance-v2",
        inputs=spec,
        seed=reg["seed"],
        rows=1024,
        dataset_sha256=hashlib.sha256(data).hexdigest(),
        trace=trace,
        teacher_labels_used=False,
        target=(
            "clipped completed current-parent root-search mover Q; terminal WDL audited separately"
        ),
        mate_score_semantics=(
            "producer search mate-range score, not an independent mate certificate"
        ),
        initializer="current own-parent WEIGHTS only; newAdam/globalTorch/privateSampler",
        generation=reg["generation"],
        parent_admission_result=reg["parent_admission_result"],
        protected_scope="entire trajectory actual-path and traced static-evaluator aliases",
        unique_search_aliases=len(aliases),
        collection_receipt_sha256=spec["receipt"]["sha256"],
        parent_candidate=reg["parent_candidate"],
    )
    return data, canonical(provenance) + b"\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seal", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    spec = json.loads(a.seal.read_bytes())
    first, end = spec["first"], spec["deadline"]
    if (
        spec["status"] != "registered"
        or not first <= time.time() < end <= min(first + 600, spec["operator_end_epoch"])
        or spec["operator_end_epoch"] > 1791448916.685839
    ):
        raise ValueError("ROOT original600")
    import shutil
    import sys

    sys.path.insert(0, spec["core_repo"] + "/src")
    from harbichess.training.cgroup_budget import CgroupMemoryBudget

    memory = CgroupMemoryBudget(15 * 2**30)

    def guard():
        memory.check()
        if time.time() >= end:
            raise TimeoutError("same original converter600")
        if shutil.disk_usage("/workspace").free < 256 * 2**20:
            raise RuntimeError("workspace floor")

    data, provenance = convert(spec, guard)
    guard()
    if len(data) + len(provenance) > 96 * 2**20 or not a.output.resolve().is_relative_to(
        "/dev/shm"
    ):
        raise ValueError("RAM96MiB phase cap")
    a.output.mkdir(parents=True, exist_ok=False)
    for name, raw in [("dataset.json", data), ("provenance.json", provenance)]:
        with (a.output / name).open("xb") as stream:
            stream.write(raw)
    guard()
    with (a.output / "result.json").open("x") as stream:
        json.dump(
            dict(
                status="PASS-own1024-fullhistory-trace-conversion-not-strength",
                first=first,
                deadline=end,
                finished=time.time(),
                dataset_sha256=sha(a.output / "dataset.json"),
                provenance_sha256=sha(a.output / "provenance.json"),
            ),
            stream,
            sort_keys=True,
        )


if __name__ == "__main__":
    main()

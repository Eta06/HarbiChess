"""Reconvert the already sealed ownQ-v2 games as selected-action afterstates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path

import chess

DATA_SCHEMA = "own-kingbucket-afterstate-search-q-data-v1"
PROVENANCE_SCHEMA = "NNUE-own1024-afterstate-search-q-provenance-v1"
PHASE = "own-afterstate-search-q-learning-v1"
TARGET_METHOD = "own-selected-action-afterstate-negated-search-q-v1"


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned(ref):
    path = Path(ref["path"]).resolve()
    if sha(path) != ref["sha256"]:
        raise ValueError("pinned input/source SHA differs")
    return path


def check_tree(value):
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            pinned(value)
        for child in value.values():
            check_tree(child)
    elif isinstance(value, list):
        for child in value:
            check_tree(child)


def module(ref, name):
    path = pinned(ref)
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    if Path(loaded.__file__).resolve() != path:
        raise ValueError("module origin differs")
    return loaded


def board_from(root_fen, history):
    if root_fen != chess.STARTING_FEN:
        raise ValueError("full standard-start history required")
    board = chess.Board(root_fen)
    for token in history:
        move = chess.Move.from_uci(token)
        if move not in board.legal_moves:
            raise ValueError("illegal chronological history")
        board.push(move)
    return board


def afterstate_target(row, child):
    """Return afterstate side-to-move target and its explicit source kind."""
    outcome = child.outcome(claim_draw=True)
    if outcome is not None:
        target = 0 if outcome.winner is None else (1 if outcome.winner == child.turn else -1)
        return float(target), "exact-terminal-afterstate-wdl"
    q = float(row["raw_q_mover"])
    if not math.isfinite(q) or abs(q) > 2:
        raise ValueError("finite source root mover Q required")
    root_clipped = max(-1.0, min(1.0, q))
    return -root_clipped, "negated-selected-root-search-q"


def convert(spec, guard=lambda: None):
    if (
        spec.get("schema") != "NNUE-own-afterstate1024-conversion-seal-v1"
        or spec.get("status") != "registered"
    ):
        raise ValueError("registered afterstate conversion seal required")
    check_tree(spec)
    source_converter_ref = spec["source_converter"]
    source_spec_path = pinned(spec["source_conversion_seal"])
    source_result_path = pinned(spec["source_conversion_result"])
    source_dataset_path = pinned(spec["source_dataset"])
    source_provenance_path = pinned(spec["source_provenance"])
    source_spec = json.loads(source_spec_path.read_bytes())
    source_result = json.loads(source_result_path.read_bytes())
    if source_spec.get("schema") != "NNUE-own1024-dataset-conversion-seal-v2":
        raise ValueError("exact original ownQ-v2 converter seal required")
    old_converter = module(source_converter_ref, "pinned_original_ownq_v2_converter")
    source_data_bytes, source_provenance_bytes = old_converter.convert(source_spec, guard)
    if (
        source_result.get("status") != "PASS-own1024-fullhistory-trace-conversion-not-strength"
        or source_result.get("dataset_sha256") != sha(source_dataset_path)
        or source_result.get("provenance_sha256") != sha(source_provenance_path)
        or source_data_bytes != source_dataset_path.read_bytes()
        or source_provenance_bytes != source_provenance_path.read_bytes()
    ):
        raise ValueError("original ownQ-v2 conversion must replay byte-for-byte")
    source_data = json.loads(source_data_bytes)
    source_provenance = json.loads(source_provenance_bytes)
    receipt = json.loads(pinned(source_spec["receipt"]).read_bytes())
    registration = json.loads(pinned(source_spec["registration"]).read_bytes())
    if (
        registration.get("schema") != "own-nnue-ownq-collection-registration-v2"
        or receipt.get("schema") != "own-nnue-ownq-collection-receipt-v2"
        or receipt.get("status") != "PASS-exact-row-budget"
        or receipt.get("train_rows") != 1024
        or registration.get("seed") != spec.get("seed")
        or source_provenance.get("schema") != "NNUE-own1024-converted-data-provenance-v1"
        or source_provenance.get("dataset_sha256") != sha(source_dataset_path)
        or source_provenance.get("teacher_labels_used") is not False
        or len(source_data.get("rows", [])) != 1024
        or len(receipt.get("training_row_ids", [])) != 1024
    ):
        raise ValueError("exact original ownQ-v2 1,024-row lineage required")
    if source_provenance.get("collection_receipt_sha256") != source_spec["receipt"]["sha256"]:
        raise ValueError("source provenance does not bind original receipt")
    parent = registration["parent_candidate"]
    pinned_parent = pinned(spec["parent_candidate"])
    if (
        parent.get("sha256") != spec["parent_candidate"]["sha256"]
        or Path(parent["path"]).resolve() != pinned_parent
    ):
        raise ValueError("same named teacher-initialized ownQ parent required")
    feature = module(spec["features"], "afterstate_original_nnue_features")
    prior = module(spec["prior"], "afterstate_original_prior")
    if (
        spec["features"]["sha256"] != registration["parent_helpers"]["model_sha256"]
        or spec["prior"]["sha256"] != registration["parent_helpers"]["prior_sha256"]
    ):
        raise ValueError("exact original feature/prior helper origin")
    event_path = pinned(source_spec["events"])
    events = [json.loads(line) for line in event_path.read_bytes().splitlines()]
    source_rows = {}
    starts, ends = {}, {}
    game_rows = {}
    active = None
    for event in events:
        guard()
        kind = event.get("type")
        if kind == "game_start":
            key = event["root_id"]
            if active is not None or key in starts:
                raise ValueError("chronological unique game start required")
            starts[key] = event
            game_rows[key] = []
            active = key
        elif kind == "search_row":
            row = event["row"]
            key = row["root_id"]
            if active != key or key not in starts or key in ends:
                raise ValueError("row must appear within its actual game history")
            row_id = f"{key}:{row['local_ply']}"
            if row_id in source_rows or row["local_ply"] != len(game_rows[key]):
                raise ValueError("unique chronological source row required")
            played = row.get("selected_action_played")
            if type(played) is not bool:
                raise ValueError("source action-played status must be explicit")
            if game_rows[key] and game_rows[key][-1]["selected_action_played"] is not True:
                raise ValueError("discarded protected row must end its source episode")
            board = board_from(row["root_fen"], row["history_uci"])
            if (
                row["root_fen"] != starts[key]["root_fen"]
                or row["root_prefix_uci"] != starts[key]["root_prefix_uci"]
                or row["history_uci"]
                != starts[key]["root_prefix_uci"] + [x["selected_best_uci"] for x in game_rows[key]]
                or row["mover"] != ("white" if board.turn else "black")
                or row["fen4"] != " ".join(board.fen().split()[:4])
                or not board.is_legal(chess.Move.from_uci(row["selected_best_uci"]))
            ):
                raise ValueError("before-state/action chronology mismatch")
            child = board.copy(stack=True)
            action = chess.Move.from_uci(row["selected_best_uci"])
            if action not in board.legal_moves:
                raise ValueError("logged selected action is not legal in the full history")
            if played:
                child.push(action)
            else:
                child = None
            raw_q = row["raw_q_mover"]
            clipped = max(-1.0, min(1.0, float(raw_q)))
            source_rows[row_id] = (row, board, child, clipped)
            game_rows[key].append(row)
        elif kind == "game_end":
            key = event["root_id"]
            if active != key or key in ends:
                raise ValueError("one matching game end required")
            ends[key] = event
            active = None
        else:
            raise ValueError("unexpected/censored event format")
    if active is not None or set(starts) != set(ends):
        raise ValueError("incomplete source games cannot be converted")
    ids = receipt["training_row_ids"]
    if (
        len(ids) != 1024
        or len(set(ids)) != 1024
        or any(row_id not in source_rows for row_id in ids)
    ):
        raise ValueError("exact original ownQ eligible row list required")
    if any(source_rows[row_id][0].get("selected_action_played") is not True for row_id in ids):
        raise ValueError("only actually played actions can become afterstate training rows")
    original_trace = {item["row_id"]: item for item in source_provenance.get("trace", [])}
    if any(row_id not in original_trace for row_id in ids):
        raise ValueError("original ownQ target/censor trace must cover selected rows")
    selected = []
    trace = []
    for index, row_id in enumerate(ids):
        guard()
        row, before, child, clipped = source_rows[row_id]
        game_end = ends[row["root_id"]]
        if not game_end.get("training_eligible"):
            raise ValueError("protected/unknown excluded episode cannot supply afterstate labels")
        target, target_kind = afterstate_target(row, child)
        after_history = [*row["history_uci"], row["selected_best_uci"]]
        if index + 1 < len(ids):
            next_row = source_rows[ids[index + 1]][0]
            if (
                next_row["root_id"] == row["root_id"]
                and next_row["local_ply"] == row["local_ply"] + 1
                and next_row["history_uci"] != after_history
            ):
                raise ValueError("afterstate differs from next actual actor position")
        indices = feature.board_indices(child)
        prior_logit = (
            sum(w * x for w, x in zip(prior.PRIOR, prior.features(child), strict=True))
            / prior.SCALE
        )
        record = {
            "source_row_id": row_id,
            "source_game_id": row["root_id"],
            "source_local_ply": row["local_ply"],
            "source_root_fen": row["root_fen"],
            "source_history_uci": list(row["history_uci"]),
            "source_history_sha256": hashlib.sha256(
                canonical({"root_fen": row["root_fen"], "prefix_uci": row["history_uci"]})
            ).hexdigest(),
            "indices": indices,
            "prior_logit": prior_logit,
            "raw_q_mover": float(row["raw_q_mover"]),
            "target": target,
            "target_kind": target_kind,
            "source_outcome_status": original_trace[row_id]["outcome_status"],
            "before_mover": row["mover"],
            "after_mover": "white" if child.turn else "black",
            "selected_action_uci": row["selected_best_uci"],
            "selected_action_played": True,
            "afterstate_terminal": child.outcome(claim_draw=True) is not None,
            "afterstate_fen4": " ".join(child.fen().split()[:4]),
            "afterstate_history_uci": after_history,
            "afterstate_history_sha256": hashlib.sha256(
                canonical({"root_fen": row["root_fen"], "prefix_uci": after_history})
            ).hexdigest(),
        }
        selected.append(record)
        trace.append(
            {
                "source_row_id": row_id,
                "source_history_sha256": hashlib.sha256(
                    canonical({"root_fen": row["root_fen"], "prefix_uci": row["history_uci"]})
                ).hexdigest(),
                "afterstate_history_sha256": record["afterstate_history_sha256"],
                "root_search_q_mover": float(row["raw_q_mover"]),
                "clipped_root_q_mover": clipped,
                "target": target,
                "target_kind": target_kind,
                "selected_action_played": True,
                "afterstate_terminal": child.outcome(claim_draw=True) is not None,
                "source_outcome_status": original_trace[row_id]["outcome_status"],
            }
        )
    if source_data["rows"]:
        for i, row_id in enumerate(ids):
            source_row, before, _child, _clipped = source_rows[row_id]
            source_data_row = source_data["rows"][i]
            expected_prior = (
                sum(w * x for w, x in zip(prior.PRIOR, prior.features(before), strict=True))
                / prior.SCALE
            )
            if (
                source_data_row["target"] != source_row["clipped_q_mover"]
                or source_data_row["indices"] != feature.board_indices(before)
                or source_data_row["prior_logit"] != expected_prior
            ):
                raise ValueError(
                    "original ownQ order/before-state features/value differ from raw source"
                )
    data = (
        canonical(
            {
                "schema": DATA_SCHEMA,
                "phase": PHASE,
                "target_method": TARGET_METHOD,
                "rows": selected,
            }
        )
        + b"\n"
    )
    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "target_method": TARGET_METHOD,
        "output_train_rows": 1024,
        "teacher_labels_used": False,
        "native_phase_required": PHASE,
        "native_schema_required": "own-kingbucket-nnue16-afterstate-full-native-cpu-v1",
        "legacy_native_resume_allowed": False,
        "source_ownq_provenance_schema": source_provenance["schema"],
        "source_ownq_dataset_schema": source_data["schema"],
        "source_ownq_dataset_sha256": sha(source_dataset_path),
        "source_ownq_provenance_sha256": sha(source_provenance_path),
        "source_converter_sha256": source_converter_ref["sha256"],
        "source_conversion_result_sha256": sha(source_result_path),
        "source_receipt_sha256": source_spec["receipt"]["sha256"],
        "collection_receipt_sha256": source_spec["receipt"]["sha256"],
        "source_events_sha256": source_spec["events"]["sha256"],
        "parent_candidate": parent,
        "seed": spec["seed"],
        "dataset_sha256": hashlib.sha256(data).hexdigest(),
        "target_perspective": "afterstate side-to-move",
        "target_rule": (
            "nonterminal=-clamp(source root mover Q); "
            "terminal=exact outcome from child side-to-move"
        ),
        "root_q_semantics": (
            "source ownQ v2 completed root search, mover perspective; "
            "selected best action actually played"
        ),
        "root_q_mate_semantics": (
            "root Q is clipped before sign flip; terminal child exact WDL overrides it"
        ),
        "unknown_source_rows_used_as_q": any(
            item["source_outcome_status"] == "UNKNOWN" for item in trace
        ),
        "unknown_q_rows": sum(item["source_outcome_status"] == "UNKNOWN" for item in trace),
        "protected_or_not_played_rows_excluded": True,
        "trace": trace,
        "inputs": spec,
    }
    return data, canonical(provenance) + b"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.seal.read_bytes())
    if args.output.exists() or not args.output.resolve().is_relative_to("/dev/shm"):
        raise ValueError("publish-once RAM output required")
    first, deadline, operator_end = (spec["first"], spec["deadline"], spec["operator_end_epoch"])
    if (
        spec["status"] != "registered"
        or not first <= time.time() < deadline <= min(first + 600, operator_end)
        or operator_end > 1791448916.685839
    ):
        raise TimeoutError("ROOT original afterstate conversion clock required")
    import shutil
    import sys

    sys.path.insert(0, spec["core_repo"] + "/src")
    from harbichess.training.cgroup_budget import CgroupMemoryBudget

    budget = CgroupMemoryBudget(15 * 2**30)

    def guard():
        budget.check()
        if time.time() >= deadline:
            raise TimeoutError("original afterstate conversion clock")
        if shutil.disk_usage("/workspace").free < 256 * 2**20:
            raise RuntimeError("workspace disk floor")

    guard()
    data, provenance = convert(spec, guard)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, raw in (("dataset.json", data), ("provenance.json", provenance)):
        with (args.output / name).open("xb") as stream:
            stream.write(raw)
    guard()
    with (args.output / "result.json").open("x") as stream:
        json.dump(
            {
                "status": "PASS-afterstate-ownQ-fullhistory-conversion-not-strength",
                "first": first,
                "deadline": deadline,
                "finished": time.time(),
                "dataset_sha256": sha(args.output / "dataset.json"),
                "provenance_sha256": sha(args.output / "provenance.json"),
            },
            stream,
            sort_keys=True,
        )


if __name__ == "__main__":
    main()

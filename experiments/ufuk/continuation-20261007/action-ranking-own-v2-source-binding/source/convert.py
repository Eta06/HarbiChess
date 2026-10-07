"""Rebuild legal-child ranking data from the sealed ownQ-v2 search trace."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import chess
from contract import DATA_SCHEMA, PHASE, PROVENANCE_SCHEMA, TARGET_METHOD

SEAL_SCHEMA = "NNUE-own-action-ranking1024-conversion-seal-v1"
OPERATOR_END_MAX = 1791448916.685839


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


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


def load_module(ref, name):
    path = pinned(ref)
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    if path.name == "convert.py" and (path.parent / "contract.py").is_file():
        helper_path = path.parent / "contract.py"
        helper_spec = importlib.util.spec_from_file_location(name + "_contract", helper_path)
        helper = importlib.util.module_from_spec(helper_spec)
        helper_spec.loader.exec_module(helper)
        previous = sys.modules.get("contract")
        sys.modules["contract"] = helper
        try:
            spec.loader.exec_module(loaded)
        finally:
            if previous is None:
                sys.modules.pop("contract", None)
            else:
                sys.modules["contract"] = previous
    else:
        spec.loader.exec_module(loaded)
    if Path(loaded.__file__).resolve() != path:
        raise ValueError("pinned module origin differs")
    return loaded


module = load_module


def board_from(root_fen, history):
    if root_fen != chess.STARTING_FEN:
        raise ValueError("complete standard-start game history required")
    board = chess.Board(root_fen)
    for token in history:
        move = chess.Move.from_uci(token)
        if move not in board.legal_moves:
            raise ValueError("illegal chronological full history")
        board.push(move)
    if board.outcome(claim_draw=True) is not None:
        raise ValueError("selected source root must be nonterminal")
    return board


def history_sha(root_fen, history):
    return hashlib.sha256(canonical({"root_fen": root_fen, "prefix_uci": history})).hexdigest()


def alias(board):
    token = min(board.board_fen(), board.mirror().board_fen()).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little", signed=True)


def alias_values(row, receipt, seal, cache):
    ref = row["search_alias_ref"]
    entries = {item["file"]: item for item in receipt["alias_chunks"]}
    entry = entries.get(ref["file"])
    source_ref = seal["alias_chunks"].get(ref["file"])
    if not entry or not source_ref:
        raise ValueError("root search alias segment must be pinned in both receipt and seal")
    filename = ref["file"]
    if filename not in cache:
        chunk = pinned(source_ref)
        payload = chunk.read_bytes()
        if len(payload) != entry["bytes"] or hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError("source root search alias chunk size/SHA differs")
        cache[filename] = payload
    chunk_bytes = cache[filename]
    offset, count = ref["offset_bytes"], ref["count"]
    if (
        type(offset) is not int
        or type(count) is not int
        or offset < 0
        or count < 0
        or offset % 8
        or offset + count * 8 > entry["bytes"]
    ):
        raise ValueError("root search alias segment bounds")
    segment = chunk_bytes[offset : offset + count * 8]
    if len(segment) != count * 8:
        raise ValueError("short alias segment")
    values = list(struct.unpack(f"<{count}q", segment))
    if values != sorted(set(values)):
        raise ValueError("search alias segment must be sorted unique int64 values")
    return values, hashlib.sha256(segment).hexdigest()


def legal_candidate(board, move, root_fen, history, feature, prior, root_aliases):
    """Build one legal class; nonterminal classes must be seen in source search."""
    if move not in board.legal_moves:
        raise ValueError("candidate action must be legal in full root history")
    child = board.copy(stack=True)
    child.push(move)
    child_history = [*history, move.uci()]
    outcome = child.outcome(claim_draw=True)
    candidate = {
        "action_uci": move.uci(),
        "after_mover": "white" if child.turn else "black",
        "afterstate_history_uci": child_history,
        "afterstate_history_sha256": history_sha(root_fen, child_history),
        "terminal": outcome is not None,
    }
    if outcome is not None:
        candidate["child_wdl"] = (
            0.0 if outcome.winner is None else (1.0 if outcome.winner == child.turn else -1.0)
        )
    else:
        child_alias = alias(child)
        if child_alias not in root_aliases:
            raise ValueError("nonterminal legal child absent from original root search trace")
        candidate.update(
            indices=feature.board_indices(child),
            prior_logit=sum(
                weight * value
                for weight, value in zip(prior.PRIOR, prior.features(child), strict=True)
            )
            / prior.SCALE,
            visited_alias=child_alias,
        )
    return candidate


def convert(spec, guard=lambda: None):
    if spec.get("schema") != SEAL_SCHEMA or spec.get("status") != "registered":
        raise ValueError("ROOT-registered distinct action-ranking conversion required")
    check_tree(spec)
    source_seal_path = pinned(spec["afterstate_conversion_seal"])
    source_result_path = pinned(spec["afterstate_conversion_result"])
    source_dataset_path = pinned(spec["afterstate_dataset"])
    source_provenance_path = pinned(spec["afterstate_provenance"])
    source_seal = json.loads(source_seal_path.read_bytes())
    converter = load_module(spec["afterstate_converter"], "pinned_afterstate_converter_for_ranking")
    after_data_bytes, after_prov_bytes = converter.convert(source_seal, guard)
    source_result = json.loads(source_result_path.read_bytes())
    if (
        source_result.get("status") != "PASS-afterstate-ownQ-fullhistory-conversion-not-strength"
        or source_result.get("dataset_sha256") != sha(source_dataset_path)
        or source_result.get("provenance_sha256") != sha(source_provenance_path)
        or after_data_bytes != source_dataset_path.read_bytes()
        or after_prov_bytes != source_provenance_path.read_bytes()
    ):
        raise ValueError("frozen afterstate source conversion must replay byte-for-byte")
    after_data = json.loads(after_data_bytes)
    after_prov = json.loads(after_prov_bytes)
    original_seal = json.loads(pinned(source_seal["source_conversion_seal"]).read_bytes())
    original_receipt = json.loads(pinned(original_seal["receipt"]).read_bytes())
    original_registration = json.loads(pinned(original_seal["registration"]).read_bytes())
    if (
        original_receipt.get("train_rows") != 1024
        or original_receipt.get("status") != "PASS-exact-row-budget"
        or original_registration.get("schema") != "own-nnue-ownq-collection-registration-v2"
        or original_registration.get("core_commit") != spec.get("core_source_commit")
        or original_seal.get("alias_chunks") is None
        or spec.get("seed") != original_registration.get("seed")
    ):
        raise ValueError("exact original ownQ-v2 1,024-row source required")

    feature = load_module(source_seal["features"], "ranking_original_feature_helper")
    prior = load_module(source_seal["prior"], "ranking_original_prior_helper")
    event_path = pinned(original_seal["events"])
    events = [json.loads(line) for line in event_path.read_bytes().splitlines()]
    source_rows = {}
    starts, ends = {}, {}
    active = None
    per_game = {}
    for event in events:
        guard()
        kind = event.get("type")
        if kind == "game_start":
            root_id = event["root_id"]
            if active is not None or root_id in starts:
                raise ValueError("ordered unique source games required")
            starts[root_id] = event
            per_game[root_id] = []
            active = root_id
        elif kind == "search_row":
            row = event["row"]
            root_id = row["root_id"]
            if active != root_id or root_id not in starts or root_id in ends:
                raise ValueError("source search row outside actual chronology")
            row_id = f"{root_id}:{row['local_ply']}"
            if row_id in source_rows or row["local_ply"] != len(per_game[root_id]):
                raise ValueError("unique chronological source row required")
            board = board_from(row["root_fen"], row["history_uci"])
            if (
                row["root_fen"] != starts[root_id]["root_fen"]
                or row["root_prefix_uci"] != starts[root_id]["root_prefix_uci"]
                or row["history_uci"]
                != starts[root_id]["root_prefix_uci"]
                + [item["selected_best_uci"] for item in per_game[root_id]]
                or row["fen4"] != " ".join(board.fen().split()[:4])
                or row["mover"] != ("white" if board.turn else "black")
                or not board.is_legal(chess.Move.from_uci(row["selected_best_uci"]))
            ):
                raise ValueError("source root/action chronology differs")
            source_rows[row_id] = row
            per_game[root_id].append(row)
        elif kind == "game_end":
            root_id = event["root_id"]
            if active != root_id or root_id in ends:
                raise ValueError("one game-end event per started episode required")
            ends[root_id] = event
            active = None
        else:
            raise ValueError("unknown/censored event type")
    if active is not None or set(starts) != set(ends):
        raise ValueError("all source game histories must close")

    source_trace = {item["source_row_id"]: item for item in after_prov["trace"]}
    alias_cache = {}
    selected_rows = []
    output_trace = []
    for selected in after_data["rows"]:
        guard()
        row_id = selected["source_row_id"]
        row = source_rows.get(row_id)
        if row is None or row.get("selected_action_played") is not True:
            raise ValueError("only actually played selected root actions may rank")
        if not ends[row["root_id"]].get("training_eligible"):
            raise ValueError("protected/discarded source game cannot provide a ranking target")
        board = board_from(row["root_fen"], row["history_uci"])
        aliases, alias_digest = alias_values(row, original_receipt, original_seal, alias_cache)
        alias_set = set(aliases)
        legal_candidates = []
        selected_index = None
        observed_child_aliases = []
        for move in board.legal_moves:
            guard()
            candidate = legal_candidate(
                board, move, row["root_fen"], row["history_uci"], feature, prior, alias_set
            )
            if not candidate["terminal"]:
                observed_child_aliases.append(candidate["visited_alias"])
            if move.uci() == row["selected_best_uci"]:
                selected_index = len(legal_candidates)
            legal_candidates.append(candidate)
        if selected_index is None:
            raise ValueError("selected played action is absent from legal candidate list")
        if (
            selected.get("selected_action_uci") != row["selected_best_uci"]
            or selected.get("selected_action_played") is not True
            or selected.get("source_history_sha256")
            != history_sha(row["root_fen"], row["history_uci"])
        ):
            raise ValueError("ranking action/Q target must match afterstate source row")
        root = {
            "source_row_id": row_id,
            "source_game_id": row["root_id"],
            "source_local_ply": row["local_ply"],
            "source_root_fen": row["root_fen"],
            "source_history_uci": list(row["history_uci"]),
            "source_history_sha256": history_sha(row["root_fen"], row["history_uci"]),
            "source_outcome_status": source_trace[row_id]["source_outcome_status"],
            "raw_q_mover": row["raw_q_mover"],
            "q_target": selected["target"],
            "target_kind": selected["target_kind"],
            "selected_action_uci": row["selected_best_uci"],
            "selected_action_played": True,
            "selected_index": selected_index,
            "candidates": legal_candidates,
            "search_alias_count": len(aliases),
            "search_alias_sha256": alias_digest,
            "observed_child_aliases": sorted(set(observed_child_aliases)),
        }
        selected_rows.append(root)
        output_trace.append(
            {
                "source_row_id": row_id,
                "selected_index": selected_index,
                "q_target": selected["target"],
                "candidate_count": len(legal_candidates),
                "observed_child_aliases_sha256": hashlib.sha256(
                    canonical(root["observed_child_aliases"])
                ).hexdigest(),
                "search_alias_count": len(aliases),
                "search_alias_sha256": alias_digest,
                "source_outcome_status": source_trace[row_id]["source_outcome_status"],
            }
        )
    if len(selected_rows) != 1024:
        raise ValueError("exact eligible 1,024 root rows required")
    data = {
        "schema": DATA_SCHEMA,
        "phase": PHASE,
        "target_method": TARGET_METHOD,
        "candidate_policy": (
            "all-legal-child-classes; only-selected-action-onehot; no-sibling-outcomes"
        ),
        "temperature": 0.25,
        "selected_action_cross_entropy_weight": 0.1,
        "rows": selected_rows,
    }
    data_bytes = canonical(data) + b"\n"
    if len(data_bytes) > 64 * 2**20:
        raise ValueError("full legal-child dataset exceeds64MiB cap")
    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "phase": PHASE,
        "target_method": TARGET_METHOD,
        "output_train_rows": 1024,
        "teacher_labels_used": False,
        "native_phase_required": PHASE,
        "native_schema_required": "own-kingbucket-nnue16-action-ranking-full-native-cpu-v1",
        "legacy_native_resume_allowed": False,
        "source_afterstate_provenance_sha256": sha(source_provenance_path),
        "dataset_sha256": hashlib.sha256(data_bytes).hexdigest(),
        "seed": spec["seed"],
        "parent_candidate": source_seal["parent_candidate"],
        "source_receipt_sha256": original_seal["receipt"]["sha256"],
        "collection_receipt_sha256": original_seal["receipt"]["sha256"],
        "unknown_source_rows_used_as_q": any(
            item["source_outcome_status"] == "UNKNOWN" for item in output_trace
        ),
        "unknown_q_rows": sum(item["source_outcome_status"] == "UNKNOWN" for item in output_trace),
        "inputs": spec,
        "trace": output_trace,
    }
    provenance_bytes = canonical(provenance) + b"\n"
    if len(provenance_bytes) > 16 * 2**20:
        raise ValueError("full lineage provenance exceeds16MiB cap")
    return data_bytes, provenance_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.seal.read_bytes())
    first, deadline, operator_end = (
        spec.get("first"),
        spec.get("deadline"),
        spec.get("operator_end_epoch"),
    )
    if (
        args.output.exists()
        or not args.output.resolve().is_relative_to("/dev/shm")
        or type(first) not in (int, float)
        or type(deadline) not in (int, float)
        or type(operator_end) not in (int, float)
        or not first <= time.time() < deadline <= min(first + 600, operator_end)
        or operator_end > OPERATOR_END_MAX
    ):
        raise ValueError("ROOT original conversion clock, RAM output, operator ceiling required")
    core = Path(spec["core_source_repo"])
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=core, text=True).strip() != spec[
        "core_source_commit"
    ] or subprocess.check_output(["git", "status", "--porcelain"], cwd=core, text=True):
        raise ValueError("clean original own-Q source pin required")
    sys.path.insert(0, str(Path(spec["core_source_repo"]) / "src"))
    from harbichess.training.cgroup_budget import CgroupMemoryBudget

    memory = CgroupMemoryBudget(15 * 2**30)

    def guard():
        memory.check()
        if not first <= time.time() < deadline:
            raise TimeoutError("original conversion phase clock; no reset")
        if shutil.disk_usage("/workspace").free < 256 * 2**20:
            raise RuntimeError("workspace disk floor256MiB")

    guard()
    data, provenance = convert(spec, guard)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "dataset.json").open("xb") as stream:
        stream.write(data)
    with (args.output / "provenance.json").open("xb") as stream:
        stream.write(provenance)
    guard()
    result = {
        "status": "PASS-action-ranking-full-replay-not-strength",
        "first": first,
        "deadline": deadline,
        "finished": time.time(),
        "dataset_sha256": sha(args.output / "dataset.json"),
        "provenance_sha256": sha(args.output / "provenance.json"),
        "rows": 1024,
    }
    with (args.output / "result.json").open("xb") as stream:
        stream.write(canonical(result) + b"\n")


if __name__ == "__main__":
    main()

"""Build a distinct TD(lambda) dataset from a sealed own-Q v2 conversion.

This module intentionally has no model, search, Stockfish, or optimizer imports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Mapping
from pathlib import Path

import chess
from tdlambda_targets import transform_episode

SOURCE_DATA_SCHEMA = "own-kingbucket-sparse-training-data-v1"
SOURCE_PROVENANCE_SCHEMA = "NNUE-own1024-converted-data-provenance-v1"
SOURCE_RECEIPT_SCHEMA = "own-nnue-ownq-collection-receipt-v2"
SOURCE_CONVERSION_STATUS = "PASS-own1024-fullhistory-trace-conversion-not-strength"
BUILD_SEAL_SCHEMA = "own-nnue-tdlambda-target-build-seal-v1"
OUTPUT_DATA_SCHEMA = "own-kingbucket-tdlambda-training-data-v1"
OUTPUT_PROVENANCE_SCHEMA = "NNUE-own1024-tdlambda-target-provenance-v1"
TRAINING_PHASE = "own-learning-tdlambda-v1"
LAMBDA = 0.5


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def alias(board: chess.Board) -> int:
    key = min(board.board_fen(), board.mirror().board_fen()).encode("ascii")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little", signed=True)


def replay(root_fen: str, history: list[str]) -> chess.Board:
    if root_fen != chess.STARTING_FEN or not isinstance(history, list):
        raise ValueError("own-Q trace requires standard-root complete UCI history")
    board = chess.Board(root_fen)
    for token in history:
        move = chess.Move.from_uci(token)
        if move not in board.legal_moves:
            raise ValueError("illegal full-history UCI move")
        board.push(move)
    if not board.is_valid():
        raise ValueError("invalid full-history state")
    return board


def _q(row: Mapping) -> float:
    raw = row.get("raw_q_mover")
    clipped = row.get("clipped_q_mover")
    if (isinstance(raw, bool) or not isinstance(raw, int | float)
            or not math.isfinite(float(raw)) or abs(float(raw)) > 2):
        raise ValueError("own search Q must be finite and within mate-range bounds")
    expected = max(-1.0, min(1.0, float(raw)))
    if clipped != expected or isinstance(clipped, bool):
        raise ValueError("clipped own-Q target differs from recorded raw value")
    return float(clipped)


def parse_final_events(events_bytes: bytes) -> tuple[list[list[dict]], list[dict]]:
    """Legality-check producer events and attach game-end WDL to each row.

    Returns episode groups and converter-compatible trace rows. Only final,
    chronologically closed producer event logs are accepted.
    """
    events = []
    for line in events_bytes.splitlines():
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("malformed producer JSONL") from exc

    episodes: list[list[dict]] = []
    expected_trace: list[dict] = []
    roots: set[str] = set()
    active: dict | None = None
    for event in events:
        kind = event.get("type")
        if kind == "game_start":
            if active is not None:
                raise ValueError("producer game overlap")
            root_id = event.get("root_id")
            if not isinstance(root_id, str) or root_id in roots:
                raise ValueError("duplicate or invalid selected root id")
            roots.add(root_id)
            root_fen = event.get("root_fen")
            root_prefix = event.get("root_prefix_uci")
            if not isinstance(root_prefix, list):
                raise ValueError("full root prefix required")
            board = replay(root_fen, root_prefix)
            if event.get("root_alias") != alias(board):
                raise ValueError("game-start board alias differs")
            active = {
                "root_id": root_id,
                "root_fen": root_fen,
                "root_prefix": list(root_prefix),
                "history": list(root_prefix),
                "board": board,
                "rows": [],
            }
        elif kind == "search_row":
            if active is None:
                raise ValueError("search row outside active trajectory")
            row = event.get("row")
            if not isinstance(row, dict):
                raise ValueError("search-row packet required")
            board = active["board"]
            local_ply = len(active["rows"])
            if (row.get("root_id") != active["root_id"]
                    or row.get("root_fen") != active["root_fen"]
                    or row.get("root_prefix_uci") != active["root_prefix"]
                    or row.get("local_ply") != local_ply
                    or row.get("history_uci") != active["history"]
                    or row.get("fen4") != " ".join(board.fen().split()[:4])
                    or row.get("mover") != ("white" if board.turn else "black")
                    or row.get("root_alias") != alias(board)):
                raise ValueError("search row is not the chronological full-history state")
            q = _q(row)
            try:
                move = chess.Move.from_uci(row["selected_best_uci"])
            except (KeyError, ValueError) as exc:
                raise ValueError("selected search move missing or malformed") from exc
            if move not in board.legal_moves:
                raise ValueError("selected search move is illegal")
            if type(row.get("selected_action_played")) is not bool:
                raise ValueError("actual action-played flag required")
            if active.get("stopped_without_play"):
                raise ValueError("no later search row may follow an unplayed protected action")
            protected_aliases = row.get("protected_search_aliases")
            if not isinstance(protected_aliases, list):
                raise ValueError("protected input witness list required")
            if row["selected_action_played"] != (not protected_aliases):
                raise ValueError("protected search witness/action-played mismatch")
            if board.outcome(claim_draw=True) is not None:
                raise ValueError("search row must be a live pre-action state")
            normalized = {
                "root_id": active["root_id"],
                "root_fen": active["root_fen"],
                "local_ply": local_ply,
                "history_uci": list(active["history"]),
                "mover": row["mover"],
                "raw_q_mover": float(row["raw_q_mover"]),
                "clipped_q_mover": q,
                "selected_best_uci": move.uci(),
                "selected_action_played": row["selected_action_played"],
                "protected_search_aliases": list(protected_aliases),
            }
            active["rows"].append(normalized)
            if row["selected_action_played"]:
                board.push(move)
                active["history"].append(move.uci())
                active["board"] = board
            else:
                active["stopped_without_play"] = True
        elif kind == "game_end":
            if active is None or event.get("root_id") != active["root_id"]:
                raise ValueError("game-end root differs from active trajectory")
            status = event.get("status")
            eligible = event.get("training_eligible")
            labels = event.get("row_labels")
            if type(eligible) is not bool or not isinstance(labels, list):
                raise ValueError("final eligibility and row labels required")
            rows = active["rows"]
            board = active["board"]
            final = event.get("final_state")
            expected_final = {
                "fen4": " ".join(board.fen().split()[:4]),
                "alias": alias(board),
                "history_uci": list(active["history"]),
            }
            if not isinstance(final, dict) or any(
                final.get(k) != v for k, v in expected_final.items()
            ):
                raise ValueError("final board/history does not replay")
            if type(final.get("protected")) is not bool:
                raise ValueError("final protected-board witness is required")
            row_protected = any(r["protected_search_aliases"] for r in rows)
            if (status == "excluded-protected-trajectory") != (
                row_protected or final["protected"]
            ):
                raise ValueError("whole-game protected status differs from exposure witnesses")
            if len(rows) != len(labels):
                raise ValueError("terminal label count differs from actual action rows")

            if status == "completed-own-terminal":
                outcome = board.outcome(claim_draw=True)
                if outcome is None or not eligible:
                    raise ValueError("completed game lacks actual terminal result")
                white_result = 0 if outcome.winner is None else (1 if outcome.winner else -1)
                expected_labels = [
                    white_result if r["mover"] == "white" else -white_result
                    for r in rows
                ]
            elif status in {"unknown-ply-cap", "unknown-row-budget-prefix"}:
                if board.outcome(claim_draw=True) is not None or not eligible:
                    raise ValueError("UNKNOWN tail must be live and eligible for own-Q bootstrap")
                if status == "unknown-ply-cap" and len(rows) != 16:
                    raise ValueError("fixed sixteen-ply cap required")
                expected_labels = [None] * len(rows)
            elif status == "excluded-protected-trajectory":
                if eligible:
                    raise ValueError("protected episode must be excluded in full")
                expected_labels = [None] * len(rows)
            else:
                raise ValueError("unsupported or incomplete episode status")
            if labels != expected_labels:
                raise ValueError("producer own-terminal WDL labels do not replay")

            for row, label in zip(rows, labels, strict=True):
                row["episode_status"] = status
                row["own_wdl_mover"] = label
                row["train_eligible"] = eligible
                history_sha = hashlib.sha256(
                    canonical({"root_fen": row["root_fen"], "prefix_uci": row["history_uci"]})
                ).hexdigest()
                expected_trace.append({
                    "row_id": row["root_id"] + ":" + str(row["local_ply"]),
                    "history_sha256": history_sha,
                    "raw_q_mover": row["raw_q_mover"],
                    "target": row["clipped_q_mover"],
                    "own_wdl_mover": label,
                    "outcome_status": "UNKNOWN" if label is None else "known-own-terminal",
                    "episode_status": status,
                    "train_eligible": eligible,
                })
            episodes.append(rows)
            active = None
        else:
            raise ValueError("only complete registered producer events are accepted")

    if active is not None or not episodes:
        raise ValueError("producer log must end at a closed nonempty episode boundary")
    return episodes, expected_trace


def build_tdlambda_dataset(
    *,
    source_dataset_bytes: bytes,
    source_provenance_bytes: bytes,
    collection_receipt_bytes: bytes,
    events_bytes: bytes,
    conversion_result_bytes: bytes,
    build_seal_bytes: bytes,
) -> tuple[bytes, bytes]:
    """Verify frozen Q-v2 conversion and emit versioned lambda targets.

    This is a preparation API only; it does not read files or launch training.
    """
    seal = json.loads(build_seal_bytes)
    if seal.get("schema") != BUILD_SEAL_SCHEMA or seal.get("status") != "registered":
        raise ValueError("ROOT-registered immutable target-build seal required")
    first, deadline, operator_end = (
        seal.get("first"), seal.get("deadline"), seal.get("operator_end_epoch")
    )
    if (seal.get("mode") != "convert"
            or any(type(x) not in (int, float) or not math.isfinite(x)
                   for x in (first, deadline, operator_end))
            or not first < time.time() < deadline <= min(first + 600, operator_end)
            or operator_end > 1791448916.685839):
        raise ValueError("separate ROOT-registered original600 conversion clock required")
    if (seal.get("lambda") != LAMBDA or seal.get("expected_train_rows") != 1024
            or type(seal.get("seed")) is not int):
        raise ValueError("fixed fallback lambda and exact original row budget")
    raw_inputs = {
        "source_dataset": source_dataset_bytes,
        "source_provenance": source_provenance_bytes,
        "collection_receipt": collection_receipt_bytes,
        "events": events_bytes,
        "conversion_result": conversion_result_bytes,
    }
    pins = seal.get("inputs")
    if not isinstance(pins, dict) or set(pins) != set(raw_inputs):
        raise ValueError("complete source conversion/collection input bindings required")
    for name, raw in raw_inputs.items():
        if pins[name] != sha_bytes(raw):
            raise ValueError(f"sealed input SHA differs: {name}")

    dataset = json.loads(source_dataset_bytes)
    provenance = json.loads(source_provenance_bytes)
    receipt = json.loads(collection_receipt_bytes)
    conversion_result = json.loads(conversion_result_bytes)
    if (
        dataset.get("schema") != SOURCE_DATA_SCHEMA
        or dataset.get("phase") != "own-learning"
            or provenance.get("schema") != SOURCE_PROVENANCE_SCHEMA
            or provenance.get("teacher_labels_used") is not False
            or provenance.get("seed") != seal.get("seed")
            or receipt.get("schema") != SOURCE_RECEIPT_SCHEMA
            or receipt.get("status") != "PASS-exact-row-budget"
            or receipt.get("train_rows") != 1024
            or receipt.get("seed") != seal.get("seed")
            or conversion_result.get("status") != SOURCE_CONVERSION_STATUS
            or conversion_result.get("dataset_sha256") != sha_bytes(source_dataset_bytes)
            or conversion_result.get("provenance_sha256") != sha_bytes(source_provenance_bytes)
            or provenance.get("dataset_sha256") != sha_bytes(source_dataset_bytes)
            or provenance.get("collection_receipt_sha256") != sha_bytes(collection_receipt_bytes)
            or provenance.get("parent_candidate", {}).get("sha256")
            != receipt.get("parent_candidate_sha256")
            or provenance.get("inputs") != seal.get("raw_collection_inputs")
            or receipt.get("events_sha256") != sha_bytes(events_bytes)
        or receipt.get("events_bytes") != len(events_bytes)
    ):
        raise ValueError(
            "source must be the exact sealed own-Q v2 conversion, not a legacy dataset"
        )

    episodes, expected_trace = parse_final_events(events_bytes)
    if not time.time() < deadline:
        raise TimeoutError("original conversion clock expired; no outputs published")
    if provenance.get("trace") != expected_trace:
        raise ValueError("legal replayed labels/events differ from frozen v2 converter trace")
    row_ids = receipt.get("training_row_ids")
    if (not isinstance(row_ids, list) or len(row_ids) != 1024 or len(set(row_ids)) != 1024
            or row_ids != [r["row_id"] for r in expected_trace if r["train_eligible"]]):
        raise ValueError("exact eligible Q-v2 source-row order required")
    source_rows = dataset.get("rows")
    if not isinstance(source_rows, list) or len(source_rows) != 1024:
        raise ValueError("exact 1,024-row source dataset required")
    trace_by_id = {r["row_id"]: r for r in expected_trace}
    targets: dict[str, dict] = {}
    for episode in episodes:
        result = transform_episode(episode)
        for row in result["rows"]:
            targets[row["source_row_id"]] = row
    if set(targets) != set(row_ids):
        raise ValueError("TD(lambda) targets do not cover exact eligible 1,024 rows")

    output_rows = []
    for row_id, source_row in zip(row_ids, source_rows, strict=True):
        trace = trace_by_id[row_id]
        if source_row.get("target") != trace["target"]:
            raise ValueError("source dataset own-Q target differs from replayed raw episode")
        target = targets[row_id]
        output_rows.append({
            "source_row_id": row_id,
            "indices": source_row["indices"],
            "prior_logit": source_row["prior_logit"],
            "target": target["target_mover"],
        })
    data_out = canonical({
        "schema": OUTPUT_DATA_SCHEMA,
        "phase": TRAINING_PHASE,
        "target_method": "frozen-parent-own-search-trajectory-tdlambda-v1",
        "lambda": LAMBDA,
        "rows": output_rows,
    }) + b"\n"

    known_rows = sum(x["own_wdl_mover"] is not None and x["train_eligible"] for x in expected_trace)
    unknown_rows = sum(x["own_wdl_mover"] is None and x["train_eligible"] for x in expected_trace)
    excluded_rows = sum(not x["train_eligible"] for x in expected_trace)
    transformed_trace = []
    for episode in episodes:
        transformed = transform_episode(episode)
        transformed_trace.extend(transformed["rows"])
    provenance_out = canonical({
        "schema": OUTPUT_PROVENANCE_SCHEMA,
        "target_method": "frozen-parent-own-search-trajectory-tdlambda-v1",
        "lambda": LAMBDA,
        "source_dataset_sha256": sha_bytes(source_dataset_bytes),
        "source_provenance_sha256": sha_bytes(source_provenance_bytes),
        "source_collection_receipt_sha256": sha_bytes(collection_receipt_bytes),
        "source_events_sha256": sha_bytes(events_bytes),
        "source_conversion_result_sha256": sha_bytes(conversion_result_bytes),
        "build_seal_sha256": sha_bytes(build_seal_bytes),
        "seed": receipt["seed"],
        "source_raw_collection_inputs": provenance.get("inputs"),
        "parent_candidate": provenance.get("parent_candidate"),
        "teacher_labels_used": False,
        "source_train_rows": 1024,
        "output_train_rows": len(output_rows),
        "known_terminal_source_rows": known_rows,
        "unknown_bootstrap_source_rows": unknown_rows,
        "excluded_protected_source_rows": excluded_rows,
        "targets": transformed_trace,
        "target_semantics": (
            "white-perspective lambda=.5 recursion converted back to mover POV; "
            "completed episodes use actual own WDL; UNKNOWN endpoint is last logged pre-action Q"
        ),
        "native_phase_required": TRAINING_PHASE,
        "native_schema_required": "own-kingbucket-nnue16-tdlambda-full-native-cpu-v1",
        "legacy_native_resume_allowed": False,
    }) + b"\n"
    return data_out, provenance_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--source-provenance", required=True)
    parser.add_argument("--collection-receipt", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--conversion-result", required=True)
    parser.add_argument("--build-seal", required=True)
    parser.add_argument("--dataset-output", required=True)
    parser.add_argument("--provenance-output", required=True)
    args = parser.parse_args()
    paths = [Path(getattr(args, name)) for name in (
        "source_dataset", "source_provenance", "collection_receipt", "events",
        "conversion_result", "build_seal",
    )]
    dataset, provenance = build_tdlambda_dataset(
        source_dataset_bytes=paths[0].read_bytes(),
        source_provenance_bytes=paths[1].read_bytes(),
        collection_receipt_bytes=paths[2].read_bytes(),
        events_bytes=paths[3].read_bytes(),
        conversion_result_bytes=paths[4].read_bytes(),
        build_seal_bytes=paths[5].read_bytes(),
    )
    outputs = [Path(args.dataset_output), Path(args.provenance_output)]
    if any(not path.resolve().is_relative_to("/dev/shm") for path in outputs):
        raise ValueError("TD(lambda) dataset/provenance writes are RAM-only")
    outputs[0].parent.mkdir(parents=True, exist_ok=True)
    outputs[1].parent.mkdir(parents=True, exist_ok=True)
    for path, raw in zip(outputs, (dataset, provenance), strict=True):
        with path.open("xb") as stream:
            stream.write(raw)


if __name__ == "__main__":
    main()

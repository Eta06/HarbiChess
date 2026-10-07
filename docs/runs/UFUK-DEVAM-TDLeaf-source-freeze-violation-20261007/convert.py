"""Replay sealed PV-leaf episodes and build sparse TDLeaf training rows.

No search, engine, network forward, or SGD is performed by this module. The
caller supplies the pinned feature and prior helpers after verifying hashes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import struct
import sys
from pathlib import Path

import chess

from tdleaf_targets import make_episode


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned(ref, *, cap=16 * 2**20):
    path = Path(ref["path"]).resolve()
    if not path.is_file() or path.is_symlink() or path.stat().st_size > cap:
        raise ValueError("pinned regular input/cap required")
    if sha(path) != ref["sha256"]:
        raise ValueError("input/source SHA differs")
    return path


def load_module(ref, name):
    path = pinned(ref)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path:
        raise ValueError("pinned module origin differs")
    return module


def alias(board):
    projection = min(board.board_fen(), board.mirror().board_fen()).encode("ascii")
    return int.from_bytes(hashlib.sha256(projection).digest()[:8], "little", signed=True)


def board_at(root_fen, history):
    if root_fen != chess.STARTING_FEN or not isinstance(history, list):
        raise ValueError("full standard-start root history required")
    board = chess.Board(root_fen)
    for token in history:
        move = chess.Move.from_uci(token)
        if move not in board.legal_moves:
            raise ValueError("illegal chronological history")
        board.push(move)
    return board


def unpack_aliases(reference, chunks):
    name = reference.get("file")
    if name not in chunks or Path(name).name != name:
        raise ValueError("sidecar reference must name a sealed chunk")
    raw = chunks[name]
    offset, count = reference.get("offset_bytes"), reference.get("count")
    if type(offset) is not int or type(count) is not int or offset < 0 or count < 0 or offset % 8:
        raise ValueError("invalid int64 alias span")
    end = offset + count * 8
    if end > len(raw):
        raise ValueError("alias span exceeds sealed sidecar")
    values = list(struct.unpack(f"<{count}q", raw[offset:end])) if count else []
    if values != sorted(set(values)):
        raise ValueError("row aliases must be sorted and unique")
    return values


def verify_search_row(row, aliases, protected):
    board = board_at(row.get("root_fen"), row.get("history_uci"))
    if board.outcome(claim_draw=True) is not None:
        raise ValueError("search root is already terminal")
    history_sha = hashlib.sha256(
        canonical({"root_fen": row["root_fen"], "history_uci": row["history_uci"]})
    ).hexdigest()
    move = chess.Move.from_uci(row["selected_action_uci"])
    if (
        row.get("history_sha256") != history_sha
        or row.get("mover") != ("white" if board.turn else "black")
        or row.get("root_alias") != alias(board)
        or move not in board.legal_moves
        or row.get("pv_uci", [None])[0] != move.uci()
    ):
        raise ValueError("root/fullhistory/mover/action binding differs")
    if (
        type(row.get("root_actions")) is not int
        or row["root_actions"] != board.legal_moves.count()
        or type(row.get("nodes")) is not int
        or not row["root_actions"] + 1 <= row["nodes"] <= 8192
        or type(row.get("evaluations")) is not int
        or row["evaluations"] != row.get("actual_eval_calls")
        or not 0 <= row["evaluations"] <= row["nodes"]
        or type(row.get("completed_depth")) is not int
        or not 0 <= row["completed_depth"] <= 8
        or row.get("search_root_value") is None
        or not math.isfinite(row["search_root_value"])
        or abs(row["search_root_value"]) > 2
    ):
        raise ValueError("completed search counters/Q range")
    pv = row["pv_uci"]
    leaf = board.copy(stack=True)
    for token in pv:
        action = chess.Move.from_uci(token)
        if action not in leaf.legal_moves:
            raise ValueError("illegal complete PV continuation")
        leaf.push(action)
    leaf_sha = hashlib.sha256(
        canonical({"root_fen": row["root_fen"], "history_uci": row["history_uci"] + pv})
    ).hexdigest()
    if row.get("leaf_fen") != leaf.fen() or row.get("leaf_history_sha256") != leaf_sha:
        raise ValueError("PV leaf full-history identity differs")
    actual_aliases = aliases
    if row.get("protected_search_aliases") != sorted(set(actual_aliases) & protected):
        raise ValueError("protected evaluator-input trace differs")
    if (
        type(row.get("selected_action_played")) is not bool
        or type(row.get("played_action")) is not bool
        or row.get("selected_action_played") is not row.get("played_action")
    ):
        raise ValueError("played-action flags disagree")
    if row["played_action"]:
        if row.get("played_action_uci") != move.uci():
            raise ValueError("only actually played selected actions enter game trajectory")
    elif row.get("played_action_uci") is not None:
        raise ValueError("unplayed action cannot be a game transition")
    outcome = leaf.outcome(claim_draw=True)
    if outcome is None:
        if row.get("leaf_kind") != "static" or row.get("leaf_terminal_mover_wdl") is not None:
            raise ValueError("nonterminal endpoint must be a static leaf")
        value = row.get("leaf_value_mover")
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or not -1 <= value <= 1
        ):
            raise ValueError("static leaf value range")
        sign = -1.0 if len(pv) % 2 else 1.0
        if not math.isclose(row["search_root_value"], sign * value, abs_tol=1e-12, rel_tol=0.0):
            raise ValueError("root score is not the signed selected PV leaf value")
        if alias(leaf) not in actual_aliases:
            raise ValueError("static PV endpoint must appear in the actual evaluator-input trace")
    else:
        expected = 0 if outcome.winner is None else (1 if outcome.winner == leaf.turn else -1)
        if (
            row.get("leaf_kind") != "terminal"
            or row.get("leaf_terminal_mover_wdl") != expected
            or row.get("leaf_value_mover") is not None
        ):
            raise ValueError("rule-terminal PV endpoint/WDL differs")
        terminal_score = row.get("leaf_search_value_mover")
        if (
            isinstance(terminal_score, bool)
            or not isinstance(terminal_score, int | float)
            or not math.isfinite(terminal_score)
        ):
            raise ValueError("terminal search score remains auditable but is never trained")
        sign = -1.0 if len(pv) % 2 else 1.0
        if not math.isclose(
            row["search_root_value"], sign * terminal_score, abs_tol=1e-12, rel_tol=0.0
        ):
            raise ValueError("terminal root score/PV parity differs")
    if any(type(value) is not int for value in actual_aliases):
        raise ValueError("trace alias type")
    return board, leaf


def convert_events(*, events, sidecars, protected_aliases, receipt, features, prior):
    """Convert already-sealed event objects to exact sparse static-leaf targets."""
    if (
        receipt.get("schema") != "human-prior-tdleaf-collection-receipt-v1"
        or receipt.get("status") != "PASS-exact-row-budget"
    ):
        raise ValueError("closed versioned PV-leaf receipt required")
    if receipt.get("teacher_labels_used") is not False or receipt.get("train_rows") != 1024:
        raise ValueError("fixed own-only 1024 target source required")
    episodes = []
    current = None
    all_rows = []
    root_order = []
    exposure = set()
    for event in events:
        kind = event.get("type")
        if kind == "game_start":
            if current is not None:
                raise ValueError("nested/open episode in event stream")
            start_board = board_at(event["root_fen"], event["root_prefix_uci"])
            start_alias = alias(start_board)
            if event.get("root_alias") != start_alias:
                raise ValueError("game-start root alias differs from legal history")
            prefix_board = chess.Board(event["root_fen"])
            prefix_aliases = [alias(prefix_board)]
            for token in event["root_prefix_uci"]:
                prefix_board.push_uci(token)
                prefix_aliases.append(alias(prefix_board))
            if event.get("root_prefix_aliases") != prefix_aliases:
                raise ValueError("complete preselected root prefix alias trace differs")
            exposure.update(prefix_aliases)
            current = {
                "root_id": event["root_id"],
                "rows": [],
                "protected_path": bool(set(prefix_aliases) & protected_aliases),
            }
            root_order.append(event["root_id"])
        elif kind == "search_row":
            if current is None or event["row"].get("root_id") != current["root_id"]:
                raise ValueError("search row outside its game")
            row = dict(event["row"])
            trace = unpack_aliases(row.pop("search_alias_ref"), sidecars)
            root_board, _leaf_board = verify_search_row(row, trace, protected_aliases)
            exposure.add(alias(root_board))
            exposure.update(trace)
            current["protected_path"] = current["protected_path"] or (
                alias(root_board) in protected_aliases or bool(set(trace) & protected_aliases)
            )
            row["_aliases"] = trace
            current["rows"].append(row)
            all_rows.append(row)
        elif kind == "game_end":
            if current is None or event.get("root_id") != current["root_id"]:
                raise ValueError("game end lacks matching start")
            rows = current["rows"]
            status = event.get("status")
            final = event["final_state"]
            final_board = board_at(chess.STARTING_FEN, final["history_uci"])
            final_alias = alias(final_board)
            exposure.add(final_alias)
            if (
                final.get("fen4") != " ".join(final_board.fen().split()[:4])
                or final.get("alias") != final_alias
            ):
                raise ValueError("final board projection differs from replay")
            if final.get("protected") is not (final_alias in protected_aliases):
                raise ValueError("final board protection flag differs from fixed set")
            current["protected_path"] = (
                current["protected_path"] or final_alias in protected_aliases
            )
            if event.get("training_eligible") is not (not current["protected_path"]):
                raise ValueError("whole-episode protected-path exclusion differs")
            if current["protected_path"] != (status == "excluded-protected-trajectory"):
                raise ValueError("protected trajectory status differs")
            for row in rows:
                row["episode_status"] = status
                row["train_eligible"] = event.get("training_eligible") is True
            terminal_wdl = None
            if status == "completed-own-terminal":
                outcome = final_board.outcome(claim_draw=True)
                if outcome is None:
                    raise ValueError("completed terminal not independently rule-verifiable")
                terminal_wdl = 0 if outcome.winner is None else (1 if outcome.winner else -1)
                expected_labels = [
                    terminal_wdl if row["mover"] == "white" else -terminal_wdl for row in rows
                ]
            else:
                expected_labels = [None for _ in rows]
            if event.get("row_labels") != expected_labels:
                raise ValueError("event row-label chronology differs from replayed outcome")
            episode = make_episode(rows, terminal_white_wdl=terminal_wdl)
            episodes.append(episode)
            current = None
        else:
            raise ValueError("unknown or incomplete event type")
    if (
        current is not None
        or len(root_order) != receipt.get("starts_considered")
        or len(all_rows) != receipt.get("all_actor_rows")
    ):
        raise ValueError("open episode or start count mismatch")
    if len(exposure) != receipt.get("unique_exposure_aliases"):
        raise ValueError("all-path plus every-evaluator-input exposure count differs")
    training_ids = set(receipt.get("training_row_ids", []))
    if len(training_ids) != 1024 or len(training_ids) != len(receipt.get("training_row_ids", [])):
        raise ValueError("exact unique chronological training ids required")
    observed_training_ids = [
        row["root_id"] + ":" + str(row["local_ply"])
        for row in all_rows
        if row.get("train_eligible") is True
    ]
    if observed_training_ids != receipt["training_row_ids"]:
        raise ValueError("receipt training IDs differ from all replayed eligible actor rows")
    targets = {}
    for episode in episodes:
        if episode["status"].startswith("excluded"):
            continue
        for row in episode["rows"]:
            if row["gradient_mask"]:
                targets[row["source_row_id"]] = row["tdleaf_target_leaf_mover"]
    output_rows = []
    for row in all_rows:
        row_id = row["root_id"] + ":" + str(row["local_ply"])
        if row_id not in training_ids or row_id not in targets:
            continue
        leaf = board_at(row["root_fen"], row["history_uci"])
        for token in row["pv_uci"]:
            leaf.push_uci(token)
        target = targets[row_id]
        indices = features.board_indices(leaf)
        values = prior.features(leaf)
        prior_logit = sum(w * x for w, x in zip(prior.PRIOR, values, strict=True)) / prior.SCALE
        if not math.isfinite(prior_logit) or not -1 <= target <= 1:
            raise ValueError("finite static-leaf target/prior")
        output_rows.append(
            {
                "source_row_id": row_id,
                "indices": indices,
                "prior_logit": prior_logit,
                "target": float(target),
                "leaf_history_sha256": row["leaf_history_sha256"],
            }
        )
    if len(output_rows) > len(training_ids) or not output_rows:
        raise ValueError("empty or oversized trainable static-leaf target set")
    return {
        "schema": "own-tdleaf-pv-leaf-sparse-training-data-v1",
        "phase": "human-prior-tdleaf-own-learning-v1",
        "seed": receipt["seed"],
        "lambda": 0.5,
        "target_source": (
            "own-parent-search-PV-leaf-sequence-TD(lambda); own-rule-WDL endpoints only"
        ),
        "eligible_actor_rows": len(training_ids),
        "gradient_rows": len(output_rows),
        "teacher_labels_used": False,
        "rows": output_rows,
    }


def convert_sealed(spec):
    """Verify all files in a registered conversion seal, then replay its ledger."""
    if spec.get("schema") != "human-prior-tdleaf-dataset-conversion-seal-v1":
        raise ValueError("versioned TDLeaf conversion seal required")
    registration_path = pinned(spec["registration"])
    receipt_path = pinned(spec["receipt"])
    events_path = pinned(spec["events"])
    reg = json.loads(registration_path.read_bytes())
    receipt = json.loads(receipt_path.read_bytes())
    if (
        reg.get("schema") != "human-prior-tdleaf-collection-registration-v1"
        or reg.get("status") != "registered"
        or receipt.get("registration_sha256") != spec["registration"]["sha256"]
        or receipt.get("events_sha256") != spec["events"]["sha256"]
        or receipt.get("events_bytes") != events_path.stat().st_size
        or receipt.get("pv_search_helper_sha256") != reg["pv_search_helper"]["sha256"]
        or receipt.get("search_helper_sha256") != reg["search_helper"]["sha256"]
        or receipt.get("producer_source_sha256") != reg["producer_source_sha256"]
        or receipt.get("parent_candidate_sha256") != reg["parent_candidate"]["sha256"]
        or receipt.get("root_pool_sha256") != reg["root_pool"]["sha256"]
        or receipt.get("protected_aliases_sha256") != reg["protected_aliases"]["sha256"]
        or receipt.get("parent_admission_seal") != reg["parent_admission_seal"]
        or receipt.get("parent_admission_result") != reg["parent_admission_result"]
        or receipt.get("search") != reg["search"]
        or receipt.get("generation_helper_sha256") != reg["generation_helper_sha256"]
        or receipt.get("parent_helpers") != reg["parent_helpers"]
        or receipt.get("pv_leaf_targets") is not True
        or receipt.get("lambda") != 0.5
        or not reg["original_first_epoch"]
        <= receipt.get("finished_epoch", 0)
        <= reg["original_deadline_epoch"]
        or receipt.get("operator_end_epoch") != reg["operator_end_epoch"]
        or events_path.parent != Path(reg["output_path"]).resolve()
        or receipt_path != Path(reg["output_path"]).resolve() / "receipt.json"
        or events_path.name != "events.jsonl"
        or not events_path.read_bytes().endswith(b"\n")
        or spec.get("protected_aliases") != reg["protected_aliases"]
    ):
        raise ValueError("registration/receipt/source/event bindings differ")
    producer_dir = Path(spec["producer_directory"]).resolve()
    for name, digest in reg["producer_source_sha256"].items():
        if Path(name).name != name or sha(producer_dir / name) != digest:
            raise ValueError("frozen collection producer source closure differs")
    for source_path, digest in reg["generation_helper_sha256"].items():
        if sha(source_path) != digest:
            raise ValueError("generation helper source changed")
    if pinned(reg["root_pool"]).stat().st_size > 8 * 2**20:
        raise ValueError("fixed root-pool size cap")
    if sha(reg["parent_candidate"]["path"]) != reg["parent_candidate"]["sha256"]:
        raise ValueError("admitted parent candidate changed")
    if len(spec["alias_chunks"]) != len(receipt["alias_chunks"]):
        raise ValueError("complete alias sidecar list required")
    sidecars = {}
    expected = {row["file"]: row for row in receipt["alias_chunks"]}
    for ref in spec["alias_chunks"]:
        path = pinned(ref, cap=8 * 2**20)
        if (
            path.parent != events_path.parent
            or path.name not in expected
            or sha(path) != expected[path.name]["sha256"]
        ):
            raise ValueError("sidecar not in collection receipt")
        if path.stat().st_size != expected[path.name]["bytes"]:
            raise ValueError("sidecar size differs")
        sidecars[path.name] = path.read_bytes()
    protected_ref = pinned(spec["protected_aliases"], cap=256 * 2**20)
    raw = protected_ref.read_bytes()
    if len(raw) % 8:
        raise ValueError("protected aliases must be int64")
    protected = list(struct.unpack(f"<{len(raw) // 8}q", raw))
    if protected != sorted(set(protected)):
        raise ValueError("canonical sorted unique protected aliases")
    if events_path.stat().st_size > 16 * 2**20:
        raise ValueError("event stream16MiB cap")
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    parent_helpers = reg["parent_helpers"]
    feature_ref = {
        "path": str(Path(parent_helpers["directory"]) / "model.py"),
        "sha256": parent_helpers["model_sha256"],
    }
    prior_ref = {"path": parent_helpers["prior_path"], "sha256": parent_helpers["prior_sha256"]}
    if spec.get("features") != feature_ref or spec.get("prior") != prior_ref:
        raise ValueError("current admitted parent feature/prior input identity")
    features = load_module(feature_ref, "tdleaf_conversion_features")
    prior = load_module(prior_ref, "tdleaf_conversion_prior")
    dataset = convert_events(
        events=events,
        sidecars=sidecars,
        protected_aliases=set(protected),
        receipt=receipt,
        features=features,
        prior=prior,
    )
    return dataset, {
        "schema": "human-prior-tdleaf-target-provenance-v1",
        "status": "PASS-replayed-closed-PV-leaf-targets",
        "seed": reg["seed"],
        "lambda": 0.5,
        "teacher_labels_used": False,
        "receipt_sha256": spec["receipt"]["sha256"],
        "registration_sha256": spec["registration"]["sha256"],
        "events_sha256": spec["events"]["sha256"],
        "dataset_rows": len(dataset["rows"]),
        "eligible_actor_rows": dataset["eligible_actor_rows"],
        "gradient_rows": dataset["gradient_rows"],
        "converter_path": str(Path(__file__).resolve()),
        "converter_sha256": sha(__file__),
        "inputs": spec,
    }

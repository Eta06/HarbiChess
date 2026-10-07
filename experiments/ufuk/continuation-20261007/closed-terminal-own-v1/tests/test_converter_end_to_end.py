from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
import time
import uuid
from pathlib import Path

import chess
import pytest

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "source"))
import contract_builder  # noqa: E402
import convert  # noqa: E402


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {"path": str(path), "sha256": sha(path)}


def write_json(path, value):
    return write(path, canonical(value) + b"\n")


_UNKNOWN_MOVES = None


def make_fixture(tmp_path):
    helper = tmp_path / "helpers"
    model = write(
        helper / "model.py",
        b"def board_indices(board):\n    return [0, 1, 2]\n",
    )
    prior_path = helper / "prior.py"
    prior = write(
        prior_path,
        b"PRIOR=(1.0, 0.0, 0.0)\nSCALE=1.0\ndef features(board):\n    return (1.0, 0.0, 0.0)\n",
    )
    parent = write(tmp_path / "parent.pt", b"synthetic-parent-marker")
    search = write(tmp_path / "search.py", b"# sealed no-op fixture\n")
    protected_board = chess.Board()
    protected_board.push_uci("h2h4")
    protected_values = [convert.alias(protected_board)]
    protected = write(
        tmp_path / "protected.bin",
        struct.pack(f"<{len(protected_values)}q", *protected_values),
    )
    selection_rows, pool_rows = [], []
    for i in range(4096):
        row_id = f"train-row-{i:04d}"
        selection_rows.append(
            {
                "row_id": row_id,
                "trajectory_id": f"trajectory-{i}",
                "root_fen": chess.STARTING_FEN,
                "prefix_uci": [],
            }
        )
        pool_rows.append(
            {
                "root_id": row_id,
                "source_row_id": row_id,
                "trajectory_id": f"trajectory-{i}",
                "root_fen": chess.STARTING_FEN,
                "prefix_uci": [],
                "role": "TRAIN",
            }
        )
    selection = write_json(tmp_path / "selection.json", {"rows": selection_rows})
    teacher_labels_ref = write_json(tmp_path / "teacher-labels.json", {"schema": "pinned-only"})
    pool_obj = {
        "schema": "teacher-selected-ownq-train-roots-v2",
        "selection_status": "pass",
        "train_only": True,
        "source_selection_sha256": selection["sha256"],
        "source_teacher_labels_sha256": teacher_labels_ref["sha256"],
        "selection_path": selection["path"],
        "selection_sha256": selection["sha256"],
        "teacher_labels_path": teacher_labels_ref["path"],
        "teacher_labels_sha256": teacher_labels_ref["sha256"],
        "rows": pool_rows,
    }
    pool_path = write_json(tmp_path / "pool.json", pool_obj)
    pool_ref = {
        **pool_path,
        "selection_path": selection["path"],
        "selection_sha256": selection["sha256"],
        "teacher_labels_path": teacher_labels_ref["path"],
        "teacher_labels_sha256": teacher_labels_ref["sha256"],
    }
    # The converter loads this pinned selection helper. The production collector
    # unit tests separately cover its duplicate-history rejection.
    collector_path = tmp_path / "synthetic_collector.py"
    collector_src = """
import hashlib
import chess
def replay(root):
    board = chess.Board(root["root_fen"])
    for token in root["prefix_uci"]:
        board.push_uci(token)
    if board.outcome(claim_draw=True):
        raise ValueError("terminal root")
    return board
def starts(pool, seed, protected, n=128, pool_size=4096):
    rows = pool["rows"]
    chosen = sorted(rows, key=lambda r: hashlib.sha256(
        f"own-nnue-ownq-start-v1|{seed}|{r['root_id']}".encode()).digest())[:n]
    return chosen, hashlib.sha256("\\n".join(r["root_id"] for r in chosen).encode()).hexdigest()
"""
    write(collector_path, collector_src.encode())
    producer_dir = tmp_path / "producer"
    producer_dir.mkdir()
    producer_collector = write(producer_dir / "collector.py", collector_src.encode())
    producer_runner = write(producer_dir / "run_collection.py", b"# fixture runner pin\n")
    producer_sources = {
        "collector.py": producer_collector["sha256"],
        "run_collection.py": producer_runner["sha256"],
    }
    seed = 20262905
    spec_reg = {
        "schema": "own-nnue-closed-terminal-collection-registration-v1",
        "status": "registered",
        "seed": seed,
        "row_limit": 1024,
        "root_limit": 128,
        "plies_per_root": 400,
        "actor_row_limit": 4096,
        "search": {"nodes": 8192, "qdepth": 2, "max_depth": 8},
        "original_first_epoch": 100.0,
        "original_deadline_epoch": 150.0,
        "operator_end_epoch": 200.0,
        "parent_candidate": {**parent, "contract_sha256": "c" * 64},
        "parent_helpers": {
            "directory": str(helper),
            "model_sha256": model["sha256"],
            "prior_path": prior_path.as_posix(),
            "prior_sha256": prior["sha256"],
        },
        "root_pool": pool_ref,
        "protected_aliases": protected,
        "search_helper": search,
        "producer_source_sha256": producer_sources,
    }
    reg_path = write_json(tmp_path / "registration.json", spec_reg)
    reg_ref = reg_path
    pool = json.loads(Path(pool_path["path"]).read_bytes())
    collector_spec = importlib.util.spec_from_file_location("fixture_order", collector_path)
    collector_mod = importlib.util.module_from_spec(collector_spec)
    collector_spec.loader.exec_module(collector_mod)
    chosen, order_sha = collector_mod.starts(pool, seed, set(protected_values))

    short_mate = ["c2c3", "c7c6", "h2h3", "h7h6", "f2f3", "e7e5", "g2g4", "d8h4"]
    long_mate = [
        "c2c4",
        "c7c5",
        "b2b3",
        "b7b6",
        "b1c3",
        "g7g6",
        "c3a4",
        "h7h6",
        "a4c5",
        "d7d6",
        "c5d3",
        "b8d7",
        "d3b4",
        "a8b8",
        "b4c6",
        "e7e6",
        "c6a5",
        "g8f6",
        "a5c6",
        "f6h5",
        "f2f3",
        "e6e5",
        "g2g4",
        "d8h4",
    ]

    def unknown_prefix(seed_value):
        global _UNKNOWN_MOVES
        if _UNKNOWN_MOVES is not None:
            return _UNKNOWN_MOVES
        import random

        for attempt in range(100):
            rng = random.Random(seed_value + attempt)
            board = chess.Board()
            moves = []
            for _ in range(400):
                safe = []
                for move in board.legal_moves:
                    child = board.copy()
                    child.push(move)
                    if (
                        child.outcome(claim_draw=True) is None
                        and not child.is_check()
                        and convert.alias(child) not in protected_values
                    ):
                        safe.append(move)
                if not safe:
                    break
                move = rng.choice(safe)
                board.push(move)
                moves.append(move.uci())
            if len(moves) == 400 and board.outcome(claim_draw=True) is None:
                _UNKNOWN_MOVES = moves
                return _UNKNOWN_MOVES
        raise AssertionError("synthetic legal UNKNOWN cap fixture generation failed")

    unknown_moves = unknown_prefix(811)
    events = []
    exposed = set()
    training_ids, games = [], []
    sidecar_values = []
    for ordinal, root in enumerate(chosen):
        board = collector_mod.replay(root)
        prefix = []
        if ordinal == 0:
            action_line = long_mate
            game_status = "completed-own-terminal"
        elif ordinal == 1:
            action_line = unknown_moves
            game_status = "unknown-ply-cap"
        elif ordinal == 2:
            action_line = ["e2e4"]
            game_status = "excluded-protected-trajectory"
        else:
            action_line = short_mate
            game_status = "completed-own-terminal"
        events.append(
            {
                "type": "game_start",
                "root_id": root["root_id"],
                "root_ordinal": ordinal,
                "root_fen": root["root_fen"],
                "root_prefix_uci": [],
                "root_alias": convert.alias(board),
            }
        )
        exposed.add(convert.alias(board))
        labels = []
        eligible = game_status == "completed-own-terminal"
        did_play = game_status != "excluded-protected-trajectory"
        for ply, token in enumerate(action_line):
            root_alias = convert.alias(board)
            exposed.add(root_alias)
            move = chess.Move.from_uci(token)
            assert move in board.legal_moves
            legal = board.legal_moves.count()
            eval_aliases = [root_alias]
            protected_hits = []
            if not did_play:
                branch = board.copy()
                branch.push_uci("h2h4")
                eval_aliases.append(convert.alias(branch))
                protected_hits = [convert.alias(branch)]
            eval_aliases = sorted(set(eval_aliases))
            offset = len(sidecar_values) * 8
            sidecar_values.extend(eval_aliases)
            exposed.update(eval_aliases)
            mover = "white" if board.turn else "black"
            labels.append((-1 if mover == "white" else 1) if eligible else None)
            row_id = f"{root['root_id']}:{ply}"
            if eligible:
                training_ids.append(row_id)
            evaluations = len(eval_aliases)
            events.append(
                {
                    "type": "search_row",
                    "row": {
                        "root_id": root["root_id"],
                        "root_ordinal": ordinal,
                        "root_fen": root["root_fen"],
                        "root_prefix_uci": [],
                        "local_ply": ply,
                        "history_uci": list(prefix),
                        "fen4": " ".join(board.fen().split()[:4]),
                        "mover": mover,
                        "root_alias": root_alias,
                        "raw_q_mover": 0.25,
                        "clipped_q_mover": 0.25,
                        "mate_range_score_returned": False,
                        "selected_best_uci": token,
                        "nodes": legal + 1,
                        "evaluations": evaluations,
                        "completed_depth": 1,
                        "root_actions": legal,
                        "actual_eval_calls": evaluations,
                        "search_alias_ref": {
                            "file": "search-aliases-0000.bin",
                            "offset_bytes": offset,
                            "count": evaluations,
                        },
                        "protected_search_aliases": protected_hits,
                        "selected_action_played": did_play,
                        "behavior_policy_available": False,
                        "label_source": "own-frozen-parent-search-diagnostic-only",
                    },
                }
            )
            if did_play:
                board.push(move)
                prefix.append(token)
            else:
                exposed.add(convert.alias(board))
                break
        if eligible:
            assert board.is_checkmate()
            exposed.add(convert.alias(board))
        elif game_status == "unknown-ply-cap":
            assert board.outcome(claim_draw=True) is None
            exposed.add(convert.alias(board))
        exposed.add(convert.alias(board))
        events.append(
            {
                "type": "game_end",
                "root_id": root["root_id"],
                "status": game_status,
                "training_eligible": eligible,
                "row_labels": labels,
                "final_state": {
                    "fen4": " ".join(board.fen().split()[:4]),
                    "alias": convert.alias(board),
                    "history_uci": prefix,
                    "protected": convert.alias(board) in protected_values,
                },
            }
        )
        games.append({"root_id": root["root_id"], "status": game_status, "rows": len(action_line)})
    event_bytes = b"".join(canonical(event) + b"\n" for event in events)
    event_ref = write(tmp_path / "events.jsonl", event_bytes)
    aliases_raw = struct.pack(f"<{len(sidecar_values)}q", *sidecar_values)
    alias_ref = write(tmp_path / "search-aliases-0000.bin", aliases_raw)
    chunks = [
        {
            "file": "search-aliases-0000.bin",
            "bytes": len(aliases_raw),
            "sha256": alias_ref["sha256"],
        }
    ]
    receipt = {
        "schema": "own-nnue-closed-terminal-collection-receipt-v1",
        "status": "PASS-exact-closed-terminal-row-budget",
        "seed": seed,
        "registration_sha256": reg_path["sha256"],
        "parent_candidate_sha256": parent["sha256"],
        "teacher_labels_used": False,
        "search": spec_reg["search"],
        "root_pool_sha256": pool_path["sha256"],
        "protected_aliases_sha256": protected["sha256"],
        "original_first_epoch": 100.0,
        "original_deadline_epoch": 150.0,
        "operator_end_epoch": 200.0,
        "finished_epoch": 110.0,
        "events_bytes": len(event_bytes),
        "events_sha256": event_ref["sha256"],
        "parent_helpers": spec_reg["parent_helpers"],
        "producer_source_sha256": producer_sources,
        "teacher_selection_path": selection["path"],
        "teacher_selection_sha256": selection["sha256"],
        "teacher_labels_path": teacher_labels_ref["path"],
        "teacher_labels_sha256": teacher_labels_ref["sha256"],
        "selected_root_order_sha256": order_sha,
        "train_rows": 1024,
        "all_actor_rows": sum(game["rows"] for game in games),
        "starts_considered": 128,
        "training_row_ids": training_ids,
        "periodic_independent_search_rows": [
            training_ids[i] for i in (0, 204, 409, 614, 819, 1023)
        ],
        "games": games,
        "exposed_board_alias_count": len(exposed),
        "alias_chunks": chunks,
    }
    receipt_ref = write_json(tmp_path / "receipt.json", receipt)
    spec = {
        "schema": "NNUE-closedterminal1024-conversion-seal-v1",
        "registration": reg_ref,
        "receipt": receipt_ref,
        "producer_directory": str(producer_dir),
        "features": model,
        "prior": prior,
        "events": event_ref,
        "alias_chunks": {"search-aliases-0000.bin": alias_ref},
    }
    return spec, pool_path


def test_converter_full_legal_terminal_sidecar_contract(tmp_path, monkeypatch):
    spec, _ = make_fixture(tmp_path)
    data_raw, provenance_raw = convert.convert(spec)
    data, provenance = json.loads(data_raw), json.loads(provenance_raw)
    assert data["schema"] == "own-kingbucket-closed-terminal-training-data-v1"
    assert len(data["rows"]) == 1024
    assert set(row["target"] for row in data["rows"]) == {-1.0, 1.0}
    assert provenance["closed_terminal_rows"] == 1024
    assert provenance["unknown_rows_excluded"] is True
    assert sum(row["selected_for_training"] for row in provenance["trace"]) == 1024
    unknown = [row for row in provenance["trace"] if row["episode_status"] == "unknown-ply-cap"]
    protected = [
        row
        for row in provenance["trace"]
        if row["episode_status"] == "excluded-protected-trajectory"
    ]
    assert len(unknown) == 400 and all(row["target"] is None for row in unknown)
    assert len(protected) == 1 and protected[0]["target"] is None
    dataset_ref = write(tmp_path / "dataset.json", data_raw)
    provenance_ref = write(tmp_path / "provenance.json", provenance_raw)
    spec["dataset"] = dataset_ref
    spec["provenance"] = provenance_ref
    parent = json.loads(Path(spec["registration"]["path"]).read_bytes())["parent_candidate"]
    parent_contract = write_json(
        tmp_path / "teacher-contract.json",
        {
            "phase": "teacher-bootstrap",
            "updates": 256,
            "seed": 20262905,
            "feature_schema": "mover-oriented-king2x2-relative-piece12-square64-v1",
            "prior_helper_sha256": spec["prior"]["sha256"],
        },
    )
    source_inputs = {
        "source_dataset": provenance["dataset_sha256"],
        "source_provenance": provenance_ref["sha256"],
        "collection_receipt": provenance["collection_receipt_sha256"],
        "events": provenance["inputs"]["events"]["sha256"],
    }
    seal = {
        "schema": contract_builder.SEAL_SCHEMA,
        "status": "registered",
        "mode": "proof",
        "first": 100.0,
        "deadline": 150.0,
        "operator_end_epoch": 200.0,
        "seed": 20262905,
        "dataset": dataset_ref,
        "provenance": provenance_ref,
        "source_inputs": source_inputs,
        "parent_candidate": parent,
        "parent_contract": parent_contract,
        "feature_schema": "mover-oriented-king2x2-relative-piece12-square64-v1",
        "raw_collection_inputs": spec,
        "prior_helper": spec["prior"],
        "inference_source_sha256": {spec["features"]["path"]: spec["features"]["sha256"]},
        "core_source_repo": "/workspace/HarbiChess",
        "core_source_commit": "synthetic-core-pin",
    }
    training_contract = contract_builder.make_contract(seal)
    assert training_contract["schema"] == "own-kingbucket-closed-terminal-training-contract-v1"
    assert training_contract["legacy_native_resume_allowed"] is False
    current = time.time()
    cli_seal = {
        **spec,
        "status": "registered",
        "first": current - 1,
        "deadline": current + 120,
        "operator_end_epoch": 1791448916.685839,
        "core_repo": "/workspace/HarbiChess",
    }
    seal_path = tmp_path / "converter-cli-seal.json"
    seal_path.write_bytes(canonical(cli_seal) + b"\n")
    output = Path("/dev/shm") / f"closed-terminal-converter-test-{uuid.uuid4().hex}"
    monkeypatch.setattr(
        sys,
        "argv",
        ["convert.py", "--seal", str(seal_path), "--output", str(output)],
    )
    convert.main()
    result = json.loads((output / "result.json").read_bytes())
    assert result["status"] == "PASS-own1024-closed-terminal-fullhistory-conversion-not-strength"
    assert sha(output / "dataset.json") == result["dataset_sha256"]
    import shutil

    shutil.rmtree(output)


@pytest.mark.parametrize(
    "corruption",
    [
        "mover",
        "history",
        "terminal-winner",
        "protected-final",
        "alias-gap",
        "duplicate-id",
        "root-order",
    ],
)
def test_converter_rejects_resealed_semantic_corruption(tmp_path, corruption):
    spec, _ = make_fixture(tmp_path)
    events_path = Path(spec["events"]["path"])
    events = [json.loads(line) for line in events_path.read_bytes().splitlines()]
    if corruption == "mover":
        next(x for x in events if x["type"] == "search_row")["row"]["mover"] = "black"
    elif corruption == "history":
        next(x for x in events if x["type"] == "search_row")["row"]["history_uci"] = ["e2e4"]
    elif corruption == "terminal-winner":
        next(x for x in events if x["type"] == "game_end")["row_labels"][0] = 1
    elif corruption == "protected-final":
        next(x for x in events if x["type"] == "game_end")["final_state"]["protected"] = True
    elif corruption == "alias-gap":
        next(x for x in events if x["type"] == "search_row")["row"]["search_alias_ref"][
            "offset_bytes"
        ] += 8
    elif corruption == "duplicate-id":
        starts = [x for x in events if x["type"] == "game_start"]
        starts[1]["root_id"] = starts[0]["root_id"]
    elif corruption == "root-order":
        starts = [x for x in events if x["type"] == "game_start"]
        starts[0]["root_id"] = starts[1]["root_id"]
    new_events = b"".join(canonical(event) + b"\n" for event in events)
    events_path.write_bytes(new_events)
    # Reseal downstream hashes: semantic replay, not checksum mismatch, must
    # reject each forged history/target/protection/sidecar/root-order field.
    spec["events"]["sha256"] = sha(events_path)
    receipt_path = Path(spec["receipt"]["path"])
    receipt = json.loads(receipt_path.read_bytes())
    receipt["events_sha256"] = sha(events_path)
    receipt["events_bytes"] = len(new_events)
    receipt_path.write_bytes(canonical(receipt) + b"\n")
    spec["receipt"]["sha256"] = sha(receipt_path)
    with pytest.raises(ValueError):
        convert.convert(spec)

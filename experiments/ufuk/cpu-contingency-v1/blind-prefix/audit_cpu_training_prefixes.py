"""Blind full-history prefix overlap audit for frozen family8 books.

Read-only: no model, engine, optimizer, self-play, or strength queries. A zero
overlap on a partial registry is explicitly not a complete-registry pass.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import time
from pathlib import Path

import chess

BASE = Path(__file__).resolve().parent
BOOK_SPECS = (
    (BASE / "20260709/opening-splits.json", "6d860c4c9a08914a50c6dfe59623f32755cd5185ccc68e3a52645f41efbd6252"),
    (BASE / "20260710/opening-splits.json", "a2885098be18e8a9e45a5ac003f89a504e50877074c80fe809f01f23443aa9ff"),
)
TINY_WHOLE = Path("/workspace/work/harbichess/cpu-contingency-actual-cli-0343/whole")
TINY_SPLIT = Path("/workspace/work/harbichess/cpu-contingency-actual-cli-0343/split")
E1_DEV_RUN = Path("/workspace/work/harbichess/cpu-contingency-E1-actual-0350/run")
SOURCE = "3be5b87db27a0fbde83464e7ea7157f0d9a76ae4"
HARD_END = 1791180000
COMMON_INPUTS = {
    "book": "1a5ca17664a828d669e58cf2bd5d9eeb7f980b83f20d3a4031d36e4ff0930ccb",
    "initial_weights": "e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03",
}


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pos_key(board: chess.Board) -> str:
    # Matches the opening consumer's position-key definition.
    return " ".join(board.fen().split()[:4])


def state_sig(state: dict) -> tuple[str, tuple[str, ...]]:
    return state["root_fen"], tuple(state["moves"])


def replay(state: dict, guard) -> chess.Board:
    board = chess.Board(state["root_fen"])
    if not board.is_valid():
        raise ValueError("invalid root FEN")
    for uci in state["moves"]:
        guard()
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"illegal full-history move: {uci}")
        board.push(move)
    return board


def full_coverage_status(missing: list[str], overlaps: list[dict]) -> str:
    if overlaps:
        return "fail-training-prefix-overlap"
    if any(item.startswith("cpu-formal-") for item in missing):
        return "incomplete-formal-registry"
    if missing:
        return "incomplete-current-development-coverage"
    return "pass-zero-overlap-all-declared-journals"


def current_subset_status(missing: list[str], overlaps: list[dict]) -> str:
    current_missing = [item for item in missing if item.startswith(("cpu-tiny-", "cpu-development-"))]
    current_overlaps = [row for row in overlaps if not row["run"].startswith("cpu-formal-")]
    if current_overlaps:
        return "fail-training-prefix-overlap"
    if current_missing:
        return "incomplete-current-development-coverage"
    return "pass-zero-overlap-current-tiny-and-E1"


def book_roots() -> tuple[set[str], list[dict]]:
    roots: set[str] = set()
    receipts = []
    for path, expected in BOOK_SPECS:
        raw = path.read_bytes()
        digest = sha_bytes(raw)
        if digest != expected:
            raise ValueError(f"frozen book hash differs: {path}")
        book = json.loads(raw)
        if book.get("schema") != 1 or sorted(book.get("splits", {})) != ["arena"]:
            raise ValueError(f"unexpected frozen book schema: {path}")
        count = 0
        for entry in book["splits"]["arena"]:
            key = " ".join(entry["opening"]["fen"].split()[:4])
            roots.add(key)
            count += 1
        receipts.append({"path": str(path), "sha256": digest, "roots": count})
    if len(roots) != 96:
        raise ValueError(f"expected exactly 96 candidate root keys, got {len(roots)}")
    return roots, receipts


def declared_registry(formal25: Path, formal26: Path) -> list[dict]:
    return [
        {
            "label": "cpu-tiny-whole-seed20261955",
            "run": TINY_WHOLE,
            "seed": 20261955,
            "source_commit": SOURCE,
            "expected_epochs": [1, 2],
            "kind": "tiny-CLI-control",
            "expected_inputs": {
                **COMMON_INPUTS,
                "experiment_config": "1ec162669b245380e4412c5ad4c45911bd982666e2b555d2e4fab0f8067550ff",
                "protocol": "b557d57940babdfd73be6aef565e986eead769ecf7560cd49ae361f33420ef84",
            },
            "split_duplicate_run": TINY_SPLIT,
            "qualification_receipt": TINY_WHOLE.parent / "result.json",
            "qualification_status": "pass",
            "qualification_schema": "actual-CPU-certificate-ledger-CLI-qualification-v1",
        },
        {
            "label": "cpu-development-E1-seed20261925",
            "run": E1_DEV_RUN,
            "seed": 20261925,
            "source_commit": SOURCE,
            "expected_epochs": [1],
            "kind": "development-E1",
            "expected_inputs": {
                **COMMON_INPUTS,
                "experiment_config": "18ecc92f92561c3da1a82c05b011ff9f702d5eee9ae86e1d2b2f924a558ec121",
                "protocol": "ac8f484daf1b0a07a4b3fd66bf55c0352d9b235b1f60cf47a6c0b52b4dc67834",
            },
            "qualification_receipt": E1_DEV_RUN.parent / "profile-result.json",
            "qualification_status": "pass-actual-CPU-E1-qualified",
            "qualification_schema": "actual-cpu-fullshape-original900-profile-v1",
        },
        {
            "label": "cpu-formal-seed20261925",
            "run": formal25,
            "seed": 20261925,
            "source_commit": SOURCE,
            "expected_epochs": list(range(1, 9)),
            "kind": "formal-fixed-E8",
            "expected_inputs": {
                **COMMON_INPUTS,
                "experiment_config": "18ecc92f92561c3da1a82c05b011ff9f702d5eee9ae86e1d2b2f924a558ec121",
                "protocol": "ac8f484daf1b0a07a4b3fd66bf55c0352d9b235b1f60cf47a6c0b52b4dc67834",
            },
        },
        {
            "label": "cpu-formal-seed20261926",
            "run": formal26,
            "seed": 20261926,
            "source_commit": SOURCE,
            "expected_epochs": list(range(1, 9)),
            "kind": "formal-fixed-E8",
            "expected_inputs": {
                **COMMON_INPUTS,
                "experiment_config": "6a0180492439297b9f4efdd53881b56ea3c293274d5d54a628b40070e7b50e4c",
                "protocol": "ac8f484daf1b0a07a4b3fd66bf55c0352d9b235b1f60cf47a6c0b52b4dc67834",
            },
        },
    ]


def verify_split_duplicate(spec: dict, epoch: int) -> dict:
    whole = spec["run"] / "journal" / f"epoch-{epoch:08d}.json.gz"
    split = spec["split_duplicate_run"] / "journal" / f"epoch-{epoch:08d}.json.gz"
    whole_bytes, split_bytes = whole.read_bytes(), split.read_bytes()
    same = whole_bytes == split_bytes
    if not same:
        raise ValueError(f"tiny whole/split journal differs at epoch {epoch}")
    return {"epoch": epoch, "whole_sha256": sha_bytes(whole_bytes), "split_sha256": sha_bytes(split_bytes), "byte_identical": True}


def audit_journal(spec: dict, epoch: int, candidates: set[str], seen_keys: set[str], inventory, guard) -> dict:
    run = spec["run"]
    meta_path = run / "metadata.json"
    meta_bytes = meta_path.read_bytes()
    meta = json.loads(meta_bytes)
    if meta.get("schema") != "search-acting-supervised-run-v2":
        raise ValueError(f"unexpected run schema: {spec['label']}")
    if meta.get("source_commit") != spec["source_commit"]:
        raise ValueError(f"source commit mismatch: {spec['label']}")
    if meta.get("config", {}).get("seed") != spec["seed"]:
        raise ValueError(f"seed mismatch: {spec['label']}")
    inputs = meta.get("inputs", {})
    if set(inputs) != set(spec["expected_inputs"]):
        raise ValueError(f"run input inventory mismatch: {spec['label']}")
    for name, digest in spec["expected_inputs"].items():
        if inputs[name].get("sha256") != digest:
            raise ValueError(f"run input digest mismatch ({name}): {spec['label']}")
    if meta.get("max_epochs", 0) < max(spec["expected_epochs"]):
        raise ValueError(f"run max_epochs underdeclares registry: {spec['label']}")
    journal = run / "journal" / f"epoch-{epoch:08d}.json.gz"
    native = run / "checkpoints" / f"epoch-{epoch:08d}"
    cp_path = native / "checkpoint.json"
    if not journal.is_file() or not cp_path.is_file():
        raise FileNotFoundError(f"epoch {epoch} journal/native not closed for {spec['label']}")
    raw = journal.read_bytes()
    record = json.loads(gzip.decompress(raw))
    if record.get("epoch") != epoch:
        raise ValueError(f"journal epoch mismatch: {spec['label']} e{epoch}")
    if record.get("schema") != "torch-fresh-sparse-search-acting-v3":
        raise ValueError(f"journal schema mismatch: {spec['label']} e{epoch}")
    if record.get("own_search", {}).get("schema") != "pre-action-masked-search-behavior-v4":
        raise ValueError(f"search ledger schema mismatch: {spec['label']} e{epoch}")
    actions = record["collection"]["actions"]
    if record.get("fresh_transitions") != len(actions):
        raise ValueError(f"fresh row count mismatch: {spec['label']} e{epoch}")
    expected_rows = meta["config"]["actors"]["games"] * meta["config"]["epoch_steps"]
    if len(actions) != expected_rows:
        raise ValueError(f"actual actor rows differ from frozen shape: {spec['label']} e{epoch}")
    cp_manifest = json.loads(cp_path.read_text())
    if cp_manifest.get("schema") != "torch-search-acting-native-cpu-v3":
        raise ValueError(f"native schema mismatch: {spec['label']} e{epoch}")
    if cp_manifest.get("source_commit") != spec["source_commit"]:
        raise ValueError(f"native source mismatch: {spec['label']} e{epoch}")
    archive = native / "last-frozen-epoch.json.gz"
    if sha_file(archive) != sha_bytes(raw):
        raise ValueError(f"frozen native archive differs from journal: {spec['label']} e{epoch}")
    if cp_manifest.get("state", {}).get("epoch") != epoch:
        raise ValueError(f"native checkpoint epoch mismatch: {spec['label']} e{epoch}")
    expected_payloads = {
        "model.safetensors",
        "base.safetensors",
        "behavior.safetensors",
        "training.pt",
        "actor.json",
        "last-frozen-epoch.json.gz",
    }
    if set(cp_manifest.get("artifacts", {})) != expected_payloads:
        raise ValueError(f"native payload inventory mismatch: {spec['label']} e{epoch}")
    for name, digest in cp_manifest["artifacts"].items():
        if sha_file(native / name) != digest:
            raise ValueError(f"native payload digest mismatch: {spec['label']} e{epoch} {name}")

    previous: dict[tuple, tuple[chess.Board, tuple]] = spec.setdefault("_previous", {})
    counts = {
        "raw_terminal_result_rows": 0,
        "raw_unknown_rollout_cutoff_rows": 0,
        "raw_unknown_active_or_boundary_rows": 0,
        "terminal_results": collections.Counter(),
        "terminal_terminations": collections.Counter(),
    }
    matches = []
    prefix_count = 0
    replayed_histories = 0
    for ordinal, action in enumerate(actions):
        if ordinal % 128 == 0:
            guard()
        trans = action["transition"]
        pre, post = trans["pre"], trans["post"]
        if post["root_fen"] != pre["root_fen"] or post["moves"] != pre["moves"] + [trans["action"]]:
            raise ValueError(f"pre/post history fields inconsistent: {spec['label']} e{epoch} row{ordinal}")
        game_key = (trans["slot"], trans["source_id"], trans["game_index"])
        if game_key in previous:
            board, expected_pre = previous[game_key]
            if state_sig(pre) != expected_pre:
                raise ValueError(f"actor full-history discontinuity: {spec['label']} e{epoch} row{ordinal}")
        else:
            board = replay(pre, guard)
            replayed_histories += 1

        pre_key = pos_key(board)
        seen_keys.add(pre_key)
        prefix_count += 1
        inventory.write(json.dumps([spec["label"], epoch, ordinal, "pre", pre_key], separators=(",", ":")) + "\n")
        if pre_key in candidates:
            matches.append({"key": pre_key, "run": spec["label"], "epoch": epoch, "row": ordinal, "where": "pre"})

        move = chess.Move.from_uci(trans["action"])
        if move not in board.legal_moves:
            raise ValueError(f"illegal actor action: {spec['label']} e{epoch} row{ordinal}")
        board.push(move)
        post_key = pos_key(board)
        seen_keys.add(post_key)
        prefix_count += 1
        inventory.write(json.dumps([spec["label"], epoch, ordinal, "post", post_key], separators=(",", ":")) + "\n")
        if post_key in candidates:
            matches.append({"key": post_key, "run": spec["label"], "epoch": epoch, "row": ordinal, "where": "post"})
        previous[game_key] = (board, state_sig(post))

        terminal = trans.get("terminal_result")
        termination = trans.get("terminal_termination")
        if terminal is None:
            if termination is not None:
                raise ValueError("UNKNOWN action carries a terminal termination")
            if trans.get("rollout_cutoff"):
                counts["raw_unknown_rollout_cutoff_rows"] += 1
            else:
                counts["raw_unknown_active_or_boundary_rows"] += 1
        else:
            if termination is None:
                raise ValueError("known terminal action lacks termination")
            if terminal not in {"1-0", "0-1", "1/2-1/2"}:
                raise ValueError(f"unexpected terminal result value: {terminal}")
            counts["raw_terminal_result_rows"] += 1
            counts["terminal_results"][terminal] += 1
            counts["terminal_terminations"][termination] += 1

    return {
        "label": spec["label"],
        "kind": spec["kind"],
        "path": str(journal),
        "epoch": epoch,
        "journal_sha256": sha_bytes(raw),
        "journal_bytes": len(raw),
        "journal_schema": record.get("schema"),
        "metadata_sha256": sha_bytes(meta_bytes),
        "native_checkpoint_sha256": sha_file(cp_path),
        "native_payload_count": len(cp_manifest.get("artifacts", {})),
        "rows": len(actions),
        "prepost_prefix_positions": prefix_count,
        "fresh_game_histories_replayed_from_root": replayed_histories,
        "raw_terminal_result_rows": counts["raw_terminal_result_rows"],
        "raw_unknown_rollout_cutoff_rows": counts["raw_unknown_rollout_cutoff_rows"],
        "raw_unknown_active_or_epoch_boundary_rows": counts["raw_unknown_active_or_boundary_rows"],
        "terminal_results": dict(sorted(counts["terminal_results"].items())),
        "terminal_terminations": dict(sorted(counts["terminal_terminations"].items())),
        "matched_positions": matches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-61925", type=Path, required=True)
    parser.add_argument("--formal-61926", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=600)
    args = parser.parse_args()
    if not 0 < args.seconds <= 600 or args.output.exists():
        raise ValueError("need a fresh output directory and a 1..600 second audit cap")
    started = time.time()
    deadline = started + args.seconds
    if deadline >= HARD_END:
        raise TimeoutError("audit window crosses hard campaign deadline")
    mono_deadline = time.monotonic() + args.seconds

    def guard():
        if time.monotonic() >= mono_deadline:
            raise TimeoutError("fixed overlap-audit time budget exhausted")

    candidates, books = book_roots()
    registry = declared_registry(args.formal_61925.resolve(), args.formal_61926.resolve())
    args.output.mkdir(parents=True)
    inventory_path = args.output / "inclusive-prefix-position-inventory.jsonl.gz"
    receipts = []
    missing = []
    overlap = []
    qualification_receipts = []
    seen_keys: set[str] = set()
    with gzip.open(inventory_path, "wt", encoding="utf-8", newline="\n") as inventory:
        for spec in registry:
            guard()
            run = spec["run"]
            qualification_path = spec.get("qualification_receipt")
            if qualification_path is not None:
                if not qualification_path.is_file():
                    missing.append(f"{spec['label']}:qualification-receipt")
                else:
                    qualification = json.loads(qualification_path.read_text())
                    if (
                        qualification.get("schema") != spec["qualification_schema"]
                        or qualification.get("status") != spec["qualification_status"]
                        or qualification.get("source_commit") != SOURCE
                    ):
                        raise ValueError(f"qualification receipt does not pass: {spec['label']}")
                    qualification_receipts.append(
                        {"label": spec["label"], "path": str(qualification_path), "sha256": sha_file(qualification_path)}
                    )
            if not run.is_dir():
                missing.extend(f"{spec['label']}:epoch-{e}" for e in spec["expected_epochs"])
                continue
            meta_path = run / "metadata.json"
            if not meta_path.is_file():
                missing.extend(f"{spec['label']}:epoch-{e}" for e in spec["expected_epochs"])
                continue
            meta = json.loads(meta_path.read_text())
            if meta.get("max_epochs") != max(spec["expected_epochs"]):
                raise ValueError(f"declared epoch schedule differs from run metadata: {spec['label']}")
            for epoch in spec["expected_epochs"]:
                journal = run / "journal" / f"epoch-{epoch:08d}.json.gz"
                checkpoint = run / "checkpoints" / f"epoch-{epoch:08d}/checkpoint.json"
                if not journal.is_file() or not checkpoint.is_file():
                    missing.append(f"{spec['label']}:epoch-{epoch}")
                    # Do not compare a later observed epoch with a stale prior
                    # board across an omitted epoch; replay its saved history.
                    spec.pop("_previous", None)
                    continue
                row = audit_journal(spec, epoch, candidates, seen_keys, inventory, guard)
                receipts.append(row)
                overlap.extend(row["matched_positions"])
                if spec["label"] == "cpu-tiny-whole-seed20261955":
                    receipts[-1]["split_resume_duplicate"] = verify_split_duplicate(spec, epoch)

    inventory_path = args.output / "inclusive-prefix-position-inventory.jsonl.gz"
    inventory_sha = sha_file(inventory_path)
    formal_missing = [m for m in missing if "cpu-formal-" in m]
    current_missing = [m for m in missing if "cpu-tiny-" in m or "cpu-development-" in m]
    full_status = full_coverage_status(missing, overlap)
    current_status = current_subset_status(missing, overlap)
    result = {
        "schema": "family8-cpu-own-training-prefix-overlap-v1",
        "status": full_status,
        "current_tiny_and_E1_status": current_status,
        "audit_script_sha256": sha_file(Path(__file__)),
        "scope": "read-only train/development actor histories; no model, engine, optimizer, SF, or quality/outcome queries",
        "started_epoch": started,
        "deadline_epoch": deadline,
        "seconds_budget": args.seconds,
        "book_roots": books,
        "candidate_root_count": len(candidates),
        "candidate_root_set_sha256": sha_bytes(("\n".join(sorted(candidates)) + "\n").encode()),
        "declared_registry": [
            {"label": spec["label"], "path": str(spec["run"]), "seed": spec["seed"], "expected_epochs": spec["expected_epochs"], "kind": spec["kind"], "expected_inputs": spec["expected_inputs"]}
            for spec in registry
        ],
        "journal_receipts": receipts,
        "current_qualification_receipts": qualification_receipts,
        "missing_expected_journals": missing,
        "formal_missing_expected_journals": formal_missing,
        "current_missing_expected_journals": current_missing,
        "matched_candidate_positions": overlap,
        "matched_position_count": len(overlap),
        "inclusive_prepost_prefix_positions": sum(row["prepost_prefix_positions"] for row in receipts),
        "unique_position_keys": len(seen_keys),
        "inclusive_position_inventory": {"path": str(inventory_path), "sha256": inventory_sha, "bytes": inventory_path.stat().st_size},
        "raw_transition_terminal_and_UNKNOWN_classifications_preserved": True,
        "raw_terminal_result_rows_are_not_learner_value_target_counts": True,
        "complete_formal_coverage_required_for_full_pass": True,
        "future_missing_natives_are_not_claimed_covered": True,
        "finished_epoch": time.time(),
    }
    out = args.output / "result.json"
    out.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": full_status, "current_status": current_status, "journals": len(receipts), "missing": len(missing), "overlap_positions": len(overlap), "unique_keys": len(seen_keys), "inventory_sha256": inventory_sha}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

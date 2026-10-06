"""Run one externally registered, outcome-blind 1,024-root reanalysis job.

This CLI refuses the local unregistered proposal template. A frozen ROOT
registration must pin every helper, input, source tree, owner core and clock.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from action_labels import run_selected
from select_roots import select_roots


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_pinned(path, expected, module_name):
    path = Path(path).resolve()
    if sha(path) != expected:
        raise ValueError(f"pinned helper SHA differs: {path.name}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path:
        raise ValueError("imported helper origin differs")
    return module


def checked_digest(text, size):
    if not isinstance(text, str) or len(text) != size:
        raise ValueError("invalid registered digest")
    int(text, 16)
    return text


def _position_groups(config, state, packets, training, protected, digest):
    known = {index: positions for _, index, positions, _ in packets}
    rows_by_game = {}
    seen = set()
    for game_index, game in enumerate(state["games"]):
        if game_index not in known:
            continue
        root = config["roots"][game["root_index"]]
        moves = [row["action"] for row in game["moves"]]
        key = digest({"root_fen": root["root_fen"], "prefix": root["prefix"], "moves": moves})
        if key not in training or key in seen:
            continue
        seen.add(key)
        rows = []
        prefix = list(root["prefix"])
        touched = set()
        for local, ((board, _mover, _stored_prior), action_row) in enumerate(
            zip(known[game_index], game["moves"], strict=True)
        ):
            touched.add(" ".join(board.fen().split()[:4]))
            rows.append(
                {
                    "row_id": f"{key}:{local}",
                    "trajectory_id": key,
                    "root_fen": root["root_fen"],
                    "prefix_uci": prefix.copy(),
                }
            )
            prefix.append(action_row["action"])
        final_board = known[game_index][-1][0].copy(stack=True)
        final_board.push_uci(game["moves"][-1]["action"])
        touched.add(" ".join(final_board.fen().split()[:4]))
        if touched.intersection(protected):
            raise ValueError("admitted training trajectory touches protected position")
        rows_by_game[key] = rows
    if set(rows_by_game) != set(training):
        raise ValueError("training trajectory map did not exactly reconstruct")
    return rows_by_game


def execute(registration_path):
    reg_path = Path(registration_path).resolve()
    reg = json.loads(reg_path.read_text())
    if reg.get("schema") != "own-search-deeper-bestmove-reanalysis-registration-v1":
        raise ValueError("ROOT registration schema required")
    if reg.get("status") != "registered":
        raise ValueError("unregistered proposal cannot run")
    if reg.get("search") != {
        "roots_per_seed": 1024,
        "nodes": 8192,
        "qdepth": 2,
        "max_depth": 8,
        "evaluator": "frozen-classical-human-prior",
        "selection": "trajectory-balanced-sha-v1",
    }:
        raise ValueError("search contract differs")

    source_repo = Path(reg["source_repo"]).resolve()
    clean_source = Path(reg["source_dir"]).resolve()
    if not clean_source.is_dir():
        raise ValueError("pinned classical helper directory is missing")
    core_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_repo, text=True
    ).strip()
    if core_head != reg["source_commit"] or subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=source_repo, text=True
    ):
        raise ValueError("source checkout must match the clean registered commit")
    helper_repo = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=clean_source, text=True
        ).strip()
    ).resolve()
    helper_commit = checked_digest(reg["helper_source_commit"], 40)

    closure = reg["closure"]
    if set(closure) != {
        "launcher",
        "selector",
        "executor",
        "action_model",
        "journal_helper",
        "learner_helper",
        "value_helper",
        "search_helper",
        "runtime_helper",
        "cgroup_helper",
    }:
        raise ValueError("source closure keyset differs")
    expected_paths = {
        "launcher": Path(__file__).resolve(),
        "selector": Path(__file__).with_name("select_roots.py").resolve(),
        "executor": Path(__file__).with_name("action_labels.py").resolve(),
        "action_model": Path(__file__).with_name("action_model.py").resolve(),
        "journal_helper": clean_source / "journal_v3.py",
        "learner_helper": clean_source / "learner.py",
        "value_helper": clean_source / "value.py",
        "search_helper": clean_source / "arena/search.py",
        "runtime_helper": clean_source / "runtime.py",
        "cgroup_helper": source_repo / "src/harbichess/training/cgroup_budget.py",
    }
    for key, path in expected_paths.items():
        expected_sha = checked_digest(closure[key], 64)
        if sha(path) != expected_sha:
            raise ValueError(f"registered closure SHA differs: {key}")
        if key in {
            "journal_helper",
            "learner_helper",
            "value_helper",
            "search_helper",
            "runtime_helper",
        }:
            relative = path.relative_to(helper_repo).as_posix()
            committed = subprocess.check_output(
                ["git", "show", f"{helper_commit}:{relative}"], cwd=helper_repo
            )
            if hashlib.sha256(committed).hexdigest() != expected_sha:
                raise ValueError(f"helper bytes are not from registered commit: {key}")

    first = float(reg["original_first_epoch"])
    deadline = float(reg["original_deadline_epoch"])
    if deadline != first + 7200 or not time.time() < deadline:
        raise TimeoutError("the original 7,200-second window is invalid or expired")
    core = reg["cpu_core"]
    if not isinstance(core, int) or isinstance(core, bool) or core < 0:
        raise ValueError("registered CPU core required")
    os.sched_setaffinity(0, {core})
    output = Path(reg["output_path"]).resolve()
    if not str(output).startswith("/dev/shm/harbichess-own-action-reanalysis/"):
        raise ValueError("all reanalysis artifacts must remain in the RAM work area")
    if output.exists():
        raise FileExistsError("publish-once output already exists")

    runtime = load_pinned(
        expected_paths["runtime_helper"], closure["runtime_helper"], "ownq_runtime"
    )
    runtime.affinity(core)

    # The repository source directory is the only allowed helper import root.
    sys.path.insert(0, str(clean_source))
    journal = load_pinned(expected_paths["journal_helper"], closure["journal_helper"], "journal_v3")
    value = load_pinned(expected_paths["value_helper"], closure["value_helper"], "value")
    learner = load_pinned(expected_paths["learner_helper"], closure["learner_helper"], "learner")
    search = load_pinned(expected_paths["search_helper"], closure["search_helper"], "ownq_search")

    config_path = Path(reg["config_path"]).resolve()
    journal_path = Path(reg["journal_path"]).resolve()
    fit_contract_path = Path(reg["fit_contract_path"]).resolve()
    if sha(config_path) != checked_digest(reg["config_sha256"], 64):
        raise ValueError("actor config changed")
    if sha(journal_path) != checked_digest(reg["journal_sha256"], 64):
        raise ValueError("actor journal changed")
    if sha(fit_contract_path) != checked_digest(reg["fit_contract_sha256"], 64):
        raise ValueError("source fit-contract receipt changed")
    config = json.loads(config_path.read_text())
    if config["seed"] != reg["seed"]:
        raise ValueError("actor seed differs")
    fit_payload = json.loads(fit_contract_path.read_text())
    fit_contract = fit_payload.get("contract")
    if (
        not isinstance(fit_contract, dict)
        or fit_contract.get("journal_sha256") != reg["journal_sha256"]
        or fit_contract.get("seed") != reg["seed"]
        or fit_contract.get("config_sha256") != reg["config_sha256"]
        or fit_contract.get("source_commit") != reg["source_commit"]
    ):
        raise ValueError("fit contract is not bound to the actor/source")
    state = journal.read(journal_path)
    packets = journal.replay(state, config)
    protected = config["excluded_training_position_keys"]
    if (
        reg["protected_position_keys"] != protected
        or reg["exclusion_book_sha256"] != config["exclusion_book_sha256"]
    ):
        raise ValueError("protected training exclusion differs")
    training, _validation, receipt, dataset_sha = learner.prepare(journal_path, config, protected)
    if (
        dataset_sha != fit_contract.get("dataset_sha256")
        or receipt != fit_contract.get("receipt")
        or dataset_sha != reg["training_dataset_sha256"]
        or receipt != reg["training_receipt"]
    ):
        raise ValueError("training split/preparation receipt differs")
    groups = _position_groups(config, state, packets, training, set(protected), journal.digest)
    selected = select_roots(groups, reg["seed"], 1024)

    prior = value.ClassicalValue()
    if value.model_dict() != reg["frozen_prior_model"]:
        raise ValueError("frozen human prior identity differs")

    def guard():
        runtime.guard(deadline, source_repo, output.parent)

    def factory():
        return search.BudgetSearch(
            prior.nonterminal,
            nodes=8192,
            quiescence_plies=2,
            max_depth=8,
            guard=guard,
        )

    guard()
    started = time.time()

    def heartbeat(completed):
        print(
            json.dumps(
                {
                    "completed_roots": completed,
                    "elapsed_seconds": round(time.time() - started, 3),
                }
            ),
            flush=True,
        )

    reference = None
    if reg.get("reference_value_labels") is not None:
        binding = reg["reference_value_labels"]
        expected_path = (
            f"/dev/shm/harbichess-own-q-reanalysis/{reg['seed']}/labels-00001024.json.gz"
        )
        if (
            set(binding) != {"path", "sha256"}
            or binding["path"] != expected_path
            or sha(binding["path"]) != binding["sha256"]
        ):
            raise ValueError("exact optional original value-label binding")
        raw_reference = gzip.decompress(Path(binding["path"]).read_bytes())
        if len(raw_reference) > 16 * 1024**2:
            raise ValueError("reference value label bound")
        old_values = json.loads(raw_reference)
        for key, expected in {
            "schema": "own-search-deeper-value-labels-v1",
            "seed": reg["seed"],
            "source_commit": reg["source_commit"],
            "journal_sha256": reg["journal_sha256"],
            "config_sha256": reg["config_sha256"],
            "training_dataset_sha256": dataset_sha,
        }.items():
            if old_values.get(key) != expected:
                raise ValueError("reference must be same own TRAIN data/source")
        reference = {row["row_id"]: row for row in old_values["roots"]}
        if len(reference) != 1024:
            raise ValueError("reference exact root coverage")
    rows = run_selected(
        selected,
        factory,
        expected_count=1024,
        nodes=8192,
        progress=heartbeat,
        state_features=value.features,
        reference=reference,
    )
    guard()
    payload = {
        "schema": "own-search-deeper-bestmove-labels-v1",
        "registration_sha256": sha(reg_path),
        "source_commit": reg["source_commit"],
        "closure": closure,
        "config_sha256": reg["config_sha256"],
        "journal_sha256": reg["journal_sha256"],
        "training_dataset_sha256": dataset_sha,
        "training_receipt": receipt,
        "frozen_prior_model": reg["frozen_prior_model"],
        "seed": reg["seed"],
        "search": reg["search"],
        "target_semantics": (
            "selected-bestmove-only legal CE; raw mover q is audit context, "
            "not new WDL/visit target"
        ),
        "roots": rows,
    }
    compressed = gzip.compress(canonical(payload), mtime=0)
    if len(compressed) > 16 * 1024 * 1024:
        raise ValueError("label artifact exceeds bounded RAM artifact size")
    guard()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(compressed)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, output)
    finally:
        temporary.unlink()
    guard()
    print(json.dumps({"output": str(output), "sha256": sha(output), "roots": len(rows)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    args = parser.parse_args()
    execute(args.registration)


if __name__ == "__main__":
    main()

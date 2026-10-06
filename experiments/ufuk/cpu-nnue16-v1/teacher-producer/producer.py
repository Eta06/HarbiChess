"""ROOT-registered teacher-once labels; metadata-only mode never opens SF."""

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

import chess
import chess.engine
from selection import canonical_key, history_sha, select, teacher_target

SF_SHA = "0f83d24cc46d2c66c60f16001af5444873bc112b7d028594513426894c12da19"
SF_ASSET = 545574859
SF_COMPRESSED_SHA = "9defc0d4e55d49c65a6d042f3e571a39fcea499ade6dbe741b53b8c65e03611f"
HERE = Path(__file__).parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def load(path, expected, name):
    path = Path(path).resolve()
    if sha(path) != expected:
        raise ValueError("pinned source SHA: " + path.name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def add_history(board, root, prefix, positions, histories):
    positions.add(canonical_key(board))
    histories.add(history_sha(root, prefix))


def played_exclusions(manifest):
    positions, histories = set(), set()
    book = manifest["known8_book"]
    if sha(book["path"]) != book["sha256"] or not manifest["played_files"]:
        raise ValueError("complete pinned known8/played-state exclusion inventory required")
    openings = json.loads(Path(book["path"]).read_bytes())["splits"]["arena"]
    if len(openings) != 8:
        raise ValueError("exact known8 book")
    for packet in openings:
        board = chess.Board()
        prefix = []
        for uci in packet["opening"]["moves"]:
            if chess.Move.from_uci(uci) not in board.legal_moves:
                raise ValueError("illegal exclusion book history")
            prefix.append(uci)
            board.push_uci(uci)
            add_history(board, chess.STARTING_FEN, prefix, positions, histories)
    for row in manifest["played_files"]:
        if sha(row["path"]) != row["sha256"]:
            raise ValueError("played-state exclusion file changed")
        if Path(row["path"]).stat().st_size > 32 * 2**20:
            raise ValueError("closed exclusion file32MiB cap")
        packet = json.loads(Path(row["path"]).read_bytes())
        if packet["opening_source_sha256"] != book["sha256"]:
            raise ValueError("wrong known8 played source")
        for game in packet["games"]:
            root = game.get("root_fen", chess.STARTING_FEN)
            board, prefix = chess.Board(root), []
            for uci in game["moves"]:
                move = chess.Move.from_uci(uci)
                if move not in board.legal_moves:
                    raise ValueError("illegal played exclusion history")
                board.push(move)
                prefix.append(uci)
                add_history(board, root, prefix, positions, histories)
    return positions, histories


def reconstruct(reg, guard):
    parent = json.loads(Path(reg["parent_registration"]["path"]).read_bytes())
    if sha(reg["parent_registration"]["path"]) != reg["parent_registration"]["sha256"]:
        raise ValueError("original CLASSIC TRAIN lineage registration differs")
    if parent["seed"] != reg["seed"] or parent["source_commit"] != reg["core_source_commit"]:
        raise ValueError("same exact original CLASSIC seed/source")
    source = Path(parent["source_dir"])
    sys.path.insert(0, str(source))
    journal = load(source / "journal_v3.py", parent["closure"]["journal_helper"], "journal_v3")
    load(source / "value.py", parent["closure"]["value_helper"], "value")
    learner = load(source / "learner.py", parent["closure"]["learner_helper"], "learner")
    for key, h in [
        ("config_path", "config_sha256"),
        ("journal_path", "journal_sha256"),
        ("fit_contract_path", "fit_contract_sha256"),
    ]:
        if sha(parent[key]) != parent[h]:
            raise ValueError("original journal/config/native lineage SHA")
    config = json.loads(Path(parent["config_path"]).read_bytes())
    state = journal.read(parent["journal_path"])
    training, validation, receipt, digest = learner.prepare(
        parent["journal_path"], config, config["excluded_training_position_keys"]
    )
    if (
        digest != parent["training_dataset_sha256"]
        or receipt != parent["training_receipt"]
        or config["seed"] != reg["seed"]
    ):
        raise ValueError("exact original TRAIN/VAL/dedup/UNKNOWN preparation")
    packets = journal.replay(state, config)
    known = {i: pos for _, i, pos, _ in packets}
    groups, excluded_positions, excluded_histories = {}, set(), set()
    games = list(state["games"]) + ([state["active"]] if state.get("active") else [])
    for index, game in enumerate(games):
        guard()
        root = config["roots"][game["root_index"]]
        moves = [r["action"] for r in game["moves"]]
        trajectory = journal.digest(
            dict(root_fen=root["root_fen"], prefix=root["prefix"], moves=moves)
        )
        prefix, board = list(root["prefix"]), chess.Board(root["root_fen"])
        for move in prefix:
            board.push_uci(move)
        admitted = index in known and trajectory in training
        rows = []
        for local, action in enumerate(moves):
            if not admitted:
                add_history(board, root["root_fen"], prefix, excluded_positions, excluded_histories)
            else:
                rows.append(
                    dict(
                        row_id=f"{trajectory}:{local}",
                        trajectory_id=trajectory,
                        root_fen=root["root_fen"],
                        prefix_uci=prefix.copy(),
                        history_sha256=history_sha(root["root_fen"], prefix),
                        fen4=canonical_key(board),
                    )
                )
            board.push_uci(action)
            prefix.append(action)
        if not admitted:
            add_history(board, root["root_fen"], prefix, excluded_positions, excluded_histories)
        elif trajectory not in groups:
            groups[trajectory] = rows
    if set(groups) != set(training):
        raise ValueError("complete realized deduplicated TRAIN trajectories not reconstructed")
    exclusion = reg["known8_exclusions"]
    if sha(exclusion["path"]) != exclusion["sha256"]:
        raise ValueError("known8 inventory SHA")
    positions, histories = played_exclusions(json.loads(Path(exclusion["path"]).read_bytes()))
    excluded_positions |= positions
    excluded_histories |= histories
    return (
        groups,
        excluded_positions,
        excluded_histories,
        dict(
            original_training_rows=receipt["training_rows"],
            original_training_games=len(training),
            original_validation_games=len(validation),
            original_training_dataset_sha256=digest,
            excluded_current_states=len(excluded_positions),
            excluded_fullhistories=len(excluded_histories),
            shared_START_ancestry_allowed=True,
            original_lineage=reg["parent_registration"],
        ),
    )


def execute(reg_path, count_only=False):
    reg = json.loads(Path(reg_path).read_bytes())
    if (
        reg["schema"] != "NNUE-teacher8192-ROOT-registration-v1"
        or reg["status"] != "registered"
        or reg["count"] != 4096
        or reg["seed"] not in [20262905, 20262906]
    ):
        raise ValueError("ROOT-fixed teacheronce4096 registration required")
    expected_phase = "metadata-count-only" if count_only else "teacher-labels"
    if reg["phase"] != expected_phase:
        raise ValueError("metadata and actual teacher require separate prospective scopes")
    first, end = reg["original_first_epoch"], reg["original_deadline_epoch"]
    if count_only and end > first + 60:
        raise ValueError("bounded metadata-only60s")
    if not first <= time.time() < end <= 1791273600:
        raise ValueError("separate prospective original teacher clock")
    os.sched_setaffinity(0, {reg["cpu_core"]})
    for path, expected in reg["source_sha256"].items():
        if sha(path) != expected:
            raise ValueError("complete teacher source closure")
    if set(reg["source_sha256"]) != {str(HERE / "producer.py"), str(HERE / "selection.py")}:
        raise ValueError("fixed two-file producer closure")
    core = Path(reg["core_source_repo"])
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=core, text=True).strip() != reg[
        "core_source_commit"
    ] or subprocess.check_output(["git", "status", "--porcelain"], cwd=core, text=True):
        raise ValueError("clean core source")
    sys.path.insert(0, str(core / "src"))
    from harbichess.training.cgroup_budget import CgroupMemoryBudget

    memory = CgroupMemoryBudget(15 * 2**30)

    def guard():
        if time.time() >= end:
            raise TimeoutError("original producer clock exhausted")
        memory.check()
        import shutil

        if shutil.disk_usage("/workspace").free < 256 * 2**20:
            raise RuntimeError("diskfloor256MiB")

    groups, positions, histories, receipt = reconstruct(reg, guard)
    rows = select(
        groups,
        seed=reg["seed"],
        count=4096,
        excluded_positions=positions,
        excluded_histories=histories,
    )
    receipt.update(
        selected_roots=len(rows),
        selected_fullhistory_sha256=hashlib.sha256(canonical(rows)).hexdigest(),
        selected_trajectories=len({r["trajectory_id"] for r in rows}),
    )
    output = Path(reg["output"])
    if not output.resolve().is_relative_to("/dev/shm"):
        raise ValueError("RAM-only immutable outputs")
    output.mkdir(parents=True, exist_ok=False)
    (output / "selection.json").write_bytes(canonical(dict(rows=rows, receipt=receipt)) + b"\n")
    if count_only:
        guard()
        (output / "metadata-only-result.json").write_bytes(canonical(receipt) + b"\n")
        return
    teacher = reg["teacher"]
    if (
        teacher["sha256"] != SF_SHA
        or teacher["asset_id"] != SF_ASSET
        or teacher["compressed_sha256"] != SF_COMPRESSED_SHA
        or sha(teacher["path"]) != SF_SHA
    ):
        raise ValueError("official SF19 fixed binary/source asset")
    parent = json.loads(Path(reg["parent_registration"]["path"]).read_bytes())
    module = load(
        Path(parent["source_dir"]) / "value.py", parent["closure"]["value_helper"], "value"
    )
    nnue = load(reg["nnue_feature"]["path"], reg["nnue_feature"]["sha256"], "nnue_teacher_features")
    labels = []
    partial = (output / "partial-labels.jsonl").open("xb")
    engine = chess.engine.SimpleEngine.popen_uci(
        teacher["path"], timeout=min(15, end - time.time())
    )
    try:
        engine.configure({"Threads": 1, "Hash": 16})
        for ordinal, row in enumerate(rows):
            guard()
            board = chess.Board(row["root_fen"])
            for uci in row["prefix_uci"]:
                board.push_uci(uci)
            if board.outcome(claim_draw=True) is not None:
                raise ValueError("terminal/claimable teacher root")
            engine.configure({"Clear Hash": None})
            started = time.time()
            engine.timeout = max(0.001, min(15, end - started))
            info = engine.analyse(board, chess.engine.Limit(nodes=8192))
            guard()
            raw = teacher_target(info["score"], board.turn)
            if info.get("nodes", 0) < 1 or not info.get("pv"):
                raise ValueError("incomplete nominal8192 teacher search")
            if info["pv"][0] not in board.legal_moves:
                raise ValueError("illegal teacher best move")
            labels.append(
                dict(
                    row,
                    ordinal=ordinal,
                    root_mover="white" if board.turn else "black",
                    teacher_nodes_actual=info["nodes"],
                    nominal_nodes=8192,
                    node_overrun=info["nodes"] - 8192,
                    teacher_depth=info.get("depth"),
                    teacher_time_seconds=time.time() - started,
                    raw_teacher=raw,
                    selected_teacher_uci=info["pv"][0].uci(),
                    indices=nnue.board_indices(board),
                    prior_logit=sum(
                        w * x for w, x in zip(module.PRIOR, module.features(board), strict=True)
                    )
                    / module.SCALE,
                    target=raw["target"],
                )
            )
            partial.write(canonical(labels[-1]) + b"\n")
            if (ordinal + 1) % 64 == 0:
                partial.flush()
                os.fsync(partial.fileno())
    finally:
        partial.close()
        engine.quit()
    result = dict(
        schema="NNUE-teacher8192-ownTRAIN-bootstrap-labels-v1",
        seed=reg["seed"],
        phase="teacher-bootstrap",
        rows=labels,
        selection_receipt=receipt,
        teacher=teacher,
        registration_sha256=sha(reg_path),
        source_closure=reg["source_sha256"],
        original_first_epoch=first,
        original_deadline_epoch=end,
        finished_epoch=time.time(),
        teacher_bootstrap_is_selflearning=False,
        targets_are_calibrated_WDL=False,
    )
    guard()
    (output / "labels-00004096.json.gz").write_bytes(gzip.compress(canonical(result), mtime=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--count-only", action="store_true")
    args = parser.parse_args()
    execute(args.registration, args.count_only)

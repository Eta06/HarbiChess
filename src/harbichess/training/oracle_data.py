"""Versioned engine-reference data, distinct from terminal-outcome replay."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import multiprocessing
import os
import random
import resource
import subprocess
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import chess
import chess.engine
import torch

from harbichess.backends.torch_network import load_weights, sha256
from harbichess.chess.actions import move_to_action
from harbichess.chess.encoding import BoardEncoder
from harbichess.core.state import ChessMove, ChessState
from harbichess.evaluation.teacher_probe import reference_info
from harbichess.training.torch_learner import TorchTrainingBatch

ORACLE_SCHEMA = 1


def complete_multipv(packets, count: int) -> list[dict]:
    """Do not combine stale merged bounds or different-depth PV scores."""
    group: list[dict] = []
    depth = None
    selected = None
    for info in packets:
        if not all(k in info for k in ("score", "wdl", "pv", "depth")) or not info["pv"]:
            continue
        index = info.get("multipv", 1)
        if index == 1:
            group, depth = [], info["depth"]
        if (
            info["depth"] != depth
            or index != len(group) + 1
            or info.get("lowerbound")
            or info.get("upperbound")
        ):
            group, depth = [], None
            continue
        group.append(dict(info))
        if len(group) == count:
            if len({x["pv"][0] for x in group}) != count:
                raise ValueError("duplicate MultiPV moves")
            if selected is None or info["depth"] >= selected[0]["depth"]:
                selected = list(group)
    if selected is None:
        raise ValueError("no coherent completed MultiPV iteration")
    return selected


def soft_policy(references: list[dict], temperature: float = 100) -> list[float]:
    scores = [r["cp_or_mate_score"] for r in references]
    if not scores or temperature <= 0 or not all(math.isfinite(s) for s in scores):
        raise ValueError("invalid policy scores/temperature")
    masses = [math.exp((s - max(scores)) / temperature) for s in scores]
    return [m / sum(masses) for m in masses]


def publish_json(path: Path, data: dict) -> None:
    """Publish only complete files, never replace an existing immutable record."""
    content = (json.dumps(data, sort_keys=True) + "\n").encode()
    if path.suffix == ".gz":
        content = gzip.compress(content, mtime=0)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    # A hard link publishes without overwrite even if another writer races.
    os.link(temporary, path)
    temporary.unlink()


def read_game(path: Path) -> dict:
    with gzip.open(path, "rt") as stream:
        data = json.load(stream)
    if data["schema"] != ORACLE_SCHEMA or data["target_semantics"] != "engine-reference":
        raise ValueError("unsupported oracle data; terminal replay is not interchangeable")
    return data


def generate_game(job: dict) -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(job["seed"])
    rng = random.Random(job["seed"])
    board = chess.Board()
    for move in job["opening"]:
        board.push_uci(move)
    network = load_weights(Path(job["weights"])).eval() if job["actor"] == "neural-engine" else None
    engine = chess.engine.SimpleEngine.popen_uci(job["stockfish"], timeout=15)
    engine.configure({"Threads": 1, "Hash": 16, "UCI_ShowWDL": True})
    started = time.perf_counter()
    rows, query_nodes, calls = [], 0, 0

    def analyse(nodes: int, count: int) -> list[dict]:
        nonlocal query_nodes, calls
        remaining = job["deadline"] - time.time()
        if remaining <= 0:
            raise TimeoutError("oracle generation wall budget exhausted")
        timer = threading.Timer(min(15, remaining), engine.close)
        timer.start()
        packets = []
        try:
            with engine.analysis(
                board, chess.engine.Limit(nodes=nodes), multipv=count, game=object()
            ) as analysis:
                for packet in analysis:
                    packets.append(packet)
                info = complete_multipv(packets, count)
                analysis.wait()
                actual = analysis.info.get("nodes", info[-1].get("nodes", 0))
        except Exception as error:
            publish_json(
                Path(job["output"]).with_suffix(".failure.json"),
                {
                    "error": str(error),
                    "job": {k: v for k, v in job.items() if k != "deadline"},
                    "fen": board.fen(),
                    "moves": [m.uci() for m in board.move_stack],
                    "rows_before_failure": len(rows),
                    "uci_packets": [str(p) for p in packets],
                },
            )
            raise
        finally:
            timer.cancel()
        calls += 1
        query_nodes += actual
        return [
            {**reference_info(i, board), "actual_query_nodes": actual, "budget_nodes": nodes}
            for i in info
        ]

    try:
        while len(board.move_stack) < 160 and board.outcome(claim_draw=True) is None:
            ply = len(board.move_stack)
            legal = sorted(board.legal_moves, key=lambda move: move.uci())
            labelled = job.get("label_every_ply", False) or (ply - 8) % 2 == job["game"] % 2
            refs = analyse(32768, min(4, len(legal))) if labelled else None
            probabilities = soft_policy(refs) if refs else None
            if labelled:
                rows.append(
                    {
                        "root_fen": chess.STARTING_FEN,
                        "moves": [move.uci() for move in board.move_stack],
                        "fen": board.fen(),
                        "position_key": " ".join(board.fen().split()[:4]),
                        "family": job["family"],
                        "game": job["game"],
                        "split": job["split"],
                        "legal": [[move.uci(), move_to_action(board, move)] for move in legal],
                        "policy": [
                            [
                                ref["move"],
                                move_to_action(board, chess.Move.from_uci(ref["move"])),
                                p,
                            ]
                            for ref, p in zip(refs, probabilities, strict=True)
                        ],
                        "wdl": refs[0]["wdl"],
                        "references": refs,
                    }
                )
            if network is not None and board.turn == job["neural_color"]:
                state = ChessState(
                    chess.STARTING_FEN, tuple(ChessMove(m.uci()) for m in board.move_stack)
                )
                position = BoardEncoder().encode_state(state, board)
                inputs = torch.tensor(position.values).reshape(1, 8, 8, 104)
                with torch.no_grad():
                    logits, _ = network(inputs)
                move = max(legal, key=lambda m: float(logits[0, move_to_action(board, m)]))
            else:
                if refs is None:
                    refs = analyse(4096, min(4, len(legal)))
                    probabilities = soft_policy(refs)
                index = (
                    rng.choices(range(len(refs)), probabilities)[0]
                    if ply < 40 and rng.random() < 0.2
                    else 0
                )
                move = chess.Move.from_uci(refs[index]["move"])
            if move not in board.legal_moves:
                raise ValueError("illegal actor/teacher move")
            board.push(move)
        outcome = board.outcome(claim_draw=True)
        result = {
            "schema": ORACLE_SCHEMA,
            "target_semantics": "engine-reference",
            "job": {k: v for k, v in job.items() if k != "deadline"},
            "engine_id": engine.id,
            "rows": rows,
            "final_moves": [move.uci() for move in board.move_stack],
            "observed_result": board.result(claim_draw=True) if outcome else None,
            "termination": outcome.termination.name if outcome else "max_plies",
            "queries": calls,
            "actual_nodes": query_nodes,
            "wall_seconds": time.perf_counter() - started,
        }
        publish_json(Path(job["output"]), result)
        return {
            k: result[k] for k in ("queries", "actual_nodes", "wall_seconds", "termination")
        } | {"rows": len(rows)}
    finally:
        engine.close()


def generate(
    directory: Path,
    openings: Path,
    weights: Path,
    stockfish: Path,
    *,
    wall_seconds=1800,
    games_per_family: int = 2,
    workers: int = 4,
    seed: int | None = None,
    balanced: bool = False,
) -> dict:
    if games_per_family < 2 or not 1 <= workers <= 4 or (balanced and games_per_family % 4):
        raise ValueError("invalid bounded actor schedule")
    directory.mkdir(parents=True, exist_ok=True)
    frozen = json.loads(openings.read_text())
    metadata = {
        "schema": ORACLE_SCHEMA,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "openings_sha256": sha256(openings),
        "weights_sha256": sha256(weights),
        "stockfish_sha256": sha256(stockfish),
        "seed": frozen["seed"] if seed is None else seed,
        "workers": workers,
        "label_nodes": 32768,
        "actor_nodes": 4096,
        "max_plies": 160,
        "policy_temperature_cp": 100,
        "target_semantics": "engine-reference",
        "games_per_family": games_per_family,
        "collection_plan": "balanced-every-ply-v2" if balanced else "original-parity-v1",
    }
    manifest = directory / "metadata.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != metadata:
            raise ValueError("generation resume requires identical inputs, code and configuration")
    else:
        publish_json(manifest, metadata)
    jobs = []
    for split in ("train", "validation"):
        for opening in frozen["splits"][split]:
            for game in range(games_per_family):
                family = opening["family"]
                path = directory / f"{split}-{family:03d}-{game}.json.gz"
                job_seed = int.from_bytes(
                    hashlib.sha256(f"{metadata['seed']}:{split}:{family}:{game}".encode()).digest()[
                        :8
                    ]
                )
                job = dict(
                    split=split,
                    family=family,
                    game=game,
                    seed=job_seed,
                    actor="engine-engine"
                    if (game % 4 < 2 if balanced else game % 2 == 0)
                    else "neural-engine",
                    neural_color=(family + (game // 4 if balanced else 0)) % 2 == 0,
                    opening=opening["opening"]["moves"],
                    weights=str(weights.resolve()),
                    stockfish=str(stockfish.resolve()),
                    output=str(path.resolve()),
                    deadline=time.time() + wall_seconds,
                    label_every_ply=balanced,
                )
                if path.exists():
                    previous = read_game(path)
                    if previous["job"] != {k: v for k, v in job.items() if k != "deadline"}:
                        raise ValueError("completed game provenance differs")
                else:
                    jobs.append(job)
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = {pool.submit(generate_game, job): job for job in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            job = futures[future]
            print(
                json.dumps(
                    {
                        "completed": i,
                        "pending_total": len(jobs),
                        "split": job["split"],
                        "family": job["family"],
                        **result,
                    }
                ),
                flush=True,
            )
    paths = sorted(directory.glob("*.json.gz"))
    result = {
        "schema": ORACLE_SCHEMA,
        "status": "completed",
        "games": len(paths),
        "rows": sum(len(read_game(path)["rows"]) for path in paths),
        "files": {p.name: sha256(p) for p in paths},
        "wall_seconds_this_invocation": time.perf_counter() - started,
        "children_cpu_seconds": resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime
        + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,
    }
    publish_json(directory / "dataset.json", result)
    return result


def validate_row(row: dict) -> chess.Board:
    board = chess.Board(row["root_fen"])
    for move in row["moves"]:
        board.push_uci(move)
    if board.fen() != row["fen"] or board.outcome(claim_draw=True) is not None:
        raise ValueError("oracle history/FEN/terminal mismatch")
    legal = {move.uci(): move_to_action(board, move) for move in board.legal_moves}
    if dict(row["legal"]) != legal or len(set(legal.values())) != len(legal):
        raise ValueError("oracle legal action mismatch")
    for key, masses in (("wdl", row["wdl"]), ("policy", [p[2] for p in row["policy"]])):
        if (
            not masses
            or any(not math.isfinite(p) or p < 0 for p in masses)
            or abs(sum(masses) - 1) > 1e-6
        ):
            raise ValueError(f"invalid oracle {key} probability distribution")
    if len(row["wdl"]) != 3 or len({p[0] for p in row["policy"]}) != len(row["policy"]):
        raise ValueError("invalid oracle target shape or duplicate policy")
    if any(legal.get(move) != action for move, action, _ in row["policy"]):
        raise ValueError("policy target outside legal support")
    return board


def prepare_rows(rows: list[dict]) -> TorchTrainingBatch:
    if not rows:
        raise ValueError("empty oracle learning panel")
    inputs = torch.empty((len(rows), 8, 8, 104))
    policies = torch.zeros((len(rows), 4672))
    masks = torch.zeros((len(rows), 4672), dtype=torch.bool)
    wdl = torch.empty((len(rows), 3))
    encoder = BoardEncoder()
    for i, row in enumerate(rows):
        board = validate_row(row)
        state = ChessState(row["root_fen"], tuple(ChessMove(m) for m in row["moves"]))
        inputs[i] = torch.tensor(encoder.encode_state(state, board).values).reshape(8, 8, 104)
        for _, action in row["legal"]:
            masks[i, action] = True
        for _, action, probability in row["policy"]:
            policies[i, action] = probability
        wdl[i] = torch.tensor(row["wdl"])
    return TorchTrainingBatch(inputs, policies, masks, wdl, torch.ones(len(rows)))


def load_panels(
    directory: Path,
    *,
    max_train_rows: int | None = None,
    max_validation_rows: int | None = None,
    seed: int = 20261003,
) -> tuple[TorchTrainingBatch, TorchTrainingBatch, dict]:
    manifest = json.loads((directory / "dataset.json").read_text())
    rows: dict[str, list] = {"train": [], "validation": []}
    for filename, digest in manifest["files"].items():
        path = directory / filename
        if sha256(path) != digest:
            raise ValueError(f"oracle dataset checksum mismatch: {filename}")
        game = read_game(path)
        for row in game["rows"]:
            if row["split"] != game["job"]["split"] or row["family"] != game["job"]["family"]:
                raise ValueError("oracle split/family provenance mismatch")
        rows[game["job"]["split"]].extend(game["rows"])
    training_keys = {row["position_key"] for row in rows["train"]}
    initial_validation = len(rows["validation"])
    rows["validation"] = [r for r in rows["validation"] if r["position_key"] not in training_keys]
    overlap = initial_validation - len(rows["validation"])
    available = {split: len(panel) for split, panel in rows.items()}
    for split, limit in (("train", max_train_rows), ("validation", max_validation_rows)):
        if limit is not None:
            if limit <= 0:
                raise ValueError("learning panel row caps must be positive")
            if len(rows[split]) > limit:
                rng = random.Random(f"{seed}:{split}")
                selected = sorted(rng.sample(range(len(rows[split])), limit))
                rows[split] = [rows[split][i] for i in selected]
    summary = {
        "train_rows": len(rows["train"]),
        "validation_rows": len(rows["validation"]),
        "removed_validation_position_overlap": overlap,
        "train_families": sorted({r["family"] for r in rows["train"]}),
        "validation_families": sorted({r["family"] for r in rows["validation"]}),
        "dataset_sha256": sha256(directory / "dataset.json"),
        "available_rows": available,
    }
    return prepare_rows(rows["train"]), prepare_rows(rows["validation"]), summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--openings", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--stockfish", required=True, type=Path)
    parser.add_argument("--wall-seconds", type=float, default=1800)
    parser.add_argument("--games-per-family", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--balanced", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            generate(
                args.directory,
                args.openings,
                args.weights,
                args.stockfish,
                wall_seconds=args.wall_seconds,
                games_per_family=args.games_per_family,
                workers=args.workers,
                seed=args.seed,
                balanced=args.balanced,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

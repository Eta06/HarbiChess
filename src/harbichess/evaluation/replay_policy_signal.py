"""Budgeted all-legal reference audit of the actual stored self-search target."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import resource
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import torch

from harbichess.backends.torch_network import sha256
from harbichess.chess.actions import move_to_action
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove, ChessState
from harbichess.evaluation.teacher_probe import StockfishReference
from harbichess.replay.shard import read_shard


def freeze(directory: Path, rules: PythonChessRules) -> tuple[list[dict], dict]:
    games = defaultdict(list)
    sources = {}
    opening_config = json.loads(
        (directory / "checkpoints/generation-000003/checkpoint.json").read_text()
    )["run_config"]
    if opening_config["opening_source_sha256"] != (
        "019c5b4a49ac287ca03effb65b33ed97daa2e702ddd4a6972491969e27912b24"
    ):
        raise ValueError("frozen actor opening book mismatch")
    for path in sorted((directory / "replay").glob("*.gz")):
        shard = read_shard(path, rules=rules)
        weights = (
            directory
            / f"checkpoints/generation-{shard.header.generation - 1:06d}/model.safetensors"
        )
        if sha256(weights) != shard.header.source_checkpoint:
            raise ValueError("replay actor snapshot checksum mismatch")
        sources[path.name] = {
            "sha256": sha256(path),
            "snapshot_sha256": shard.header.source_checkpoint,
        }
        for record in shard.records:
            if list(record.moves[:8]) != opening_config["actor_openings"][record.game_index % 48]:
                raise ValueError("actual replay opening differs from frozen input")
            games[record.game_index].append((record, path.name))
    if len(games) != 96:
        raise ValueError("source game count differs from preregistration")
    rng, panel = random.Random(20261011), []
    for family in range(48):
        for index, game in enumerate((family, family + 48)):
            turn = "white" if index == 0 else "black"
            eligible = [(r, name) for r, name in games[game] if r.side_to_move == turn]
            if not eligible:
                raise ValueError("missing source side in frozen game")
            record, filename = rng.choice(eligible)
            board = rules.board(record.state)
            legal = {move_to_action(board, m): m.uci() for m in board.legal_moves}
            if not record.raw_policy:
                raise ValueError("stored raw policy missing; do not rerun teacher")
            raw = {legal[a]: p for a, p in record.raw_policy}
            target = {legal[a]: p for a, p in record.policy}
            if set(raw) != set(target) or set(raw) != set(legal.values()):
                raise ValueError("stored probability action coverage mismatch")
            panel.append(
                {
                    "family": family,
                    "game": game,
                    "source_file": filename,
                    "root_fen": record.root_fen,
                    "moves": list(record.moves),
                    "fen": board.fen(),
                    "raw": raw,
                    "target": target,
                    "selected_action": legal[record.selected_action],
                    "observed_outcome": record.outcome_value,
                    "stored_kl": record.teacher_policy_kl,
                    "record_sha256": hashlib.sha256(
                        json.dumps(record.to_dict(), sort_keys=True).encode()
                    ).hexdigest(),
                }
            )
    return panel, sources


def run(directory: Path, stockfish: Path, output: Path, wall_seconds: float) -> dict:
    torch.set_num_threads(1)
    started = time.perf_counter()
    rules = PythonChessRules()
    panel, sources = freeze(directory, rules)
    output.mkdir(parents=True, exist_ok=False)
    (output / "panel.json").write_text(json.dumps(panel, indent=2) + "\n")
    metadata = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "collector_sha256": sha256(Path(__file__)),
        "source_shards": sources,
        "panel_sha256": sha256(output / "panel.json"),
        "stockfish_sha256": sha256(stockfish),
        "seed": 20261011,
        "wall_budget_seconds": wall_seconds,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows, status, reason = [], "completed", "all96 stored targets measured"
    with (
        (output / "queries.jsonl").open("x") as queries,
        (output / "positions.jsonl").open("x") as positions,
    ):
        reference = StockfishReference(stockfish, rules, started + wall_seconds, queries)
        try:
            for item in panel:
                state = ChessState(item["root_fen"], tuple(ChessMove(m) for m in item["moves"]))
                scores = {
                    m: reference.analyse(state, 32768, ChessMove(m)) for m in sorted(item["raw"])
                }
                values = {
                    arm: sum(p * scores[m]["expected_score"] for m, p in item[arm].items())
                    for arm in ("raw", "target")
                }
                row = {
                    **item,
                    "candidate_scores": scores,
                    "expected_score": values,
                    "effect": values["target"] - values["raw"],
                }
                rows.append(row)
                positions.write(json.dumps(row) + "\n")
                positions.flush()
                queries.flush()
                if len(rows) % 16 == 0:
                    print(
                        json.dumps(
                            {"positions": len(rows), "elapsed": time.perf_counter() - started}
                        ),
                        flush=True,
                    )
        except Exception as error:
            status, reason = "incomplete", f"{type(error).__name__}: {error}"
        finally:
            reference.close()
    effects = [sum(r["effect"] for r in rows if r["family"] == f) / 2 for f in range(48)]
    mean = sum(effects) / 48
    rng = random.Random(20261011)
    samples = sorted(sum(rng.choices(effects, k=48)) / 48 for _ in range(10000))
    radius = math.sqrt(2 * math.log(40) / 48)
    report = {
        **metadata,
        "status": status,
        "reason": reason,
        "positions_completed": len(rows),
        "mean_expected_score_effect": mean if len(rows) == 96 else None,
        "family_effects": effects,
        "conditional_bootstrap_95": [samples[249], samples[9749]] if len(rows) == 96 else None,
        "hoeffding_95": [max(-1, mean - radius), min(1, mean + radius)]
        if len(rows) == 96
        else None,
        "mechanism_gate_passed": len(rows) == 96 and mean >= 0.01 and samples[249] > 0,
        "reference_queries": reference.calls,
        "reference_actual_nodes": reference.nodes,
        "wall_seconds": time.perf_counter() - started,
        "parent_cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime
        + resource.getrusage(resource.RUSAGE_SELF).ru_stime,
        "parent_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "scope": (
            "Actual stored self-play targets/training families/native finite-budget reference, "
            "not new played strength or independent families."
        ),
    }
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("directory", "stockfish", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--wall-seconds", type=float, default=1800)
    args = parser.parse_args()
    if args.wall_seconds <= 0:
        parser.error("positive wall budget required")
    result = run(args.directory, args.stockfish, args.output, args.wall_seconds)
    print(json.dumps(result))
    if result["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

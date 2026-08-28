"""Segment fresh search-Q instability without changing teacher thresholds."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import chess

from harbichess.chess.actions import action_to_legal_move
from harbichess.chess.rules import PythonChessRules
from harbichess.evaluation.consensus_target import _record_index
from harbichess.evaluation.teacher_qualification import _atomic_json, _source_commit
from harbichess.replay.coverage import _material_balance, _phase, _value_bucket
from harbichess.replay.shard import read_shard


@dataclass(frozen=True, slots=True)
class TeacherInstabilityConfig:
    dataset_result: Path
    label_result: Path
    validation_shard: Path
    output_dir: Path
    minimum_segment_positions: int = 5

    def __post_init__(self) -> None:
        if self.minimum_segment_positions <= 0:
            raise ValueError("teacher instability segment size must be positive")


def _branching(count: int) -> str:
    if count <= 20:
        return "low"
    if count <= 35:
        return "medium"
    return "high"


def _tactical(board: chess.Board, selected_action: int) -> str:
    move = action_to_legal_move(board, selected_action)
    tactical = (
        board.is_check()
        or board.is_capture(move)
        or move.promotion is not None
        or board.gives_check(move)
    )
    return "tactical" if tactical else "quiet"


def _row(
    raw: Mapping[str, object],
    label: Mapping[str, object],
    *,
    board: chess.Board,
    root_value: float,
    selected_action: int,
) -> dict[str, object]:
    stable = [entry for entry in label["labels"] if float(entry[3]) > 0]
    q_values = [float(entry[1]) for entry in stable]
    drifts = [float(entry[2]) for entry in stable]
    return {
        "game_id": raw["game_id"],
        "game_index": raw["game_index"],
        "ply": raw["ply"],
        "phase": _phase(int(raw["ply"])),
        "branching": _branching(len(raw["verified_values"])),
        "tacticality": _tactical(board, selected_action),
        "value_bucket": _value_bucket(root_value),
        "material_balance": _material_balance(board),
        "stable_q_verified_spearman": float(label["stable_q_verified_spearman"]),
        "high_q_verified_spearman": float(raw["high_q_verified_spearman"]),
        "cross_budget_q_spearman": float(raw["cross_budget_q_spearman"]),
        "cross_budget_q_drift": float(raw["cross_budget_q_drift"]),
        "stable_visit_mass": float(label["stable_visit_mass"]),
        "stable_actions": len(stable),
        "stable_q_spread": max(q_values) - min(q_values),
        "mean_stable_drift": mean(drifts),
    }


def _segment(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    stable_rhos = [float(row["stable_q_verified_spearman"]) for row in rows]
    return {
        "positions": len(rows),
        "mean_stable_q_verified_spearman": mean(stable_rhos),
        "below_gate_ratio": sum(value < 0.35 for value in stable_rhos) / len(rows),
        "negative_rho_ratio": sum(value < 0 for value in stable_rhos) / len(rows),
        "mean_high_q_verified_spearman": mean(
            float(row["high_q_verified_spearman"]) for row in rows
        ),
        "mean_cross_budget_q_spearman": mean(
            float(row["cross_budget_q_spearman"]) for row in rows
        ),
        "mean_cross_budget_q_drift": mean(
            float(row["cross_budget_q_drift"]) for row in rows
        ),
        "mean_stable_visit_mass": mean(float(row["stable_visit_mass"]) for row in rows),
        "mean_stable_actions": mean(float(row["stable_actions"]) for row in rows),
        "mean_stable_q_spread": mean(float(row["stable_q_spread"]) for row in rows),
        "mean_stable_drift": mean(float(row["mean_stable_drift"]) for row in rows),
    }


def run_teacher_instability(config: TeacherInstabilityConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"teacher instability output exists: {config.output_dir}")
    dataset = json.loads(config.dataset_result.read_text(encoding="utf-8"))
    labels = json.loads(config.label_result.read_text(encoding="utf-8"))
    rules = PythonChessRules()
    records = _record_index(read_shard(config.validation_shard, rules=rules).records)
    raw_by_key = {
        (str(row["game_id"]), int(row["game_index"]), int(row["ply"])): row
        for row in dataset["rows"]["validation"]
    }
    rows = []
    for label in labels["rows"]["validation"]:
        key = (str(label["game_id"]), int(label["game_index"]), int(label["ply"]))
        raw = raw_by_key.get(key)
        record = records.get(key)
        if raw is None or record is None:
            raise ValueError(f"teacher instability row is absent: {key}")
        rows.append(
            _row(
                raw,
                label,
                board=rules.board(record.state),
                root_value=record.root_value,
                selected_action=record.selected_action,
            )
        )

    dimensions = ("phase", "branching", "tacticality", "value_bucket", "material_balance")
    segments = {}
    worst = {}
    for dimension in dimensions:
        grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for row in rows:
            grouped[str(row[dimension])].append(row)
        summaries = {
            name: _segment(group)
            for name, group in sorted(grouped.items())
            if len(group) >= config.minimum_segment_positions
        }
        segments[dimension] = summaries
        worst[dimension] = min(
            summaries,
            key=lambda name: float(summaries[name]["mean_stable_q_verified_spearman"]),
        )

    result_path = config.output_dir / "audit.json"
    _atomic_json(
        result_path,
        {
            "source_commit": _source_commit(),
            "config": {
                **asdict(config),
                "dataset_result": str(config.dataset_result),
                "label_result": str(config.label_result),
                "validation_shard": str(config.validation_shard),
                "output_dir": str(config.output_dir),
            },
            "overall": _segment(rows),
            "segments": segments,
            "worst_segments": worst,
            "rows": rows,
            "training_authorized": False,
            "search_qualification_authorized": False,
            "arena_authorized": False,
            "generation_authorized": False,
            "promotion_authorized": False,
        },
    )
    return result_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-result", required=True, type=Path)
    parser.add_argument("--label-result", required=True, type=Path)
    parser.add_argument("--validation-shard", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args(argv)
    result = run_teacher_instability(
        TeacherInstabilityConfig(
            dataset_result=arguments.dataset_result,
            label_result=arguments.label_result,
            validation_shard=arguments.validation_shard,
            output_dir=arguments.output_dir,
        )
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

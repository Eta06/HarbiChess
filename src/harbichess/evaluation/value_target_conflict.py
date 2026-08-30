"""Audit historical/fresh replay for state overlap and contradictory WDL outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from array import array
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from harbichess.chess.encoding import BoardEncoder
from harbichess.chess.rules import PythonChessRules
from harbichess.evaluation.corrected_replay_value_transfer import (
    CorrectedReplayValueTransferConfig,
    _load_games,
)
from harbichess.evaluation.teacher_qualification import _atomic_json
from harbichess.replay.schema import ReplayRecord
from harbichess.training.stable_plastic_ablation import (
    StablePlasticAblationConfig,
    _load_fresh_records,
)


@dataclass(frozen=True, slots=True)
class ValueTargetConflictConfig:
    output_dir: Path
    model_path: Path
    source_continuous_result: Path
    value_result: Path
    runs_root: Path = Path("artifacts/runs")


def _entropy(counts: Counter[int]) -> float:
    total = sum(counts.values())
    return -sum(
        count / total * math.log2(count / total)
        for count in counts.values()
        if count
    )


def _summarize_keyed(
    historical: Sequence[ReplayRecord],
    fresh: Sequence[ReplayRecord],
    key: Callable[[ReplayRecord], str],
) -> dict[str, object]:
    groups: dict[str, dict[str, Counter[int]]] = defaultdict(
        lambda: {"historical": Counter(), "fresh": Counter()}
    )
    for label, records in (("historical", historical), ("fresh", fresh)):
        for record in records:
            if record.outcome_value is not None:
                groups[key(record)][label][int(record.outcome_value)] += 1
    overlapping = [group for group in groups.values() if group["historical"] and group["fresh"]]
    conflicted = [
        group
        for group in overlapping
        if set(group["historical"]) != set(group["fresh"])
        or len(set(group["historical"]) | set(group["fresh"])) > 1
    ]
    any_conflict = [
        group
        for group in groups.values()
        if len(set(group["historical"]) | set(group["fresh"])) > 1
    ]
    return {
        "unique_states": len(groups),
        "overlapping_states": len(overlapping),
        "overlap_rows": sum(
            sum(group["historical"].values()) + sum(group["fresh"].values())
            for group in overlapping
        ),
        "cross_domain_conflicted_states": len(conflicted),
        "all_conflicted_states": len(any_conflict),
        "mean_conflict_entropy_bits": (
            sum(_entropy(group["historical"] + group["fresh"]) for group in conflicted)
            / len(conflicted)
            if conflicted
            else 0.0
        ),
        "examples": [
            {
                "historical": dict(sorted(group["historical"].items())),
                "fresh": dict(sorted(group["fresh"].items())),
            }
            for group in conflicted[:16]
        ],
    }


def _distribution(records: Sequence[ReplayRecord]) -> dict[str, object]:
    outcomes = Counter(int(row.outcome_value) for row in records if row.outcome_value is not None)
    phases = Counter(
        "opening" if row.ply < 20 else "middlegame" if row.ply < 80 else "endgame"
        for row in records
        if row.outcome_value is not None
    )
    return {
        "rows": sum(outcomes.values()),
        "games": len({row.game_id for row in records if row.outcome_value is not None}),
        "outcomes": dict(sorted(outcomes.items())),
        "phases": dict(sorted(phases.items())),
    }


def run_value_target_conflict_audit(config: ValueTargetConflictConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"value target audit output exists: {config.output_dir}")
    config.output_dir.mkdir(parents=True)
    pool_config = CorrectedReplayValueTransferConfig(
        output_dir=config.output_dir,
        model_path=config.model_path,
        runs_root=config.runs_root,
    )
    games, provenance = _load_games(pool_config)
    historical = tuple(row for game in games.values() for row in game)
    stable_config = StablePlasticAblationConfig(
        output_dir=config.output_dir,
        value_result=config.value_result,
        model_path=config.model_path,
        source_continuous_result=config.source_continuous_result,
        runs_root=config.runs_root,
    )
    fresh, replay_paths = _load_fresh_records(stable_config)
    rules = PythonChessRules()
    encoder = BoardEncoder(rules, cache_size=256)

    def trajectory_key(record: ReplayRecord) -> str:
        payload = json.dumps(
            (record.root_fen, record.moves), separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def board_key(record: ReplayRecord) -> str:
        fields = rules.board(record.state).fen(en_passant="fen").split()
        return " ".join(fields[:5])

    def encoded_key(record: ReplayRecord) -> str:
        values = array("f", encoder.encode(record.state).values)
        return hashlib.sha256(values.tobytes()).hexdigest()

    result_path = config.output_dir / "result.json"
    _atomic_json(
        result_path,
        {
            "created_at": datetime.now(UTC).isoformat(),
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "config": {
                **asdict(config),
                **{
                    key: str(getattr(config, key))
                    for key in (
                        "output_dir",
                        "model_path",
                        "source_continuous_result",
                        "value_result",
                        "runs_root",
                    )
                },
            },
            "provenance": provenance,
            "fresh_replay_paths": replay_paths,
            "distribution": {
                "historical": _distribution(historical),
                "fresh": _distribution(fresh),
            },
            "identity_levels": {
                "trajectory": _summarize_keyed(historical, fresh, trajectory_key),
                "current_board": _summarize_keyed(historical, fresh, board_key),
                "encoded_history": _summarize_keyed(historical, fresh, encoded_key),
            },
        },
    )
    return result_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--source-continuous-result", required=True, type=Path)
    parser.add_argument("--value-result", required=True, type=Path)
    parser.add_argument("--runs-root", type=Path, default=Path("artifacts/runs"))
    arguments = parser.parse_args(argv)
    result = run_value_target_conflict_audit(
        ValueTargetConflictConfig(
            output_dir=arguments.output_dir,
            model_path=arguments.model,
            source_continuous_result=arguments.source_continuous_result,
            value_result=arguments.value_result,
            runs_root=arguments.runs_root,
        )
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

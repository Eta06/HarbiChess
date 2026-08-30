"""Deduplicated, game-disjoint WDL transfer over compatible corrected replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import Side
from harbichess.dashboard.state import PilotStatus, RunMode, SnapshotStore
from harbichess.evaluation.teacher_qualification import _atomic_json
from harbichess.evaluation.value_signal_audit import (
    ValueSignalAuditConfig,
    _describe,
    _train_arm,
)
from harbichess.replay.schema import ReplayRecord
from harbichess.replay.shard import read_shard

DEFAULT_RUNS = (
    "kopru-clean-target-sanity-20260828-01",
    "kopru-dual-search-sanity-20260828-01",
    "kopru-fresh-replay-20260828-01",
    "kopru-qualified-replay-20260828-01",
    "kopru-selfplay-perf-w16-20260828-01",
    "kopru-selfplay-perf-w24-20260828-01",
    "omurga-qualified-sanity-20260827-01",
    "omurga-qualified-sanity-20260827-02",
)


@dataclass(frozen=True, slots=True)
class CorrectedReplayValueTransferConfig:
    output_dir: Path
    model_path: Path
    runs_root: Path = Path("artifacts/runs")
    telemetry_path: Path = Path("artifacts/dashboard/state.json")
    run_ids: tuple[str, ...] = DEFAULT_RUNS
    expected_model_sha256: str = (
        "5a094285413d52e663441926afe09aa6efda3aa5e5ec195919b54c159d830eca"
    )
    learning_rate: float = 5e-4
    batch_size: int = 64
    steps: int = 400
    validation_interval: int = 20
    seed: int = 2026083049

    def __post_init__(self) -> None:
        if not self.run_ids or len(self.run_ids) != len(set(self.run_ids)):
            raise ValueError("corrected replay run ids must be unique and non-empty")
        if (
            min(self.batch_size, self.steps, self.validation_interval, self.seed) <= 0
            or self.steps % self.validation_interval
            or self.learning_rate <= 0
            or len(self.expected_model_sha256) != 64
        ):
            raise ValueError("corrected replay value transfer configuration is invalid")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trajectory_fingerprint(records: Sequence[ReplayRecord]) -> str:
    ordered = sorted(records, key=lambda row: row.ply)
    if not ordered or tuple(row.ply for row in ordered) != tuple(range(len(ordered))):
        raise ValueError("replay game must contain a contiguous trajectory")
    identity = (ordered[0].root_fen, ordered[-1].moves, ordered[-1].selected_action)
    return hashlib.sha256(
        json.dumps(identity, separators=(",", ":")).encode()
    ).hexdigest()


def _white_outcome(records: Sequence[ReplayRecord]) -> int | None:
    values = {record.outcome_value for record in records}
    if values == {None}:
        return None
    if values == {0}:
        return 0
    if None in values or 0 in values or not values <= {-1, 1}:
        raise ValueError("trajectory contains inconsistent value targets")
    white_values = {
        int(record.outcome_value)
        * (1 if record.side_to_move is Side.WHITE else -1)
        for record in records
    }
    if len(white_values) != 1:
        raise ValueError("trajectory value perspective does not alternate with side to move")
    return white_values.pop()


def _load_games(
    config: CorrectedReplayValueTransferConfig,
) -> tuple[dict[str, tuple[ReplayRecord, ...]], dict[str, object]]:
    rules = PythonChessRules()
    unique: dict[str, tuple[ReplayRecord, ...]] = {}
    sources: dict[str, list[str]] = defaultdict(list)
    named_games = known_named_games = unknown_named_games = rows = 0
    schemas: dict[str, list[int]] = {}
    for run_id in config.run_ids:
        result_path = config.runs_root / run_id / "result.json"
        result = json.loads(result_path.read_text())
        baseline_sha = result.get("baseline", {}).get("model_sha256") or result.get(
            "baseline_model_sha256"
        )
        if baseline_sha != config.expected_model_sha256:
            raise ValueError(f"{run_id} was not produced by the frozen baseline")
        schemas[run_id] = []
        for shard_path in sorted((config.runs_root / run_id / "replay").glob("*.jsonl.gz")):
            shard = read_shard(shard_path, rules=rules)
            if shard.header.target_schema < 10:
                raise ValueError(f"{shard_path} predates corrected max-ply targets")
            schemas[run_id].append(shard.header.target_schema)
            grouped: dict[str, list[ReplayRecord]] = defaultdict(list)
            for record in shard.records:
                grouped[record.game_id].append(record)
            for game_records in grouped.values():
                named_games += 1
                rows += len(game_records)
                outcome = _white_outcome(game_records)
                if outcome is None:
                    unknown_named_games += 1
                else:
                    known_named_games += 1
                fingerprint = _trajectory_fingerprint(game_records)
                sources[fingerprint].append(run_id)
                if fingerprint not in unique:
                    unique[fingerprint] = tuple(sorted(game_records, key=lambda row: row.ply))
                elif _white_outcome(unique[fingerprint]) != outcome:
                    raise ValueError("duplicate trajectory has conflicting outcomes")
    known = {
        fingerprint: game
        for fingerprint, game in unique.items()
        if _white_outcome(game) is not None
    }
    provenance = {
        "run_ids": config.run_ids,
        "target_schemas": schemas,
        "named_games": named_games,
        "known_named_games": known_named_games,
        "unknown_named_games": unknown_named_games,
        "rows": rows,
        "unique_trajectories": len(unique),
        "unique_known_trajectories": len(known),
        "duplicate_named_games": named_games - len(unique),
        "duplicate_fingerprints": sum(len(run_ids) > 1 for run_ids in sources.values()),
        "unique_known_outcomes_white": dict(
            Counter(str(_white_outcome(game)) for game in known.values())
        ),
    }
    return known, provenance


def _split_games(
    games: dict[str, tuple[ReplayRecord, ...]], *, seed: int
) -> tuple[tuple[ReplayRecord, ...], tuple[ReplayRecord, ...], dict[str, object]]:
    grouped: dict[int, list[str]] = defaultdict(list)
    for fingerprint, records in games.items():
        outcome = _white_outcome(records)
        assert outcome is not None
        grouped[outcome].append(fingerprint)
    rng = random.Random(seed)
    train_fingerprints: set[str] = set()
    validation_fingerprints: set[str] = set()
    for outcome in (-1, 0, 1):
        fingerprints = sorted(grouped[outcome])
        rng.shuffle(fingerprints)
        boundary = round(len(fingerprints) * 0.75)
        train_fingerprints.update(fingerprints[:boundary])
        validation_fingerprints.update(fingerprints[boundary:])
    if train_fingerprints & validation_fingerprints:
        raise AssertionError("trajectory fingerprint leaked across split")

    def rows(selected: set[str]) -> tuple[ReplayRecord, ...]:
        return tuple(
            replace(record, game_id=fingerprint)
            for fingerprint in sorted(selected)
            for record in games[fingerprint]
        )

    return rows(train_fingerprints), rows(validation_fingerprints), {
        "train_trajectories": len(train_fingerprints),
        "validation_trajectories": len(validation_fingerprints),
        "fingerprint_overlap": 0,
        "train_outcomes_white": dict(
            Counter(str(_white_outcome(games[key])) for key in train_fingerprints)
        ),
        "validation_outcomes_white": dict(
            Counter(str(_white_outcome(games[key])) for key in validation_fingerprints)
        ),
    }


def run_corrected_replay_value_transfer(config: CorrectedReplayValueTransferConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"corrected replay output exists: {config.output_dir}")
    if _sha256(config.model_path) != config.expected_model_sha256:
        raise ValueError("corrected replay baseline checksum mismatch")
    games, provenance = _load_games(config)
    train_records, validation_records, split = _split_games(games, seed=config.seed)
    config.output_dir.mkdir(parents=True)
    store = SnapshotStore(config.telemetry_path)
    snapshot = replace(
        store.read(),
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.TRAINING,
        mode_detail="KRITIK deduplicated corrected replay WDL transfer",
        run_id=config.output_dir.name,
        pilot_status=PilotStatus.TRAINING,
        pilot_steps_planned=config.steps,
        pilot_steps_completed=0,
    )
    store.write_atomic(snapshot)
    started = time.perf_counter()
    audit_config = ValueSignalAuditConfig(
        output_dir=config.output_dir,
        model_path=config.model_path,
        train_shard=Path("unused-train"),
        validation_shard=Path("unused-validation"),
        telemetry_path=config.telemetry_path,
        expected_model_sha256=config.expected_model_sha256,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        steps=config.steps,
        validation_interval=config.validation_interval,
        seed=config.seed,
    )
    arm, snapshot = _train_arm(
        "deduplicated-corrected-replay",
        train_records,
        validation_records,
        config=audit_config,
        store=store,
        snapshot=snapshot,
        arm_index=0,
    )
    verdict = "passed" if arm["passed"] else (
        "partial_improvement" if int(arm["selected_step"]) > 0 else "no_generalization"
    )
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
                    for key in ("output_dir", "model_path", "runs_root", "telemetry_path")
                },
            },
            "provenance": provenance,
            "split": split,
            "descriptive": {
                "train": _describe(train_records, PythonChessRules()),
                "validation": _describe(validation_records, PythonChessRules()),
            },
            "arm": arm,
            "verdict": verdict,
            "continuous_learning_authorized": False,
            "generation_authorized": False,
            "promotion_authorized": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    detail = f"KRITIK corrected replay WDL · {verdict}"
    snapshot = replace(
        snapshot,
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.IDLE,
        mode_detail=detail,
        pilot_status=PilotStatus.PASSED if arm["passed"] else PilotStatus.FAILED,
        pilot_stop_reason="corrected_replay_value_gate",
        pilot_stop_detail=detail,
        pilot_reasons=tuple(arm["reasons"]),
        promotion_ready=False,
    )
    store.write_atomic(snapshot)
    return result_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--runs-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--telemetry", type=Path, default=Path("artifacts/dashboard/state.json"))
    arguments = parser.parse_args(argv)
    print(
        run_corrected_replay_value_transfer(
            CorrectedReplayValueTransferConfig(
                output_dir=arguments.output_dir,
                model_path=arguments.model,
                runs_root=arguments.runs_root,
                telemetry_path=arguments.telemetry,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

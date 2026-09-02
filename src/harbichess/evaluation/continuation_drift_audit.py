"""Audit cumulative continuation drift without recomputing the tactical oracle per model."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

import mlx.core as mx

from harbichess.chess.encoding import BoardEncoder
from harbichess.chess.rules import PythonChessRules
from harbichess.replay.shard import read_shard
from harbichess.search.value_oracle import DeterministicTacticalOracle, TacticalOracleConfig
from harbichess.training.continuous_policy_iteration import (
    ContinuousPolicyIterationConfig,
    _clone,
    _final_continuation_noninferiority_gate,
    _load_initial,
    _paired_mean_interval,
    _select_game_paired_continuation_records,
)
from harbichess.training.joint_policy_value_transfer import _spearman


@dataclass(frozen=True, slots=True)
class ContinuationDriftAuditConfig:
    pilot_result: Path
    output_dir: Path
    positions: int = 1_440
    depth: int = 4
    bootstrap_samples: int = 20_000
    margin: float = 0.020


def _expected_scores(logits: list[list[float]]) -> tuple[float, ...]:
    values = []
    for row in logits:
        maximum = max(row)
        weights = [math.exp(value - maximum) for value in row]
        total = sum(weights)
        values.append(-(weights[0] - weights[2]) / total)
    return tuple(values)


def _load_models(payload: dict[str, object]):
    config = payload["config"]
    base_config = ContinuousPolicyIterationConfig(
        output_dir=Path("unused"),
        value_result=Path(str(config["value_result"])),
        model_path=Path(str(config["model_path"])),
        stable_plastic_value=True,
        final_qualification_games=1,
        old_qualification_games=1,
    )
    initial, _ = _load_initial(base_config)
    models = {"initial": initial}
    for update in payload["updates"]:
        name = f"update-{int(update['update']):03d}"
        calibrated = _clone(initial)
        calibrated.load_weights(str(update["checkpoint_path"]))
        mx.eval(calibrated.parameters())
        uncalibrated = _clone(calibrated)
        uncalibrated.set_value_logit_scale(1.0)
        models[name] = calibrated
        models[f"{name}-scale1"] = uncalibrated
    return models


def _summarize(rows: tuple[dict[str, object], ...]) -> dict[str, object]:
    return {
        "positions": len(rows),
        "candidate_mean_spearman": mean(float(row["candidate_spearman"]) for row in rows),
        "candidate_verified_top_agreement": mean(
            bool(row["candidate_verified_top"]) for row in rows
        ),
        "rows": rows,
        "passed": True,
    }


def run_continuation_drift_audit(config: ContinuationDriftAuditConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"continuation drift output exists: {config.output_dir}")
    payload = json.loads(config.pilot_result.read_text(encoding="utf-8"))
    if payload.get("passed") or not payload.get("continuation_gate"):
        raise ValueError("audit requires a failed game-paired continuation pilot")
    config.output_dir.mkdir(parents=True)
    rules = PythonChessRules()
    shard = read_shard(Path(payload["old_qualification"]["path"]), rules=rules)
    records = _select_game_paired_continuation_records(
        shard.records,
        rules=rules,
        count=config.positions,
        seed=int(payload["config"]["seed"]) + 3500,
    )
    models = _load_models(payload)
    encoder = BoardEncoder(rules)
    oracle = DeterministicTacticalOracle(
        rules=rules,
        config=TacticalOracleConfig(depth=config.depth),
    )
    rows = defaultdict(list)
    try:
        for record in records:
            moves = tuple(rules.legal_moves(record.state))
            children = tuple(rules.apply(record.state, move) for move in moves)
            encoded = tuple(encoder.encode(child) for child in children)
            shape = encoded[0].shape
            inputs = mx.array(
                [position.values for position in encoded], dtype=mx.float32
            ).reshape((len(encoded), *shape))
            logits = {name: model(inputs)[1] for name, model in models.items()}
            mx.eval(list(logits.values()))
            verified = tuple(-oracle.value(child) for child in children)
            best = max(verified)
            verified_best = {
                index for index, value in enumerate(verified) if value == best
            }
            for name, values in logits.items():
                expected = _expected_scores(values.tolist())
                rows[name].append(
                    {
                        "identity": f"{record.game_id}:{record.game_index}:{record.ply}",
                        "game_id": record.game_id,
                        "phase": (
                            "opening"
                            if record.ply < 20
                            else "middlegame"
                            if record.ply < 80
                            else "endgame"
                        ),
                        "legal_actions": len(moves),
                        "candidate_spearman": _spearman(expected, verified),
                        "candidate_verified_top": max(
                            range(len(moves)), key=expected.__getitem__
                        )
                        in verified_best,
                    }
                )
    finally:
        oracle.close()

    summaries = {name: _summarize(tuple(value)) for name, value in rows.items()}
    initial = summaries["initial"]
    comparisons = {}
    for name, summary in summaries.items():
        if name == "initial":
            continue
        summary["passed"] = True
        comparisons[name] = _final_continuation_noninferiority_gate(
            initial,
            summary,
            samples=config.bootstrap_samples,
            seed=int(payload["config"]["seed"]) + 8000 + len(comparisons) * 2,
            margin=config.margin,
        )
        phase_rows = defaultdict(list)
        initial_by_id = {row["identity"]: row for row in initial["rows"]}
        for row in summary["rows"]:
            baseline = initial_by_id[row["identity"]]
            phase_rows[row["phase"]].append(
                float(row["candidate_spearman"])
                - float(baseline["candidate_spearman"])
            )
        comparisons[name]["phase_spearman"] = {
            phase: _paired_mean_interval(
                tuple(values),
                samples=config.bootstrap_samples,
                seed=int(payload["config"]["seed"]) + 9000 + index,
            )
            for index, (phase, values) in enumerate(sorted(phase_rows.items()))
        }

    result_path = config.output_dir / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now(UTC).isoformat(),
                "source_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "config": {
                    **asdict(config),
                    "pilot_result": str(config.pilot_result),
                    "output_dir": str(config.output_dir),
                },
                "model_summaries": {
                    name: {key: value for key, value in summary.items() if key != "rows"}
                    for name, summary in summaries.items()
                },
                "comparisons": comparisons,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-result", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args(argv)
    print(
        run_continuation_drift_audit(
            ContinuationDriftAuditConfig(
                pilot_result=arguments.pilot_result,
                output_dir=arguments.output_dir,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

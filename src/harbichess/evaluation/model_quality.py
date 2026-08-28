"""Independent policy-imitation and WDL-calibration metrics for learner gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean

import mlx.core as mx

from harbichess.backends.mlx_network import HarbiChessNetwork
from harbichess.chess.encoding import BoardEncoder
from harbichess.chess.rules import PythonChessRules
from harbichess.replay.schema import ReplayRecord


@dataclass(frozen=True, slots=True)
class ModelQualityMetrics:
    samples: int
    known_value_samples: int
    teacher_policy_cross_entropy: float
    teacher_top_action_agreement: float
    value_cross_entropy: float
    value_accuracy: float
    expected_score_ece: float
    expected_score_brier: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _expected_score_ece(
    predictions: list[float],
    targets: list[float],
    *,
    bins: int,
) -> float:
    if not predictions:
        return 0.0
    total = len(predictions)
    error = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        selected = [
            item
            for item, prediction in enumerate(predictions)
            if low <= prediction < high or (index == bins - 1 and prediction == 1.0)
        ]
        if selected:
            predicted = mean(predictions[item] for item in selected)
            actual = mean(targets[item] for item in selected)
            error += len(selected) / total * abs(predicted - actual)
    return error


def evaluate_model_quality(
    network: HarbiChessNetwork,
    records: tuple[ReplayRecord, ...],
    *,
    batch_size: int = 256,
    calibration_bins: int = 10,
    rules: PythonChessRules | None = None,
) -> ModelQualityMetrics:
    if not records or batch_size <= 0 or calibration_bins <= 1:
        raise ValueError("model quality requires records, a batch, and calibration bins")
    engine = rules or PythonChessRules()
    encoder = BoardEncoder(engine)
    policy_losses = []
    top_action_matches = []
    value_losses = []
    value_matches = []
    expected_scores = []
    score_targets = []

    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        positions = tuple(encoder.encode(record.state) for record in chunk)
        shape = positions[0].shape
        inputs = mx.array([position.values for position in positions], dtype=mx.float32)
        inputs = inputs.reshape((len(positions), *shape))
        policy_logits, wdl_logits = network(inputs)
        policy_log_probs = policy_logits - mx.logsumexp(policy_logits, axis=1, keepdims=True)
        wdl_log_probs = wdl_logits - mx.logsumexp(wdl_logits, axis=1, keepdims=True)
        wdl_probs = mx.softmax(wdl_logits, axis=1)
        mx.eval(policy_log_probs, wdl_log_probs, wdl_probs)
        policy_rows = policy_log_probs.tolist()
        wdl_log_rows = wdl_log_probs.tolist()
        wdl_rows = wdl_probs.tolist()

        for index, record in enumerate(chunk):
            policy_losses.append(
                -sum(
                    probability * policy_rows[index][action]
                    for action, probability in record.policy
                )
            )
            legal_actions = tuple(action for action, _ in record.raw_policy) or tuple(
                action for action, _ in record.policy
            )
            predicted_action = min(
                legal_actions,
                key=lambda action: (-policy_rows[index][action], action),
            )
            teacher_action = min(record.policy, key=lambda item: (-item[1], item[0]))[0]
            top_action_matches.append(predicted_action == teacher_action)
            if record.outcome_value is None:
                continue
            target_class = {1: 0, 0: 1, -1: 2}[record.outcome_value]
            row = wdl_rows[index]
            value_losses.append(-wdl_log_rows[index][target_class])
            value_matches.append(max(range(3), key=row.__getitem__) == target_class)
            expected_scores.append(row[0] + 0.5 * row[1])
            score_targets.append((record.outcome_value + 1.0) / 2.0)

    return ModelQualityMetrics(
        samples=len(records),
        known_value_samples=len(value_losses),
        teacher_policy_cross_entropy=mean(policy_losses),
        teacher_top_action_agreement=mean(top_action_matches),
        value_cross_entropy=mean(value_losses) if value_losses else 0.0,
        value_accuracy=mean(value_matches) if value_matches else 0.0,
        expected_score_ece=_expected_score_ece(
            expected_scores,
            score_targets,
            bins=calibration_bins,
        ),
        expected_score_brier=(
            mean(
                (prediction - target) ** 2
                for prediction, target in zip(expected_scores, score_targets, strict=True)
            )
            if expected_scores
            else 0.0
        ),
    )

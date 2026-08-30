from pathlib import Path
from types import SimpleNamespace

import pytest
from mlx.utils import tree_flatten

mx = pytest.importorskip("mlx.core")

from harbichess.backends.decoupled_value_network import (  # noqa: E402
    HarbiChessDecoupledValueNetwork,
)
from harbichess.backends.mlx_backend import MLXPolicyValueBackend  # noqa: E402
from harbichess.backends.plastic_value_network import (  # noqa: E402
    PLASTIC_VALUE_PREFIXES,
    HarbiChessPlasticValueNetwork,
)
from harbichess.core.state import Side  # noqa: E402
from harbichess.training.continuous_policy_iteration import (  # noqa: E402
    ContinuousPolicyIterationConfig,
    _clone,
    _combine_policy,
    _compose_headwise_state,
    _continuous_wdl_gate,
    _LearnerState,
    _paired_mean_interval,
    _policy_gate,
    _select_continuation_starts,
    _select_numeric_checkpoint,
    _select_qualification_starts,
    _soft_value_targets,
    _split_fit_tuning,
)
from harbichess.training.full_gumbel_transfer import (  # noqa: E402
    PreparedTransfer,
    _network,
)


def _config(**overrides) -> ContinuousPolicyIterationConfig:
    values = {
        "output_dir": Path("output"),
        "value_result": Path("value.json"),
        "model_path": Path("model.safetensors"),
    }
    values.update(overrides)
    return ContinuousPolicyIterationConfig(**values)


def _wdl(**overrides) -> dict[str, float]:
    values = {
        "cross_entropy": 0.90,
        "macro_cross_entropy": 0.92,
        "expected_score_pearson": 0.45,
        "loss_draw_margin": 0.20,
        "win_draw_margin": 0.20,
        "ece_10": 0.03,
    }
    values.update(overrides)
    return values


def test_config_rejects_impossible_rolling_window() -> None:
    with pytest.raises(ValueError, match="rolling generations"):
        _config(updates=2, rolling_generations=3)


def test_config_defaults_use_scaled_qualified_teacher_set() -> None:
    config = _config()

    assert config.train_targets_per_update == 768
    assert config.validation_targets_per_update == 192
    assert config.steps_per_update == 40
    assert config.batch_size == 64
    assert config.selfplay_games_per_update == 96
    assert config.selfplay_workers == 24
    assert config.minimum_known_selfplay_games == 24


def test_config_requires_an_even_mixed_value_batch() -> None:
    with pytest.raises(ValueError, match="value batch size must be even"):
        _config(batch_size=63)


def test_config_rejects_unreachable_known_game_floor() -> None:
    with pytest.raises(ValueError, match="minimum known games"):
        _config(selfplay_games_per_update=3, minimum_known_selfplay_games=4)


def test_config_requires_equal_opening_middle_endgame_quota() -> None:
    with pytest.raises(ValueError, match="three phases"):
        _config(selfplay_games_per_update=10)


def test_stable_plastic_config_requires_fresh_final_qualification() -> None:
    with pytest.raises(ValueError, match="final qualification"):
        _config(stable_plastic_value=True)


def test_continuation_starts_are_phase_balanced_and_non_overlapping() -> None:
    records = tuple(
        SimpleNamespace(game_id=f"{phase}-{index}", game_index=index, ply=ply)
        for phase, ply in (("opening", 8), ("middle", 40), ("end", 90))
        for index in range(8)
    )

    starts = _select_continuation_starts(records, updates=2, games_per_update=6, seed=7)

    assert tuple(
        (
            sum(record.ply < 20 for record in update),
            sum(20 <= record.ply < 80 for record in update),
            sum(record.ply >= 80 for record in update),
        )
        for update in starts
    ) == ((2, 2, 2), (2, 2, 2))
    identities = {(record.game_id, record.ply) for update in starts for record in update}
    assert len(identities) == 12


def test_policy_gate_requires_imitation_gain_without_top_action_regression() -> None:
    before = {"cross_entropy": 2.0, "top_action_agreement": 0.25}

    assert _policy_gate(before, {"cross_entropy": 1.98, "top_action_agreement": 0.30}) == ()
    assert len(_policy_gate(before, {"cross_entropy": 1.995, "top_action_agreement": 0.20})) == 2


def test_continuous_wdl_gate_keeps_relative_and_absolute_floors() -> None:
    assert _continuous_wdl_gate(_wdl(), _wdl(cross_entropy=0.89)) == ()

    reasons = _continuous_wdl_gate(
        _wdl(),
        _wdl(
            cross_entropy=1.01,
            macro_cross_entropy=1.02,
            expected_score_pearson=0.15,
            loss_draw_margin=0.01,
            ece_10=0.13,
        ),
    )

    assert len(reasons) == 8


def test_rolling_policy_buffer_preserves_generation_order() -> None:
    first = PreparedTransfer(
        records=("first",),
        inputs=mx.array([[1.0]]),
        targets=mx.array([[0.75, 0.25]]),
        legal_masks=mx.array([[True, True]]),
        wdl_targets=(1,),
    )
    second = PreparedTransfer(
        records=("second",),
        inputs=mx.array([[2.0]]),
        targets=mx.array([[0.25, 0.75]]),
        legal_masks=mx.array([[True, True]]),
        wdl_targets=(0,),
    )

    combined = _combine_policy((first, second))

    assert combined.records == ("first", "second")
    assert combined.inputs.tolist() == [[1.0], [2.0]]
    assert combined.targets.tolist() == [[0.75, 0.25], [0.25, 0.75]]
    assert combined.wdl_targets == (1, 0)


def test_inference_clone_does_not_quantize_live_learner() -> None:
    live = HarbiChessDecoupledValueNetwork.from_base(_network())
    before = {name: mx.array(value) for name, value in tree_flatten(live.parameters())}
    mx.eval(list(before.values()))

    inference = _clone(live)
    MLXPolicyValueBackend(inference)
    after = dict(tree_flatten(live.parameters()))

    assert {str(value.dtype) for _, value in tree_flatten(inference.parameters())} == {
        "mlx.core.bfloat16"
    }
    assert {str(value.dtype) for value in after.values()} == {"mlx.core.float32"}
    assert all(
        float(mx.max(mx.abs(after[name] - value)).item()) == 0.0 for name, value in before.items()
    )


def test_checkpoint_selection_uses_earliest_eligible_step() -> None:
    checkpoints = (
        {
            "local_step": 10,
            "policy": {"cross_entropy": 1.8},
            "wdl": {"macro_cross_entropy": 0.9},
            "reasons": (),
        },
        {
            "local_step": 20,
            "policy": {"cross_entropy": 1.7},
            "wdl": {"macro_cross_entropy": 0.91},
            "reasons": (),
        },
        {
            "local_step": 40,
            "policy": {"cross_entropy": 1.6},
            "wdl": {"macro_cross_entropy": 1.1},
            "reasons": ("WDL regression",),
        },
    )

    selected, eligible = _select_numeric_checkpoint(checkpoints)

    assert eligible is True
    assert selected["local_step"] == 10


def test_headwise_checkpoint_composition_keeps_policy_and_value_times_separate() -> None:
    policy_state = _LearnerState(
        step=20,
        weights=(
            ("policy_linear.bias", mx.array([20.0])),
            ("global_value_output.bias", mx.array([200.0])),
            ("stem.bias", mx.array([2.0])),
        ),
        optimizer=(("step", mx.array(20)),),
    )
    value_state = _LearnerState(
        step=10,
        weights=(
            ("policy_linear.bias", mx.array([10.0])),
            ("global_value_output.bias", mx.array([100.0])),
            ("stem.bias", mx.array([1.0])),
        ),
        optimizer=(("step", mx.array(10)),),
    )

    composed = _compose_headwise_state(
        {"state": policy_state},
        {"state": value_state},
    )

    weights = dict(composed.weights)
    assert composed.step == 20
    assert weights["policy_linear.bias"].item() == 20.0
    assert weights["global_value_output.bias"].item() == 100.0
    assert weights["stem.bias"].item() == 2.0


def test_plastic_clone_preserves_network_type_and_outputs() -> None:
    base = HarbiChessDecoupledValueNetwork.from_base(_network())
    network = HarbiChessPlasticValueNetwork.from_mihver(base)
    inputs = mx.zeros((2, 8, 8, network.config.input_channels))
    before = network(inputs)

    cloned = _clone(network)
    after = cloned(inputs)
    mx.eval(*before, *after)

    assert isinstance(cloned, HarbiChessPlasticValueNetwork)
    assert all(
        float(mx.max(mx.abs(left - right)).item()) == 0.0
        for left, right in zip(before, after, strict=True)
    )


def test_plastic_headwise_composition_selects_only_plastic_value_parameters() -> None:
    policy_state = _LearnerState(
        step=20,
        weights=(
            ("policy_linear.bias", mx.array([20.0])),
            ("global_value_output.bias", mx.array([200.0])),
            ("plastic_value_output.bias", mx.array([2000.0])),
        ),
        optimizer=(("step", mx.array(20)),),
    )
    value_state = _LearnerState(
        step=10,
        weights=(
            ("policy_linear.bias", mx.array([10.0])),
            ("global_value_output.bias", mx.array([100.0])),
            ("plastic_value_output.bias", mx.array([1000.0])),
        ),
        optimizer=(("step", mx.array(10)),),
    )

    composed = _compose_headwise_state(
        {"state": policy_state},
        {"state": value_state},
        value_prefixes=PLASTIC_VALUE_PREFIXES,
    )

    weights = dict(composed.weights)
    assert weights["policy_linear.bias"].item() == 20.0
    assert weights["global_value_output.bias"].item() == 200.0
    assert weights["plastic_value_output.bias"].item() == 1000.0


def test_soft_value_targets_preserve_conflicting_outcome_uncertainty() -> None:
    historical = (
        SimpleNamespace(root_fen="same", moves=(1, 2), outcome_value=1),
        SimpleNamespace(root_fen="other", moves=(3,), outcome_value=-1),
    )
    fresh = (
        SimpleNamespace(root_fen="same", moves=(1, 2), outcome_value=0),
    )

    historical_targets, fresh_targets, summary = _soft_value_targets(historical, fresh)

    assert historical_targets.tolist() == [[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]
    assert fresh_targets.tolist() == [[0.5, 0.5, 0.0]]
    assert summary == {
        "unique_fit_states": 2,
        "ambiguous_fit_states": 1,
        "ambiguous_fit_rows": 2,
    }


def test_qualification_starts_are_phase_balanced_and_deterministic() -> None:
    records = tuple(
        SimpleNamespace(game_id=f"{phase}-{index}", game_index=index, ply=ply)
        for phase, ply in (("opening", 8), ("middle", 40), ("end", 90))
        for index in range(5)
    )

    first = _select_qualification_starts(records, count=9, seed=17)
    second = _select_qualification_starts(records, count=9, seed=17)

    assert tuple((row.game_id, row.game_index, row.ply) for row in first) == tuple(
        (row.game_id, row.game_index, row.ply) for row in second
    )
    assert sum(row.ply < 20 for row in first) == 3
    assert sum(20 <= row.ply < 80 for row in first) == 3
    assert sum(row.ply >= 80 for row in first) == 3


def test_fit_tuning_split_is_game_disjoint_and_outcome_stratified() -> None:
    records = tuple(
        SimpleNamespace(
            game_id=f"{outcome}-{game}",
            game_index=game,
            ply=ply,
            outcome_value=outcome if ply % 2 == 0 else -outcome,
            side_to_move=Side.WHITE if ply % 2 == 0 else Side.BLACK,
        )
        for outcome in (-1, 0, 1)
        for game in range(10)
        for ply in range(2)
    )

    fit, tuning, summary = _split_fit_tuning(records, seed=29)

    fit_games = {row.game_id for row in fit}
    tuning_games = {row.game_id for row in tuning}
    assert fit_games.isdisjoint(tuning_games)
    assert len(fit_games) == 24
    assert len(tuning_games) == 6
    assert {int(row.game_id.rsplit("-", 1)[0]) for row in tuning} == {-1, 0, 1}
    assert summary["game_overlap"] == 0


def test_paired_mean_interval_is_reproducible() -> None:
    values = (-0.02, 0.01, 0.03, 0.04)

    first = _paired_mean_interval(values, samples=1000, seed=23)
    second = _paired_mean_interval(values, samples=1000, seed=23)

    assert first == second
    assert first["estimate"] == pytest.approx(0.015)

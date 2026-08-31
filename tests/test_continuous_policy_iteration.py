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
from harbichess.training.continuous_checkpoint import save_continuous_resume  # noqa: E402
from harbichess.training.continuous_policy_iteration import (  # noqa: E402
    _VALUE_TRUST_REGION_ALPHAS,
    ContinuousPolicyIterationConfig,
    _clone,
    _combine_policy,
    _compose_headwise_state,
    _continuous_wdl_gate,
    _ContinuousHeadLearner,
    _empirical_fresh_value_targets,
    _fresh_wdl_calibration_gate,
    _fresh_wdl_direction_gate,
    _fresh_wdl_harm_gate,
    _interpolate_value_state,
    _LearnerState,
    _local_arena_catastrophic_gate,
    _old_wdl_point_noninferiority_gate,
    _old_wdl_statistical_noninferiority_gate,
    _paired_mean_interval,
    _policy_gate,
    _search_tactical_gate,
    _select_continuation_starts,
    _select_continuation_state_starts,
    _select_numeric_checkpoint,
    _select_qualification_starts,
    _select_tactical_policy_checkpoint,
    _select_value_checkpoint,
    _soft_value_targets,
    _split_fit_tuning,
    _stable_value_distillation_targets,
    _verify_resume_exactness,
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
    assert config.historical_value_weight == 2.0
    assert config.final_ranking_positions == 1_440
    assert _VALUE_TRUST_REGION_ALPHAS[-1] == pytest.approx(0.0078125)


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


def test_stable_plastic_config_requires_powered_old_qualification() -> None:
    with pytest.raises(ValueError, match="old qualification"):
        _config(
            stable_plastic_value=True,
            final_qualification_games=3,
            minimum_final_known_games=1,
        )


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


def test_state_continuation_starts_allow_distinct_positions_from_same_game() -> None:
    records = tuple(
        SimpleNamespace(game_id=phase, game_index=0, ply=base_ply + offset)
        for phase, base_ply in (("opening", 0), ("middle", 20), ("end", 80))
        for offset in range(6)
    )

    starts = _select_continuation_state_starts(
        records,
        updates=2,
        games_per_update=6,
        seed=11,
    )

    identities = {
        (record.game_id, record.game_index, record.ply)
        for update in starts
        for record in update
    }
    assert len(identities) == 12
    assert all(
        (
            sum(record.ply < 20 for record in update),
            sum(20 <= record.ply < 80 for record in update),
            sum(record.ply >= 80 for record in update),
        )
        == (2, 2, 2)
        for update in starts
    )


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


def test_cumulative_pilot_local_wdl_gate_uses_its_own_tuning_baseline() -> None:
    baseline = _wdl(cross_entropy=0.95, macro_cross_entropy=1.03)
    candidate = _wdl(cross_entropy=0.951, macro_cross_entropy=1.029)

    assert _continuous_wdl_gate(
        baseline,
        candidate,
        require_legacy_absolute_floors=False,
    ) == ()


def test_old_tuning_gate_uses_frozen_noninferiority_margins() -> None:
    baseline = {**_wdl(), "brier": 0.55}
    within = {
        **_wdl(
            cross_entropy=0.903,
            macro_cross_entropy=0.925,
            expected_score_pearson=0.44,
            ece_10=0.04,
        ),
        "brier": 0.553,
    }

    assert _old_wdl_point_noninferiority_gate(baseline, within) == ()
    assert len(
        _old_wdl_point_noninferiority_gate(
            baseline,
            {
                **within,
                "cross_entropy": 0.904,
                "macro_cross_entropy": 0.926,
                "brier": 0.554,
                "expected_score_pearson": 0.439,
                "ece_10": 0.041,
            },
        )
    ) == 5


def test_old_local_gate_requires_statistical_margin_violation() -> None:
    intervals = {
        metric: {"low": -0.01, "high": 0.01}
        for metric in ("cross_entropy", "macro_cross_entropy", "brier", "pearson", "ece_10")
    }
    inconclusive = {"intervals": intervals, "candidate": {"ece_10": 0.05}}
    harmful = {
        "intervals": {
            **intervals,
            "cross_entropy": {"low": 0.004, "high": 0.006},
            "pearson": {"low": -0.03, "high": -0.02},
        },
        "candidate": {"ece_10": 0.13},
    }

    assert _old_wdl_statistical_noninferiority_gate(inconclusive) == ()
    assert len(_old_wdl_statistical_noninferiority_gate(harmful)) == 3


def test_search_tactical_gate_ignores_raw_policy_but_preserves_solved_cases() -> None:
    baseline = {
        "raw": {"solved": 1},
        "budgets": [{"solved": 5, "cases": [{"case": str(i), "solved": i < 5} for i in range(8)]}],
    }
    candidate = {
        "raw": {"solved": 0},
        "budgets": [{"solved": 5, "cases": [{"case": str(i), "solved": i < 5} for i in range(8)]}],
    }

    assert _search_tactical_gate(baseline, candidate) == ()


def test_local_arena_rejects_only_supported_catastrophic_regression() -> None:
    uncertain = {"score_interval": {"low": 0.25, "high": 0.4375}}
    catastrophic = {"score_interval": {"low": 0.0, "high": 0.25}}

    assert _local_arena_catastrophic_gate(uncertain) == ()
    assert len(_local_arena_catastrophic_gate(catastrophic)) == 1


def test_fresh_wdl_gate_requires_all_preregistered_directions() -> None:
    baseline = {
        **_wdl(),
        "brier": 0.55,
    }
    candidate = {
        **_wdl(cross_entropy=0.89, macro_cross_entropy=0.91, expected_score_pearson=0.46),
        "brier": 0.54,
    }

    assert _fresh_wdl_direction_gate(baseline, candidate) == ()
    assert len(
        _fresh_wdl_direction_gate(
            baseline,
            {**candidate, "cross_entropy": 0.899},
        )
    ) == 1
    assert len(
        _fresh_wdl_direction_gate(
            baseline,
            {**candidate, "cross_entropy": 0.91, "brier": 0.56},
        )
    ) == 2


def test_fresh_calibration_gate_uses_preregistered_ece_margin() -> None:
    baseline = {"ece_10": 0.08}

    assert _fresh_wdl_calibration_gate(baseline, {"ece_10": 0.10}) == ()
    assert len(_fresh_wdl_calibration_gate(baseline, {"ece_10": 0.101})) == 1


def test_fresh_local_gate_rejects_only_supported_harm() -> None:
    inconclusive = {
        "intervals": {
            metric: {"low": -0.01, "high": 0.01}
            for metric in ("cross_entropy", "macro_cross_entropy", "brier", "pearson")
        }
    }
    harmful = {
        "intervals": {
            **inconclusive["intervals"],
            "cross_entropy": {"low": -0.02, "high": -0.001},
        }
    }

    assert _fresh_wdl_harm_gate(inconclusive) == ()
    assert _fresh_wdl_harm_gate(harmful) == ("paired fresh tuning CE shows harm",)


def test_value_checkpoint_uses_best_fresh_ce_among_safe_steps() -> None:
    checkpoints = (
        {
            "local_step": 1,
            "fresh_wdl": {"cross_entropy": 0.90, "macro_cross_entropy": 0.91},
            "reasons": (),
        },
        {
            "local_step": 8,
            "fresh_wdl": {"cross_entropy": 0.84, "macro_cross_entropy": 0.89},
            "reasons": (),
        },
        {
            "local_step": 20,
            "fresh_wdl": {"cross_entropy": 0.80, "macro_cross_entropy": 0.85},
            "reasons": ("historical regression",),
        },
        {
            "local_step": 30,
            "fresh_wdl": {"cross_entropy": 0.79, "macro_cross_entropy": 0.84},
            "reasons": ("paired fresh Pearson harm",),
        },
    )

    selected, eligible = _select_value_checkpoint(checkpoints)

    assert eligible is True
    assert selected["local_step"] == 8


def test_policy_checkpoint_prefers_earliest_tactical_safe_step() -> None:
    checkpoints = (
        {"local_step": 20, "reasons": ("lost tactic",)},
        {"local_step": 30, "reasons": ()},
        {"local_step": 40, "reasons": ()},
    )

    selected, eligible = _select_tactical_policy_checkpoint(checkpoints)

    assert eligible is True
    assert selected["local_step"] == 30


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


def test_value_trust_region_scales_only_plastic_value_delta() -> None:
    base = _LearnerState(
        step=4,
        weights=(
            ("policy_linear.bias", mx.array([2.0])),
            ("plastic_value_output.bias", mx.array([4.0])),
        ),
        optimizer=(("state", mx.array([1.0])),),
    )
    candidate = _LearnerState(
        step=9,
        weights=(
            ("policy_linear.bias", mx.array([20.0])),
            ("plastic_value_output.bias", mx.array([12.0])),
        ),
        optimizer=(("state", mx.array([3.0])),),
    )

    interpolated = _interpolate_value_state(
        base,
        candidate,
        alpha=0.25,
        value_prefixes=PLASTIC_VALUE_PREFIXES,
    )
    weights = dict(interpolated.weights)

    assert interpolated.step == 9
    assert weights["policy_linear.bias"].item() == 2.0
    assert weights["plastic_value_output.bias"].item() == 6.0
    assert interpolated.optimizer == candidate.optimizer


def test_value_trust_region_rejects_invalid_alpha() -> None:
    state = _LearnerState(0, (("plastic_value_output.bias", mx.array([0.0])),), ())

    with pytest.raises(ValueError, match="alpha"):
        _interpolate_value_state(
            state,
            state,
            alpha=1.1,
            value_prefixes=PLASTIC_VALUE_PREFIXES,
        )


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


def test_fresh_soft_targets_do_not_mix_historical_outcomes() -> None:
    fresh = (
        SimpleNamespace(root_fen="same", moves=(1, 2), outcome_value=1),
        SimpleNamespace(root_fen="same", moves=(1, 2), outcome_value=0),
        SimpleNamespace(root_fen="other", moves=(3,), outcome_value=-1),
    )

    targets, summary = _empirical_fresh_value_targets(fresh)

    assert targets.tolist() == [[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]
    assert summary == {
        "unique_fresh_fit_states": 2,
        "ambiguous_fresh_fit_states": 1,
        "ambiguous_fresh_fit_rows": 2,
    }


def test_stable_value_targets_distill_reference_probabilities() -> None:
    network = HarbiChessDecoupledValueNetwork.from_base(_network())
    inputs = mx.zeros((3, 8, 8, network.config.input_channels))
    expected = mx.softmax(network(inputs)[1], axis=1)

    targets = _stable_value_distillation_targets(network, inputs)

    assert mx.max(mx.abs(targets - expected)).item() == 0.0
    assert mx.max(mx.abs(mx.sum(targets, axis=1) - 1.0)).item() < 1e-6


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


def test_saved_update_reproduces_next_controlled_training_step(tmp_path: Path) -> None:
    network = HarbiChessPlasticValueNetwork.from_mihver(
        HarbiChessDecoupledValueNetwork.from_base(_network())
    )
    network.freeze_to_stable_continuous_heads()
    learner = _ContinuousHeadLearner(network, learning_rate=1e-4)
    state = learner.snapshot()
    checkpoint = tmp_path / "checkpoints" / "update-001"
    checkpoint.mkdir(parents=True)
    network.save_weights(str(checkpoint / "model.safetensors"))
    replay = tmp_path / "replay.jsonl.gz"
    target = tmp_path / "target.json"
    replay.write_bytes(b"replay")
    target.write_text("{}\n", encoding="utf-8")
    save_continuous_resume(
        checkpoint,
        update=1,
        learner_step=state.step,
        next_update_seed=53,
        source_commit="abc123",
        config_sha256="a" * 64,
        optimizer_state=state.optimizer,
        rolling_replay_files=(replay,),
        rolling_target_files=(target,),
    )
    inputs = mx.zeros((4, 8, 8, network.config.input_channels))
    policy_targets = mx.full((4, network.config.policy_size), 1 / network.config.policy_size)
    policy_buffer = PreparedTransfer(
        records=(None,) * 4,
        inputs=inputs,
        targets=policy_targets,
        legal_masks=mx.ones((4, network.config.policy_size), dtype=mx.bool_),
        wdl_targets=(0,) * 4,
    )
    value_targets = mx.array(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.5, 0.5, 0.0))
    )

    result = _verify_resume_exactness(
        checkpoint,
        in_memory_state=state,
        network=network,
        learning_rate=1e-4,
        policy_buffer=policy_buffer,
        historical_inputs=inputs,
        historical_targets=value_targets,
        fresh_inputs=inputs,
        fresh_targets=value_targets,
        batch_size=4,
        historical_value_weight=2.0,
        seed=53,
    )

    assert result["passed"] is True
    assert result["maximum_parameter_delta"] == 0.0
    assert result["maximum_metric_delta"] == 0.0

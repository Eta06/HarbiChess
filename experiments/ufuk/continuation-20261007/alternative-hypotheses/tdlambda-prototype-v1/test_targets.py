import pytest

from targets import transform_episode


def row(root, ply, mover, q, status, outcome=None, **extra):
    return {
        "root_id": root,
        "local_ply": ply,
        "mover": mover,
        "clipped_q_mover": q,
        "episode_status": status,
        "own_wdl_mover": outcome,
        "selected_action_played": True,
        "train_eligible": True,
        "protected_search_aliases": [],
        **extra,
    }


def test_one_ply_terminal_black_mover_uses_mover_result_and_white_frame():
    result = transform_episode(
        [row("g", 0, "black", 0.6, "completed-own-terminal", 1)]
    )
    assert result["rows"][0]["target_white"] == -1.0
    assert result["rows"][0]["target_mover"] == 1.0
    assert result["rows"][0]["endpoint"] == "exact-own-terminal-wdl"


def test_two_ply_white_win_uses_next_black_q_and_exact_terminal():
    result = transform_episode(
        [
            row("g", 0, "white", 0.8, "completed-own-terminal", 1),
            row("g", 1, "black", -0.2, "completed-own-terminal", -1),
        ]
    )
    assert result["rows"][1]["target_white"] == 1.0
    # Black's q=-.2 means White score +.2; lambda=.5 mixes +.2 and terminal +1.
    assert result["rows"][0]["target_white"] == pytest.approx(0.6)
    assert result["rows"][0]["target_mover"] == pytest.approx(0.6)


def test_draw_outcome_is_zero_and_propagates_with_lambda():
    result = transform_episode(
        [
            row("g", 0, "black", 0.4, "completed-own-terminal", 0),
            row("g", 1, "white", 0.2, "completed-own-terminal", 0),
        ]
    )
    assert result["rows"][1]["target_white"] == 0.0
    assert result["rows"][0]["target_white"] == pytest.approx(0.1)
    assert result["rows"][0]["target_mover"] == pytest.approx(-0.1)


def test_unknown_tail_bootstraps_from_last_logged_preaction_q():
    result = transform_episode(
        [
            row("g", 0, "white", 0.8, "unknown-ply-cap"),
            row("g", 1, "black", -0.4, "unknown-ply-cap"),
            row("g", 2, "white", -0.2, "unknown-ply-cap"),
        ]
    )
    assert result["rows"][2]["target_white"] == -0.2
    assert result["rows"][2]["endpoint"] == "last-recorded-preaction-q-bootstrap"
    assert result["rows"][1]["target_white"] == pytest.approx(-0.2)
    assert result["rows"][0]["target_white"] == pytest.approx(0.1)
    assert all(not r["terminal_wdl_used"] for r in result["rows"])


def test_unknown_single_row_retains_own_q_and_never_adds_terminal_label():
    result = transform_episode([row("g", 0, "black", 0.35, "unknown-row-budget-prefix")])
    assert result["rows"][0]["target_mover"] == pytest.approx(0.35)
    assert result["rows"][0]["target_white"] == pytest.approx(-0.35)
    assert result["rows"][0]["terminal_wdl_used"] is False


def test_protected_episode_is_excluded_wholesale_even_if_only_late_row_hits():
    source = [
        row("g", 0, "white", 0.2, "unknown-ply-cap"),
        row("g", 1, "black", 0.1, "unknown-ply-cap", protected_search_aliases=[7]),
    ]
    result = transform_episode(source)
    assert result["status"] == "excluded-protected-trajectory"
    assert result["rows"] == []
    assert source[0]["clipped_q_mover"] == 0.2  # source packet remains untouched


def test_known_terminal_rejects_unknown_or_conflicting_wdl():
    with pytest.raises(ValueError, match="exact mover WDL"):
        transform_episode([row("g", 0, "white", 0.0, "completed-own-terminal", None)])
    with pytest.raises(ValueError, match="disagree"):
        transform_episode(
            [
                row("g", 0, "white", 0.0, "completed-own-terminal", 1),
                row("g", 1, "black", 0.0, "completed-own-terminal", 1),
            ]
        )


def test_gap_and_mixed_episode_status_reject():
    with pytest.raises(ValueError, match="contiguous"):
        transform_episode(
            [row("g", 0, "white", 0.0, "unknown-ply-cap"), row("g", 2, "black", 0.0, "unknown-ply-cap")]
        )
    with pytest.raises(ValueError, match="share one final status"):
        transform_episode(
            [row("g", 0, "white", 0.0, "unknown-ply-cap"), row("g", 1, "black", 0.0, "unknown-row-budget-prefix")]
        )

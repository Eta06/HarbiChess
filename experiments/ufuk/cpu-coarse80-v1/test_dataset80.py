from __future__ import annotations

from dataset80 import EXPECTED_ROW_KEYS, history_sha, labels_to_groups


class FakeBoard:
    turn = True
    legal_moves = ("synthetic-legal-move",)

    def fen(self):
        return "8/8/8/8/8/8/P7/4K2k w - - 0 1"

    def outcome(self, claim_draw=True):
        return None

    def pieces_mask(self, piece, color):
        if piece == 1 and color:
            return 1 << 8
        if piece == 1 and not color:
            return 1 << 48
        return 0


def test_full_history_q_row_conversion_and_unknown_is_not_a_row():
    payload = {"schema": "own-search-deeper-value-labels-v1", "roots": []}
    for i in range(1024):
        prefix = []
        root_fen = "synthetic-root"
        payload["roots"].append(
            {
                "ordinal": i,
                "row_id": f"row-{i}",
                "trajectory_id": f"trajectory-{i // 128}",
                "history_sha256": history_sha(root_fen, prefix),
                "history": {"root_fen": root_fen, "prefix_uci": prefix},
                "root_mover": "white",
                "root_fen4": "8/8/8/8/8/8/P7/4K2k w - -",
                "raw_q_mover": 0.25,
                "target_mover": 0.25,
                "exact_mate_score_range": False,
                "zero_value_is_not_a_draw_certificate": False,
                "nodes": 512,
                "evaluations": 128,
                "completed_depth": 4,
                "root_actions": 1,
            }
        )
    assert set(payload["roots"][0]) == EXPECTED_ROW_KEYS
    groups = labels_to_groups(
        payload,
        replay_position=lambda fen, prefix: FakeBoard(),
        classical_features=lambda board: [0.0] * 18,
        prior_weights=[0.0] * 18,
        prior_scale=600.0,
    )
    assert len(groups) == 8
    assert sum(map(len, groups.values())) == 1024
    assert groups["trajectory-0"][0][2] == 0.25

"""Pure selection and receipt-mismatch tests; no model inference/training."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove

p = Path(__file__).with_name("a100-mc-native-terminal-audit-v3-incremental.py")
spec = importlib.util.spec_from_file_location("neural_audit", p)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


class Tests(unittest.TestCase):
    def test_selection_diverse_fullhistories_both_movers_and_no_label_dependence(self):
        rules = PythonChessRules()
        state = rules.initial_state()
        rows = []
        for index in range(40):
            rows.append(
                NS(
                    transition=NS(pre=state, source_id="synthetic", game_index=0),
                    target_wdl=(1, 0, 0),
                )
            )
            state = rules.apply(
                state, ChessMove(("g1f3", "g8f6", "f3g1", "f6g8")[index % 4])
            )
        one = m.select_neural_rows(rows, rules)
        for row in rows:
            row.target_wdl = (0, 0, 1)
        two = m.select_neural_rows(rows, rules)
        self.assertEqual(one, two)
        self.assertEqual(len({row.transition.pre for row in one}), 18)
        self.assertEqual(
            sum(rules.view(row.transition.pre).side_to_move == "white" for row in one),
            9,
        )
        self.assertEqual(
            sum(rules.view(row.transition.pre).side_to_move == "black" for row in one),
            9,
        )

    def test_behavior_and_base_wdl_mismatch_rejected(self):
        row = NS(
            transition=NS(
                pre=NS(root_fen="x", moves=()), source_id="s", game_index=1, slot=0
            ),
            legal_actions=(1, 2),
            policy=(0.4, 0.6),
            behavior_policy=(0.4, 0.6),
            online_pre_wdl=(0.3, 0.4, 0.3),
            base_policy=(0.5, 0.5),
            base_wdl=(0.2, 0.3, 0.5),
        )
        native = NS(
            behavior=object(), base=object(), encoder=None, config=NS(device="cpu")
        )
        result = NS(
            policy=((0.4, 0.6),) * 18,
            wdl=((0.3, 0.4, 0.3),) * 18,
            base_policy=((0.5, 0.5),) * 18,
            base_wdl=((0.2, 0.3, 0.5),) * 18,
        )
        with (
            patch.object(
                m,
                "make_torch_epoch_inference",
                return_value=lambda states, legal: result,
            ),
            patch.object(m, "torch_model_digest", return_value="synthetic"),
        ):
            self.assertEqual(
                m.check_neural_rows(native, [row] * 18, lambda: None)[
                    "max_absolute_errors"
                ]["base_wdl"],
                0,
            )
            result.base_wdl = ((0.3, 0.3, 0.4),) * 18
            with self.assertRaises(AssertionError):
                m.check_neural_rows(native, [row] * 18, lambda: None)


if __name__ == "__main__":
    unittest.main()

import ast
import copy
import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import chess
import collector
import root_bank as bank
from execute_forensic_native_v4 import phase_paths


class SourceTests(unittest.TestCase):
    def test_native_contract_file_does_not_precreate_prover_output_directory(self):
        output, contract = phase_paths(20262905, "proof")
        self.assertFalse(contract.is_relative_to(output))

    def recipe(self, count=16):
        return {**bank.RECIPE, "count": count, "max_attempts": 64}

    def test_seeded_walk_replay_order_and_global_rng_untouched(self):
        before = random.getstate()
        selected, trace = bank.generate(700001, set(), self.recipe())
        self.assertEqual(random.getstate(), before)
        self.assertEqual((selected, trace), bank.generate(700001, set(), self.recipe()))
        self.assertNotEqual(selected, bank.generate(700002, set(), self.recipe())[0])
        self.assertTrue(selected["synthetic_only"])
        self.assertEqual(len({r["final_alias"] for r in selected["rows"]}), 16)
        for row in selected["rows"]:
            board = chess.Board(row["root_fen"])
            aliases = [bank.alias(board)]
            self.assertLessEqual(6, len(row["prefix_uci"]))
            self.assertLessEqual(len(row["prefix_uci"]), 24)
            for uci in row["prefix_uci"]:
                move = chess.Move.from_uci(uci)
                self.assertIn(move, board.legal_moves)
                board.push(move)
                aliases.append(bank.alias(board))
            self.assertEqual(aliases, row["walk_aliases"])
            self.assertEqual(board.fen(), row["final_fen"])
            self.assertIsNone(board.outcome(claim_draw=True))
            self.assertEqual(collector.alias(board), row["final_alias"])
            self.assertEqual(row["root_id"], row["trajectory_id"])

    def test_protected_final_discard_and_fixed_attempt_no_relaxation(self):
        selection, _ = bank.generate(123, set(), self.recipe(1))
        protected = {selection["rows"][0]["final_alias"]}
        _, trace = bank.generate(123, protected, self.recipe(1))
        self.assertEqual(trace[0]["status"], "protected-final-alias")
        self.assertEqual(trace[1]["status"], "accepted")
        recipe = {**self.recipe(1), "max_attempts": 1}
        with self.assertRaisesRegex(bank.BankExhausted, "cap exhausted") as caught:
            bank.generate(123, protected, recipe)
        self.assertEqual(len(caught.exception.trace), 1)
        self.assertEqual(caught.exception.trace[0]["status"], "protected-final-alias")

    def test_interior_protection_is_exposure_not_claimed_whole_history_filter(self):
        selection, _ = bank.generate(555, set(), self.recipe(1))
        aliases = selection["rows"][0]["walk_aliases"]
        protected = {aliases[0]}
        selected, trace = bank.generate(555, protected, self.recipe(1))
        self.assertEqual(selected["rows"], selection["rows"])
        self.assertEqual(trace[0]["interior_protected_aliases"], sorted(protected))
        self.assertEqual(trace[0]["status"], "accepted")

    def test_terminal_and_duplicate_rejections_are_rule_only(self):
        class Terminal(chess.Board):
            def outcome(self, claim_draw=False):
                return chess.Outcome(chess.Termination.CHECKMATE, chess.WHITE)

        with (
            patch.object(bank.chess, "Board", Terminal),
            self.assertRaisesRegex(ValueError, "cap exhausted"),
        ):
            bank.generate(123, set(), self.recipe(1))
        with (
            patch.object(bank, "alias", return_value=42),
            self.assertRaisesRegex(ValueError, "cap exhausted"),
        ):
            bank.generate(123, set(), self.recipe(2))

    def test_new_sampler_accepts_typed_pool_rejects_fake_ancestry_and_bad_history(self):
        selection, _ = bank.generate(31, set(), self.recipe())
        pool = dict(
            schema=collector.POOL,
            selection_status="pass",
            train_only=True,
            source_selection_sha256="a" * 64,
            procedural_receipt_sha256="b" * 64,
            rows=selection["rows"],
        )
        before = copy.deepcopy(pool)
        chosen, digest = collector.starts(pool, 20262905, set(), n=8, pool_size=16)
        self.assertEqual(pool, before)
        self.assertEqual(len(chosen), 8)
        self.assertEqual(
            digest, hashlib.sha256("\n".join(r["root_id"] for r in chosen).encode()).hexdigest()
        )
        bad = copy.deepcopy(pool)
        bad["rows"][0]["prefix_uci"] = ["e2e5"]
        with self.assertRaises(ValueError):
            collector.starts(bad, 20262905, set(), n=8, pool_size=16)
        pool.pop("procedural_receipt_sha256")
        pool["ancestral_teacher_labels_sha256"] = "b" * 64
        with self.assertRaises(ValueError):
            collector.starts(pool, 20262905, set(), n=8, pool_size=16)

    def test_production_verify_rejects_fixture_even_with_correct_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            receipt = Path(d) / "receipt.json"
            receipt.write_bytes(
                bank.canonical(
                    dict(
                        schema=bank.RECEIPT,
                        status="PASS-rule-only-procedural-root-bank",
                        synthetic_only=True,
                        recipe=self.recipe(),
                        generator_sha256=bank.sha(bank.__file__),
                        chess_version=chess.__version__,
                    )
                )
            )
            with self.assertRaisesRegex(ValueError, "production recipe"):
                bank.verify(bank.ref(receipt))
            receipt.write_text(json.dumps({"schema": "wrong"}))
            with self.assertRaisesRegex(ValueError, "source binding"):
                bank.pinned(dict(path=str(receipt), sha256="0" * 64))

    def test_v4_schema_derivative_preserves_model_math_and_parent_bridge(self):
        original = Path(__file__).resolve().parent.parent / "human-prior-own-v1"
        here = Path(__file__).resolve().parent
        for name in (
            "model.py",
            "parent_bridge.py",
            "parent_seal.py",
            "zero_parent.py",
            "initialize.py",
            "admit_parent.py",
            "prepare_zero.py",
        ):
            self.assertEqual((original / name).read_bytes(), (here / name).read_bytes(), name)

        def learner_advance(path):
            tree = ast.parse(path.read_text())
            learner = next(
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == "Learner"
            )
            method = next(
                node
                for node in learner.body
                if isinstance(node, ast.FunctionDef) and node.name == "advance"
            )
            return ast.dump(method, include_attributes=False)

        self.assertEqual(
            learner_advance(original / "native.py"),
            learner_advance(here / "native.py"),
        )
        old_native = ast.parse((original / "native.py").read_text())
        new_native = ast.parse((here / "native.py").read_text())
        old_math = next(
            node for node in old_native.body if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "MATH" for target in node.targets
            )
        )
        new_math = next(
            node for node in new_native.body if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "MATH" for target in node.targets
            )
        )
        self.assertEqual(
            ast.dump(old_math, include_attributes=False),
            ast.dump(new_math, include_attributes=False),
        )


if __name__ == "__main__":
    unittest.main()

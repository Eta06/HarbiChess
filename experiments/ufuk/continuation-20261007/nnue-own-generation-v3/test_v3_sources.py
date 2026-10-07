"""Dormant-loop pure source/schema tests; no evaluator or learner instantiated."""

import ast
import unittest
from pathlib import Path

import collector
import metadata_factory
import specs

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "nnue-own-training-v2"
PROD = Path(
    "/workspace/HarbiChess/experiments/ufuk/continuation-20261007/nnue-own-producer-v2"
)


def function(path, name):
    tree = ast.parse(path.read_text())
    return next(
        x
        for x in tree.body
        if isinstance(x, ast.FunctionDef | ast.ClassDef) and x.name == name
    )


class Sources(unittest.TestCase):
    def test_network_native_optimizer_and_replay_math_preserved(self):
        self.assertEqual(
            (HERE / "model.py").read_bytes(), (OLD / "model.py").read_bytes()
        )
        for name in ("Learner", "bits_equal", "validate_weights", "load_native"):
            self.assertEqual(
                ast.dump(function(HERE / "native.py", name)),
                ast.dump(function(OLD / "native.py", name)),
            )
        for name in ("Traced", "AliasWriter", "replay", "_label"):
            self.assertEqual(
                ast.dump(function(HERE / "collector.py", name)),
                ast.dump(function(PROD / "collector.py", name)),
            )
        self.assertEqual(
            metadata_factory.SEARCH_SHA,
            "de53c14728a67b7772f18b396ac8ef5c35a144d4e4e616fec40099cd461a6670",
        )

    def test_no_new_teacher_label_file_is_read(self):
        for name in (
            "metadata_factory.py",
            "convert.py",
            "run_collection.py",
            "contracts.py",
        ):
            tree = ast.parse((HERE / name).read_text())
            # Frozen SHA string is ancestry; no path to an SF label file exists in new source.
            constants = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)]
            self.assertNotIn("teacher_labels_path", constants)
        self.assertEqual(collector.POOL, "own-generation-train-roots-v3")

    def test_replay_proof_uses_new_schema_and_new_helper_pin(self):
        self.assertEqual(specs.AUDIT_SHA, specs.sha(HERE / "audit_collection_six.py"))
        tree = ast.parse((HERE / "prove.py").read_text())
        values = [x.value for x in ast.walk(tree) if isinstance(x, ast.Constant)]
        self.assertIn("NNUE-own-training-orchestration-v3", values)
        self.assertNotIn("NNUE-own-training-orchestration-v2", values)
        self.assertIn(
            "own-kingbucket-nnue16-generation-native-cpu-v3",
            (HERE / "native.py").read_text(),
        )

    def test_actual_cli_parser_interfaces_are_present(self):
        # Compile every source, no code execution/model import needed for syntax.
        for path in HERE.glob("*.py"):
            compile(path.read_bytes(), str(path), "exec")
        tree = ast.parse((HERE / "parent_seal.py").read_text())
        self.assertTrue(
            any(
                isinstance(n, ast.Constant) and n.value == "--generation"
                for n in ast.walk(tree)
            )
        )


if __name__ == "__main__":
    unittest.main()

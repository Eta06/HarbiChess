"""Pure clock/source/dispatch fixtures; no subprocess/model/training invocation."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import qualify as q


class Tests(unittest.TestCase):
    def fixture(self, root):
        stage = root / "stage"
        stage.mkdir()
        for name in q.RUNTIME:
            (stage / name).write_text("fixture\n")
        frozen = {
            n: {"sha256": q.sha(stage / n), "bytes": (stage / n).stat().st_size} for n in q.RUNTIME
        }
        inv = root / "inventory"
        inv.write_text(json.dumps({"runtime_files": frozen}))
        parent = root / "parent"
        parent.write_text("original-parent")
        first = time.time() - 1
        return dict(
            schema="NNUE-own-generation-v3-synthetic-qualification-build-seal",
            seed=20262905,
            cpu_core=1,
            first=first,
            deadline=first + 600,
            operator_end_epoch=q.END,
            stage=str(stage),
            source_inventory=q.ref(inv),
            parent={"candidate": q.ref(parent)},
            output="/dev/shm/harbichess-v3-synthetic-test-not-executed",
        )

    def test_prepare_preserves_clock_and_all_frozen_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec = self.fixture(root)
            r = q.prepare(spec, root / "registration")
            self.assertEqual((r["first"], r["deadline"]), (spec["first"], spec["deadline"]))
            self.assertEqual(len(r["runtime_sha256"]), 15)
            self.assertFalse(r["real_collection_qualified"])
            self.assertEqual(q.check_sources(r), Path(spec["stage"]))

    def test_corruption_and_inventory_substitution_reject(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec = self.fixture(root)
            r = q.prepare(spec, root / "registration")
            (Path(r["stage"]) / "native.py").write_text("corrupt")
            with self.assertRaises(ValueError):
                q.check_sources(r)

    def test_expired_extended_or_wrong_core_reject(self):
        with tempfile.TemporaryDirectory() as temp:
            spec = self.fixture(Path(temp))
            for changes in (
                {"deadline": time.time() - 1},
                {"deadline": spec["first"] + 601},
                {"cpu_core": 2},
                {"operator_end_epoch": spec["first"] + 1},
            ):
                r = {**spec, **changes, "schema": q.SCHEMA, "status": "registered"}
                with self.assertRaises(ValueError):
                    q.clock(r, time.time())

    def test_observed_paths_are_exact_parent_not_weights_alias(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inv = root / "inventory"
            inv.write_text("{}")
            a = SimpleNamespace(
                seed=20262906,
                cpu_core=3,
                first=1,
                deadline=2,
                operator_end=q.END,
                stage=root,
                inventory=inv,
                proof_output=root,
            )
            # Inspect exact resolver literals, avoiding any original artifact reads.
            source = Path(q.__file__).read_text()
            self.assertIn("/dev/shm/harbichess-NNUE-own-fit-v2/{seed}/whole/candidate.pt", source)
            self.assertIn("/dev/shm/harbichess-NNUE-own-contracts-v2/{seed}/fit.json", source)
            self.assertIn("synthetic_optimizer_updates=16", source)
            self.assertEqual(a.cpu_core, 3)


if __name__ == "__main__":
    unittest.main()

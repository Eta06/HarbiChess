"""Pure confirmation bookkeeping/resource evidence; no models, outcomes or jobs."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


class ConfirmationTests(unittest.TestCase):
    def test_identical_newbook_pins_across_all_evaluation_helpers(self):
        base = module("a100-mc-baseline-strength-controller-v3")
        final = module("a100-mc-final-two-arm-controller")
        analysis = module("a100-mc-strength-analysis")
        self.assertEqual(base.BOOKS, final.BOOKS)
        self.assertEqual(base.BOOKS, analysis.BOOKS)
        self.assertEqual(set(base.BOOKS), {20261205, 20261206})

    def test_guard_snapshot_names_exact_failure_and_boundary(self):
        for name in (
            "a100-mc-baseline-strength-controller-v3",
            "a100-mc-final-two-arm-controller",
        ):
            controller = module(name)
            memory_owner = controller.c if hasattr(controller, "c") else controller
            with (
                patch.object(
                    memory_owner,
                    "cpu_total_memory",
                    return_value=(64 * 1024**3, "cgroup.memory.current"),
                ),
                patch.object(controller.time, "time", return_value=100),
                patch("shutil.disk_usage", return_value=SimpleNamespace(free=8 * 1024**3)),
            ):
                snapshot = controller.resource_snapshot(Path("/unused"), 101)
                self.assertEqual(snapshot["violations"], [])
                self.assertEqual(snapshot["memory_metric"], "cgroup.memory.current")
                self.assertEqual(snapshot["memory_limit_bytes"], 64 * 1024**3)
            with (
                patch.object(
                    memory_owner,
                    "cpu_total_memory",
                    return_value=(64 * 1024**3 + 1, "system-used-MemTotal-minus-MemAvailable"),
                ),
                patch.object(controller.time, "time", return_value=101),
                patch("shutil.disk_usage", return_value=SimpleNamespace(free=8 * 1024**3 - 1)),
            ):
                snapshot = controller.resource_snapshot(Path("/unused"), 101)
                self.assertEqual(
                    snapshot["violations"],
                    ["absolute_deadline", "memory_above_64GiB", "disk_free_below_8GiB"],
                )
                self.assertEqual(
                    snapshot["memory_metric"], "system-used-MemTotal-minus-MemAvailable"
                )

    def test_formal2_manifest_cannot_be_silently_qualified_as_slot3(self):
        for name in ("a100-mc-fixed-candidate-eligibility-v2-neural", "a100-mc-strength-analysis"):
            loaded = module(name)
            with tempfile.TemporaryDirectory() as temporary:
                manifest = Path(temporary) / "manifest.json"
                manifest.write_text(json.dumps({"qualification_ledger_slot": 2}))
                output = Path(temporary) / "output.json"
                with (
                    patch("sys.argv", [name, "--manifest", str(manifest), "--output", str(output)]),
                    self.assertRaises(AssertionError),
                ):
                    loaded.main()
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

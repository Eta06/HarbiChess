"""Synthetic deadline/quiescence checks only; no training, arenas, GPU or SSH."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "orchestrator", Path(__file__).with_name("orchestrate.py")
)
o = importlib.util.module_from_spec(spec)
spec.loader.exec_module(o)


class ScaffoldTests(unittest.TestCase):
    def test_ready_original_finished_file_is_accepted_after_its_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text("{}")
            with patch.object(o.time, "time", return_value=20):
                o.wait_ready([path], 10)

    def test_missing_original_deadline_receipt_fails_without_reset(self):
        with patch.object(o.time, "time", return_value=20), self.assertRaises(TimeoutError):
            o.wait_ready([Path("/absent-immutable-result")], 10)

    def test_inflight_json_publication_waits_but_semantic_failure_is_not_masked(self):
        with (
            patch.object(
                o,
                "read",
                side_effect=[o.json.JSONDecodeError("inflight", "", 0), {"status": "failed"}],
            ),
            patch.object(o.time, "time", return_value=1),
            patch.object(o.time, "sleep") as sleep,
        ):
            self.assertEqual(o.wait_json(Path("/synthetic"), 10), {"status": "failed"})
            sleep.assert_called_once_with(1)
        with (
            patch.object(o, "read", side_effect=o.json.JSONDecodeError("inflight", "", 0)),
            patch.object(o.time, "time", return_value=11),
            self.assertRaises(TimeoutError),
        ):
            o.wait_json(Path("/synthetic"), 10)

    def test_quiescence_wait_is_bounded_and_not_recursive(self):
        with (
            patch.object(o, "quiescent", side_effect=[AssertionError(), None]),
            patch.object(o.time, "time", return_value=1),
            patch.object(o.time, "sleep"),
        ):
            o.wait_quiescent(10)
        with (
            patch.object(o, "quiescent", side_effect=AssertionError()),
            patch.object(o.time, "time", return_value=11),
            self.assertRaises(TimeoutError),
        ):
            o.wait_quiescent(10)


if __name__ == "__main__":
    unittest.main()

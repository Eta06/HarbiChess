"""CPU synthetic audit invariants only: no model loads, inference, queries or optimizer."""

import hashlib
import importlib.util
import json
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))
spec = importlib.util.spec_from_file_location(
    "own_audit", Path(__file__).with_name("ownv1_full_audit.py")
)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


@dataclass(frozen=True)
class State:
    ply: int
    history: tuple


class AuditTests(unittest.TestCase):
    def test_original_actor_serialization_normalizes_float32_reference_once(self):
        import math

        raw = (0.20000000298023224, 0.30000001192092896, 0.5)
        pi = tuple(x / math.fsum(raw) for x in raw)
        logs = [math.log(v) for v in pi]
        scaled = [math.exp(v - max(logs)) for v in logs]
        row = SimpleNamespace(
            policy=pi,
            base_policy=pi,
            online_pre_wdl=pi,
            base_wdl=pi,
            behavior_policy=tuple(v / math.fsum(scaled) for v in scaled),
        )
        a.compare_actor_packet(row, raw, raw, raw, raw)
        row.base_policy = (pi[1], pi[0], pi[2])
        with self.assertRaises(AssertionError):
            a.compare_actor_packet(row, raw, raw, raw, raw)

    def test_same_production_terminal_and_search_mutation_checks(self):
        a.compare_terminal_packet((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        with self.assertRaises(AssertionError):
            a.compare_terminal_packet((0.0, 1.0, 0.0), (1.0, 0.0, 0.0))

        def canonical(value):
            return json.dumps(value, sort_keys=True)

        original = {"collection_index": 0, "search_policy": [0.4, 0.6]}
        a.compare_search_packet(original, original, canonical)
        with self.assertRaises(AssertionError):
            a.compare_search_packet({**original, "search_policy": [0.6, 0.4]}, original, canonical)

    def test_raw_packet_exactness_and_original128_actor_batch_composition(self):
        import math

        groups = a.actor_packet_groups([0, 127, 129, 32767], 32768)
        self.assertEqual(
            groups, [list(range(128)), list(range(128, 256)), list(range(32640, 32768))]
        )
        pi = (0.2, 0.3, 0.5)
        logs = [math.log(v) for v in pi]
        scaled = [math.exp(v - max(logs)) for v in logs]
        mu = tuple(v / math.fsum(scaled) for v in scaled)
        row = SimpleNamespace(
            policy=pi, base_policy=pi, online_pre_wdl=pi, base_wdl=pi, behavior_policy=mu
        )
        a.compare_actor_packet(row, pi, pi, pi, pi)
        with self.assertRaises(AssertionError):
            a.compare_actor_packet(row, (0.3, 0.2, 0.5), pi, pi, pi)
        with self.assertRaises(AssertionError):
            a.compare_actor_packet(row, pi, pi, pi, (0.3, 0.2, 0.5))
        with self.assertRaises(AssertionError):
            a.actor_packet_groups([32768], 32768)

    def test_chain_rejects_mutated_ledger_not_only_digest_presence(self):
        parent = "b" * 64
        payload = {
            "previous_sample_chain_sha256": parent,
            "own_search": {"selected_indices": [0, 3]},
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(bytes.fromhex(parent) + canonical).hexdigest()
        record = {**payload, "sample_chain_sha256": digest}
        a.check_chain(record, parent, digest)
        record["own_search"]["selected_indices"].append(4)
        with self.assertRaises(AssertionError):
            a.check_chain(record, parent, digest)

    def test_designated108_scope_keeps_complete_firstlast_originalgroups(self):
        epoch = SimpleNamespace(
            actions=[
                SimpleNamespace(transition=SimpleNamespace(pre=State(i, tuple(range(i)))))
                for i in range(80)
            ]
        )
        rules = SimpleNamespace(
            view=lambda state: SimpleNamespace(
                side_to_move=SimpleNamespace(value="white" if state.ply % 2 == 0 else "black")
            )
        )
        groups = [list(range(40)), list(range(40, 60)), list(range(60, 80))]
        chosen = a.prescribe_roots(epoch, groups, rules)
        self.assertEqual(len(set(chosen)), 18)
        self.assertTrue(set(chosen) <= set(groups[0] + groups[-1]))
        self.assertEqual(sum(i % 2 == 0 for i in chosen), 9)
        # Root group sizes are deliberately larger than18; no scalar equivalence claim.
        self.assertEqual(sum(len(group) for group in (groups[0], groups[-1])), 60)

    def test_insufficient_mover_coverage_fails_without_changing_groups(self):
        epoch = SimpleNamespace(
            actions=[
                SimpleNamespace(transition=SimpleNamespace(pre=State(2 * i, (i,))))
                for i in range(40)
            ]
        )
        rules = SimpleNamespace(
            view=lambda state: SimpleNamespace(side_to_move=SimpleNamespace(value="white"))
        )
        with self.assertRaises(AssertionError):
            a.prescribe_roots(epoch, [list(range(40))], rules)


if __name__ == "__main__":
    unittest.main()

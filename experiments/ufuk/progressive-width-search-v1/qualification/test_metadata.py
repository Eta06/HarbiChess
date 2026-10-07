import ast
import unittest
from pathlib import Path

import develop


class MetadataTests(unittest.TestCase):
    def registration(self):
        return dict(
            schema="progressive-width-human-prior-development-registration-v1",
            status="registered",
            mode="profile",
            cpu_core=0,
            advanced_incheck_extensions=1,
            first=1000.0,
            deadline=1600.0,
            operator_end_epoch=2000.0,
            old_search={"sha256": develop.OLD},
            advanced_search={"sha256": develop.ADV},
            helper_sha256=develop.sha(develop.__file__),
            search_math=dict(nodes=512, qdepth=2, max_depth=8),
            root_width="progressive",
        )

    def test_generic_core0_and_fixed_budget_schema(self):
        r = self.registration()
        develop.clock(r, 1000.0)
        for key, value in [
            ("cpu_core", -1),
            ("root_width", "adaptive"),
            ("deadline", 1601.0),
            ("schema", "pv-history-human-prior-development-registration-v1"),
        ]:
            bad = {**r, key: value}
            with self.assertRaises(ValueError):
                develop.clock(bad, 1000.0)

    def test_runtime_affinity_uses_registered_core_not_old_literal2(self):
        tree = ast.parse(Path(develop.__file__).read_text())
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "sched_setaffinity"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(ast.unparse(calls[0].args[1]), "{reg['cpu_core']}")


if __name__ == "__main__":
    unittest.main()

import unittest

from select_roots import select_roots


class SelectRootsTests(unittest.TestCase):
    def test_is_deterministic_balanced_and_outcome_blind(self):
        groups = {
            f"game-{g:02d}": [
                {"row_id": f"{g}:{i}", "result": "white-win", "q": 0.1}
                for i in range(60 + g)
            ]
            for g in range(20)
        }
        first = select_roots(groups, 20262705, 1024)
        changed_labels = {
            game: [dict(row, result="draw", q=-0.9) for row in rows]
            for game, rows in groups.items()
        }
        second = select_roots(changed_labels, 20262705, 1024)
        self.assertEqual([r["row_id"] for r in first], [r["row_id"] for r in second])
        counts = {}
        for row in first:
            game = row["row_id"].split(":", 1)[0]
            counts[game] = counts.get(game, 0) + 1
        self.assertEqual(len(first), 1024)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_rejects_bad_or_insufficient_inputs(self):
        with self.assertRaises(ValueError):
            select_roots({"one": [{"row_id": "a"}]}, 1, 1)
        groups = {f"g{i}": [{"row_id": f"{i}:0"}] for i in range(16)}
        with self.assertRaises(ValueError):
            select_roots(groups, 1, 17)
        groups["g0"].append({"row_id": "1:0"})
        with self.assertRaises(ValueError):
            select_roots(groups, 1, 16)


if __name__ == "__main__":
    unittest.main()

"""Metadata/trace tests only: no search/model/Stockfish subprocesses."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import chess
import develop


class MetadataTests(unittest.TestCase):
    def reg(self):
        return dict(
            schema="pv-history-human-prior-development-registration-v1",
            status="registered",
            mode="profile",
            first=100.0,
            deadline=700.0,
            operator_end_epoch=1000.0,
            cpu_core=2,
            advanced_incheck_extensions=1,
            old_search={"sha256": develop.OLD},
            advanced_search={"sha256": develop.ADV},
            helper_sha256=develop.sha(develop.__file__),
            search_math=dict(nodes=512, qdepth=2, max_depth=8),
        )

    def test_clock_and_ablation_fail_closed(self):
        reg = self.reg()
        develop.clock(reg, 100.0)
        for now in (99.0, 700.0):
            with self.assertRaises(ValueError):
                develop.clock(reg, now)
        reg.update(mode="arena", deadline=900.0, advanced_incheck_extensions=0)
        with self.assertRaisesRegex(ValueError, "profile-only"):
            develop.clock(reg, 101.0)
        reg["advanced_incheck_extensions"] = 1
        reg["advanced_search"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            develop.clock(reg, 101.0)

    def test_exact_trace_bytes_order_and_full_history_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = develop.Trace(Path(tmp), lambda _: -0.0)
            a = chess.Board()
            b = chess.Board()
            for move in ["g1f3", "g8f6", "f3g1", "f6g8"]:
                b.push_uci(move)
            trace.begin(a)
            trace(a)
            trace(b)
            ref = trace.finish()
            trace.close()
            raw = (Path(tmp) / ref["file"]).read_bytes()
            self.assertEqual(ref["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(len(raw), 2 * develop.RECORD.size)
            first = develop.RECORD.unpack(raw[: develop.RECORD.size])
            second = develop.RECORD.unpack(raw[develop.RECORD.size :])
            self.assertNotEqual(first[0], second[0])
            self.assertEqual(first[1], second[1])
            self.assertEqual(first[2].hex(), "-0x0.0p+0")

    def test_legal_fullhistory_and_terminal_claim(self):
        b = develop.replay(chess.STARTING_FEN, ["e2e4", "a7a6", "e4e5", "d7d5"])
        self.assertTrue(b.is_en_passant(chess.Move.from_uci("e5d6")))
        self.assertEqual(len(b.move_stack), 4)
        with self.assertRaises(ValueError):
            develop.replay(chess.STARTING_FEN, ["e2e5"])
        with self.assertRaisesRegex(ValueError, "nonterminal"):
            develop.replay(chess.STARTING_FEN, ["g1f3", "g8f6", "f3g1", "f6g8"] * 2)

    def test_actual_counter_packet_validation(self):
        board = chess.Board()
        result = SimpleNamespace(move=chess.Move.from_uci("e2e4"), info={"nodes": 519, "depth": 4})
        self.assertEqual(develop.sf_packet(result, board)["overrun_nodes"], 7)
        result.info["nodes"] = True
        with self.assertRaises(ValueError):
            develop.sf_packet(result, board)
        result = SimpleNamespace(
            move=chess.Move.from_uci("e2e4"),
            value=0.0,
            root_actions=20,
            nodes=20,
            evaluations=20,
            completed_depth=0,
        )
        with self.assertRaisesRegex(ValueError, "charged"):
            develop.packet(result, board)
        result.nodes = 21
        self.assertEqual(develop.packet(result, board)["completed_depth"], 0)

    def test_fixed_first24_train_rows_no_score_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pool.json"
            rows = [
                dict(
                    root_id=str(i),
                    role="TRAIN",
                    root_fen=chess.STARTING_FEN,
                    prefix_uci=["e2e4"],
                    unused_score=-i,
                )
                for i in range(4096)
            ]
            path.write_text(
                json.dumps(
                    dict(
                        schema="teacher-selected-ownq-train-roots-v2",
                        train_only=True,
                        selection_status="pass",
                        rows=rows,
                    )
                )
            )
            reg = {"train_roots": {"path": str(path), "sha256": develop.sha(path)}}
            roots = develop.prepare_profile_roots(reg)
            self.assertEqual([r["root_id"] for r, _ in roots], [str(i) for i in range(24)])
            rows[0]["role"] = "VAL"
            path.write_text(
                json.dumps(
                    dict(
                        schema="teacher-selected-ownq-train-roots-v2",
                        train_only=True,
                        selection_status="pass",
                        rows=rows,
                    )
                )
            )
            reg["train_roots"]["sha256"] = develop.sha(path)
            with self.assertRaises(ValueError):
                develop.prepare_profile_roots(reg)


if __name__ == "__main__":
    unittest.main()

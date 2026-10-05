"""Small synthetic checks only; no actual fit, search, or game."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import chess
from frozen_reference.value import FEATURE_SCALES, ClassicalValue, features
from mixed_value import MixedValue
from pst_features import FEATURE_COUNT, material_phase, piece_square_features
from pst_learner import PSTLearner, make_contract
from pst_native import canonical, decode, encode
from pst_value import PST_L2, PST_SMOOTHNESS, PSTValue, model_dict
from train_pst import validate_output_mode


def synthetic_groups():
    result = {}
    boards = []
    board = chess.Board()
    for move in ("e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5"):
        boards.append(board.copy(stack=True))
        board.push_uci(move)
    for i, position in enumerate(boards):
        raw = features(position)
        phi = tuple(x / s for x, s in zip(raw, FEATURE_SCALES, strict=True))
        prior_logit = (
            sum(
                w * x
                for w, x in zip(
                    (
                        100,
                        320,
                        330,
                        500,
                        900,
                        30,
                        5,
                        14,
                        8,
                        18,
                        2,
                        18,
                        -16,
                        -12,
                        -10,
                        10,
                        2,
                        -8,
                    ),
                    raw,
                    strict=True,
                )
            )
            / 600
        )
        rows = result.setdefault(f"group-{i % 2}", [])
        rows.append(
            (
                phi,
                __import__("math").tanh(prior_logit),
                1.0 if i % 2 else -1.0,
                0.25,
                piece_square_features(position),
                prior_logit,
            )
        )
    return result, boards


class PSTTests(unittest.TestCase):
    def test_output_modes_allow_only_existing_resume_or_audit_branch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            branch = root / "branch"
            validate_output_mode(branch, resume=None, audit_only=False)
            branch.mkdir()
            with self.assertRaises(FileExistsError):
                validate_output_mode(branch, resume=None, audit_only=False)
            validate_output_mode(
                branch, resume=Path("checkpoint.native.gz"), audit_only=False
            )
            validate_output_mode(branch, resume=None, audit_only=True)
            with self.assertRaises(FileNotFoundError):
                validate_output_mode(
                    root / "missing", resume=Path("checkpoint"), audit_only=False
                )

    def test_fit_contract_binds_same_final_data_and_update_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            journal = root / "journal.gz"
            journal.write_bytes(b"final-16384-frozen-journal")
            config_path = root / "config.json"
            protected = ["fen-key"]
            config = {
                "seed": 17,
                "max_actions": 16384,
                "source_commit": "6fcc8b476d25495d1c9c413e55b2c7ba4794013e",
                "model_sha256": "c" * 64,
                "excluded_training_position_keys": protected,
            }
            config_path.write_text(json.dumps(config))

            def file_sha(path):
                return hashlib.sha256(path.read_bytes()).hexdigest()

            protocol = {
                "schema": "classical-own-pst-offline-protocol-v1",
                "status": "registered-pst-proposal-fit-not-strength",
                "source_commit": config["source_commit"],
                "prior_model_sha256": config["model_sha256"],
                "seeds": [17],
                "final_actions": 16384,
                "inputs": {
                    "17": {
                        "journal_sha256": file_sha(journal),
                        "config_sha256": file_sha(config_path),
                        "classical18_dataset_sha256": "a" * 64,
                        "PST_dataset_sha256": "b" * 64,
                    }
                },
                "config_paths": {"17": str(config_path)},
                "deadline_by_seed": {"17": 1791270000.0},
                "phase_clocks": {
                    "fit": {
                        "first_epoch": 1791268200.0,
                        "deadline_epoch": 1791270000.0,
                    }
                },
                "protected_position_keys": protected,
                "objective": {
                    "terminal_weight": 0.75,
                    "search_weight": 0.25,
                    "linear_prior_l2": 0.01,
                    "pst_l2": PST_L2,
                    "pst_neighbor_smoothness": PST_SMOOTHNESS,
                },
                "optimizer": {
                    "lr": 0.01,
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "eps": 1e-8,
                    "batch": 256,
                    "slots": 4,
                    "max_updates": 1024,
                },
            }
            protocol_path = root / "protocol.json"
            protocol_path.write_text(json.dumps(protocol, sort_keys=True))
            receipt = {
                "actions": 16384,
                "training_rows": 12345,
                "linear_dataset_sha256": "a" * 64,
            }
            contract = make_contract(
                protocol, protocol_path, config, journal, 17, receipt, "b" * 64
            )
            self.assertEqual(contract["updates"], 192)
            with self.assertRaises(ValueError):
                make_contract(
                    protocol,
                    protocol_path,
                    config,
                    journal,
                    17,
                    {**receipt, "linear_dataset_sha256": "d" * 64},
                    "b" * 64,
                )
            with self.assertRaises(ValueError):
                make_contract(
                    protocol, protocol_path, config, journal, 17, receipt, "e" * 64
                )
            journal.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                make_contract(
                    protocol, protocol_path, config, journal, 17, receipt, "b" * 64
                )

    def test_zero_pst_is_exact_classical18_identity(self):
        board = chess.Board()
        positions = [board.copy(stack=True)]
        for move in ("e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4"):
            board.push_uci(move)
            positions.append(board.copy(stack=True))
        linear = ClassicalValue(theta=[0.01 * i for i in range(18)])
        pst = PSTValue(
            theta=[0.01 * i for i in range(18)], pst_theta=[0.0] * FEATURE_COUNT
        )
        for position in positions:
            self.assertEqual(pst(position), linear(position))

    def test_zero_pst_preserves_frozen_human_prior_exactly(self):
        board = chess.Board()
        self.assertEqual(PSTValue()(board), ClassicalValue()(board))

    def test_color_rank_reflection_preserves_feature_vector(self):
        board = chess.Board()
        for move in ("e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4"):
            board.push_uci(move)
        reflected = board.mirror()
        self.assertEqual(piece_square_features(board), piece_square_features(reflected))

    def test_feature_extraction_does_not_change_full_history(self):
        board = chess.Board()
        for move in ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5"):
            board.push_uci(move)
        before = (board.fen(), tuple(board.move_stack))
        values = piece_square_features(board)
        self.assertEqual(len(values), 224)
        self.assertEqual(before, (board.fen(), tuple(board.move_stack)))

    def test_king_phase_interpolation_endpoints(self):
        opening = chess.Board()
        bare_kings = chess.Board("8/8/8/8/8/8/6k1/K7 w - - 0 1")
        self.assertEqual(material_phase(opening), 1.0)
        self.assertEqual(material_phase(bare_kings), 0.0)
        opening_features = piece_square_features(opening)
        ending_features = piece_square_features(bare_kings)
        self.assertTrue(all(x == 0.0 for x in opening_features[192:224]))
        self.assertTrue(all(x == 0.0 for x in ending_features[160:192]))

    def test_terminal_claims_keep_mover_perspective(self):
        board = chess.Board()
        for move in ("g1f3", "g8f6", "f3g1", "f6g8") * 2:
            board.push_uci(move)
        self.assertTrue(board.can_claim_threefold_repetition())
        self.assertEqual(PSTValue()(board), 0.0)

    def test_adam_native_pause_resume_exact(self):
        groups, _ = synthetic_groups()
        contract = {"updates": 4, "seed": 31, "dataset_sha256": "d" * 64}
        whole = PSTLearner(31, contract)
        whole.advance(groups, 4)
        self.assertTrue(any(whole.theta[18:]))
        split = PSTLearner(31, contract)
        split.advance(groups, 2)
        pause = decode(encode(split.native(), contract), contract)
        resumed = PSTLearner(31, contract, pause)
        resumed.advance(groups, 4)
        self.assertEqual(
            encode(whole.native(), contract), encode(resumed.native(), contract)
        )

    def test_json_native_strict_roundtrip_canonicalizes_rng_tuples(self):
        groups, _ = synthetic_groups()
        contract = {"updates": 1, "seed": 41, "dataset_sha256": "f" * 64}
        learner = PSTLearner(41, contract)
        learner.advance(groups, 1)
        decoded = decode(encode(learner.native(), contract), contract)
        restored = PSTLearner(41, contract, decoded)
        self.assertEqual(canonical(restored.native()), canonical(decoded))

    def test_native_contract_tamper_rejected(self):
        groups, _ = synthetic_groups()
        contract = {"updates": 1, "seed": 3, "dataset_sha256": "a" * 64}
        state = PSTLearner(3, contract).advance(groups, 1)
        tampered = copy.deepcopy(state)
        tampered["candidate"]["pst_theta"][0] = 1.0
        from pst_native import validate

        with self.assertRaises(ValueError):
            validate(tampered, contract)
        with self.assertRaises(ValueError):
            decode(encode(state, contract), {**contract, "dataset_sha256": "b" * 64})

    def test_model_schema_rejects_nonfinite_or_wrong_length(self):
        state = model_dict()
        state["pst_theta"] = state["pst_theta"][:-1]
        from pst_value import load_pst

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "classical-pst-bad.json"
            path.write_text(json.dumps(state))
            with self.assertRaises(ValueError):
                load_pst(path)

    def test_mixed_value_routes_four_separate_roles(self):
        class Stub:
            def __call__(self, _board):
                return 0.125

        self.assertEqual(MixedValue("e8", e8_evaluator=Stub())(chess.Board()), 0.125)
        with self.assertRaises(ValueError):
            MixedValue("unknown", model_path="unused")


if __name__ == "__main__":
    unittest.main()

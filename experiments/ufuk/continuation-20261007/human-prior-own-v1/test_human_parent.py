"""Synthetic initial-state and metadata regression; no training/search/teacher."""

import copy
import importlib.util
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import native
import torch
from model import FEATURE_SCHEMA
from zero_parent import PHASE, origin, zero_head


def contract():
    return dict(
        phase=PHASE,
        updates=0,
        generation=0,
        seed=123,
        math=native.MATH,
        feature_schema=FEATURE_SCHEMA,
    )


class ZeroTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_literal_zero_native_candidate_and_rng_restore(self):
        c = contract()
        learner = native.Learner(c)
        state = learner.native()
        zero_head(state["model"])
        self.assertEqual(learner.step, 0)
        self.assertFalse(learner.optimizer.state)
        self.assertTrue(native.bits_equal(learner.candidate()["model"], state["baseline"]))
        restored = native.Learner(c, state=state)
        self.assertTrue(native.bits_equal(state, restored.native()))
        random.random()
        torch.rand(1)
        restored.sampler.random()
        restored = native.Learner(c, state=state)
        self.assertTrue(native.bits_equal(state, restored.native()))

    def test_corruption_signedzero_embedding_rng_phase_rejected(self):
        c = contract()
        state = native.Learner(c).native()
        for part in ("head", "embedding", "rng", "optimizer"):
            broken = copy.deepcopy(state)
            if part == "head":
                broken["model"]["head.bias"][0] = -0.0
            if part == "embedding":
                broken["model"]["embedding.weight"][0, 0] += 0.01
            if part == "rng":
                broken["sampler_rng"] = random.Random(0).getstate()
            if part == "optimizer":
                broken["step"] = 1
            with self.assertRaises(ValueError):
                native.Learner(c, state=broken)
        bad = {**c, "phase": "teacher-bootstrap", "updates": 256}
        with self.assertRaises(ValueError):
            native.Learner(bad)
        with self.assertRaises(ValueError):
            native.Learner({**c, "phase": "own-learning", "updates": 64})

    def test_compiled_reader_literal_zero_delegates_authoritative_fake_prior_hex(self):
        path = Path("/workspace/work/harbichess/cpu-kingbucket-nnue-proposal/evaluator.py")
        spec = importlib.util.spec_from_file_location("test_literal_reader", path)
        reader = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = reader
        spec.loader.exec_module(reader)

        class Never:
            def forward(self, *_):
                raise AssertionError("zero-head must not reconstruct prior in C")

        prior = SimpleNamespace(nonterminal=lambda _: -0.0)
        state = native.Learner(contract()).model.state_dict()
        evaluator = reader.Evaluator(state, prior=prior, compiled=Never())
        self.assertEqual(evaluator.nonterminal(object()).hex(), "-0x0.0p+0")
        prior.nonterminal = lambda _: 0.123456789
        self.assertEqual(evaluator.nonterminal(object()).hex(), (0.123456789).hex())

    def test_initializer_source_closure_and_no_teacher_scope(self):
        import initialize

        stage = Path("/workspace/work/harbichess/cpu-kingbucket-nnue-proposal")
        prior = Path("/workspace/HarbiChess/experiments/ufuk/cpu-classical-own-v1/value.py")
        old = Path("/workspace/HarbiChess/experiments/ufuk/cpu-budget-search-v1/search.py")
        files = [
            stage / n
            for n in (
                "model.py",
                "native.py",
                "evaluator.py",
                "forward.c",
                "_kingbucket16.cpython-312-x86_64-linux-gnu.so",
                "build-receipt.json",
            )
        ]
        files.append(prior)

        def ref(p):
            return dict(path=str(p), sha256=initialize.sha(p))

        seal = dict(
            schema="human-prior-zero-initialization-seal-v1",
            status="registered",
            seed=20262905,
            first=100.0,
            deadline=600.0,
            operator_end_epoch=1000.0,
            core_repo="/workspace/work/harbichess/cpu-additive-source-6fcc8b4",
            core_commit="6fcc8b476d25495d1c9c413e55b2c7ba4794013e",
            prior_helper=ref(prior),
            search_helper=ref(old),
            teacher_labels_used=False,
            inference_source_sha256={str(p): initialize.sha(p) for p in files},
        )
        c = initialize.build_contract(seal)
        self.assertEqual(c["generation"], 0)
        self.assertEqual(c["phase"], PHASE)
        bad = copy.deepcopy(seal)
        bad["teacher_labels_used"] = True
        with self.assertRaises(ValueError):
            initialize.build_contract(bad)
        bad = copy.deepcopy(seal)
        bad["teacher_dataset"] = ref(old)
        with self.assertRaises(ValueError):
            initialize.build_contract(bad)
        bad = copy.deepcopy(seal)
        bad["inference_source_sha256"].pop(str(stage / "forward.c"))
        with self.assertRaisesRegex(ValueError, "closure"):
            initialize.build_contract(bad)

    def test_zero_receipt_actual_argv_namespace_and_corruption(self):
        from parent_bridge import sha
        from zero_parent import validate_zero_metadata

        directory = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            def save(name, value):
                path = tmp / name
                path.write_text(json.dumps(value))
                return dict(path=str(path), sha256=sha(path))

            def ref(path):
                return dict(path=str(path), sha256=sha(path))

            candidate = save("candidate.fake", {})
            native_ref = save("native.fake", {})
            c = dict(
                phase=PHASE,
                generation=0,
                updates=0,
                seed=123,
                operator_end_epoch=1000.0,
                prior_helper_sha256="p",
                lineage_origin=dict(
                    phase=PHASE, seed=123, prior_sha256="p", teacher_labels_used=False
                ),
                execution_scope_schema="human-prior-zero-initialization-contract-v1",
                initializer_seal={},
                inference_source_sha256={},
                source_sha256={
                    str(directory / n): sha(directory / n)
                    for n in ("model.py", "native.py", "initialize.py")
                },
            )
            contract_ref = save("contract.json", c)
            commands = []
            rows = []
            for i in range(2):
                log = save(f"load-{i}.json", dict(status="PASS-strict-native-readonly", step=0))
                commands.append(
                    dict(
                        command=[
                            sys.executable,
                            str(directory / "initialize.py"),
                            "--audit-only",
                            "--contract",
                            contract_ref["path"],
                            "--native",
                            native_ref["path"],
                            "--native-sha256",
                            native_ref["sha256"],
                        ],
                        returncode=0,
                        first=101.0,
                        finished=102.0,
                        log_sha256=log["sha256"],
                    )
                )
                rows.append(
                    dict(**native_ref, step=0, actual_fresh_process=True, log_path=log["path"])
                )
            result = dict(
                schema="human-prior-zero-initialization-result-v1",
                status="PASS-literal-zero-parent-and-two-fresh-native-loads-not-strength",
                contract_sha256=contract_ref["sha256"],
                candidate=candidate,
                native=native_ref,
                source_sha256=c["source_sha256"],
                teacher_labels_used=False,
                optimizer_updates=0,
                first=100.0,
                finished=103.0,
                deadline=600.0,
                commands=commands,
                native_payloads=rows,
            )
            result_ref = save("result.json", result)
            spec = dict(
                generation=1,
                seed=123,
                parent_initialization_result=result_ref,
                parent_contract=contract_ref,
                parent_candidate=candidate,
                parent_native=native_ref,
                parent_native_helper=ref(directory / "native.py"),
                parent_model_helper=ref(directory / "model.py"),
            )
            self.assertEqual(validate_zero_metadata(spec, c), c)
            result["commands"][0]["command"][1] = "wrong.py"
            spec["parent_initialization_result"] = save("corrupted-result.json", result)
            with self.assertRaisesRegex(ValueError, "argv"):
                validate_zero_metadata(spec, c)

    def test_direct_collection_parent_search_and_source_tamper_rejected(self):
        import hashlib

        from parent_bridge import canonical, require_collection_parent

        directory = Path("/synthetic-original-inference")
        helper = dict(
            directory=str(directory),
            model_sha256="m",
            native_sha256="n",
            evaluator_sha256="e",
            prior_path=str(directory / "value.py"),
            prior_sha256="p",
            extension_path=str(directory / "extension.so"),
            extension_sha256="x",
        )
        c = dict(
            seed=7,
            generation=0,
            search_helper=dict(path="/original/search", sha256="s"),
            inference_source_sha256={
                str(directory / name): digest
                for name, digest in (
                    ("model.py", "m"),
                    ("native.py", "n"),
                    ("evaluator.py", "e"),
                    ("value.py", "p"),
                    ("extension.so", "x"),
                )
            },
        )
        seal = dict(parent_candidate=dict(path="/actual/zero-candidate", sha256="z"))
        reg = dict(
            seed=7,
            generation=1,
            search_helper=c["search_helper"],
            parent_helpers=helper,
            parent_candidate={
                **seal["parent_candidate"],
                "contract_sha256": hashlib.sha256(canonical(c)).hexdigest(),
            },
        )
        require_collection_parent(reg, seal, c)
        for part in ("search", "candidate", "source"):
            bad = copy.deepcopy(reg)
            if part == "search":
                bad["search_helper"]["sha256"] = "other"
            if part == "candidate":
                bad["parent_candidate"]["path"] = "/claimed-other"
            if part == "source":
                bad["parent_helpers"]["evaluator_sha256"] = "other"
            with self.assertRaises(ValueError):
                require_collection_parent(bad, seal, c)
        # Exercise the ACTUAL runtime entrypoint: reject tampering BEFORE any helper code load.
        from unittest.mock import patch

        import run_collection

        bad = copy.deepcopy(reg)
        bad["parent_admission_seal"] = {"path": "/unit/seal", "sha256": "unit"}
        bad["parent_admission_result"] = {"path": "/unit/result", "sha256": "unit"}
        bad["search_helper"]["sha256"] = "other"
        with (
            patch("parent_bridge.validate_admission_result", return_value=(seal, c)),
            patch(
                "run_collection.load", side_effect=AssertionError("must reject before code loading")
            ),
            self.assertRaisesRegex(ValueError, "SAME"),
        ):
            run_collection.parent(bad)

    def test_typed_ancestry_never_teacher_as_zero(self):
        c = dict(
            seed=7,
            prior_helper_sha256="p",
            lineage_origin=dict(phase=PHASE, seed=7, prior_sha256="p", teacher_labels_used=False),
        )
        origin(c)
        for key, value in [
            ("phase", "teacher-bootstrap"),
            ("seed", 8),
            ("teacher_labels_used", True),
        ]:
            broken = copy.deepcopy(c)
            broken["lineage_origin"][key] = value
            with self.assertRaises(ValueError):
                origin(broken)


if __name__ == "__main__":
    unittest.main()

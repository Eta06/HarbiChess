"""Pure receipt/lineage fixtures, not actual model admission qualification."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import parent_bridge as b


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.serial = 0

    def put(self, body, binary=False):
        self.serial += 1
        path = self.root / str(self.serial)
        path.write_bytes(body if binary else json.dumps(body).encode())
        return {"path": str(path), "sha256": b.sha(path)}

    def fixture(self):
        helper, model, candidate, data = [self.put(b"x", True) for _ in range(4)]
        provenance = self.put({"own": True})
        contract = dict(
            phase="own-learning",
            updates=64,
            seed=20262905,
            math={"fixed": True},
            feature_schema="fixed16",
            bootstrap_candidate_sha256="1" * 64,
            dataset_sha256=data["sha256"],
            prior_helper_sha256="2" * 64,
            teacher_labels_used_in_own_phase=False,
            operator_end_epoch=1000,
            source_sha256={
                helper["path"]: helper["sha256"],
                model["path"]: model["sha256"],
            },
            raw_collection_inputs={},
            execution_helpers_sha256={},
            inference_source_sha256={},
            target_provenance_path=provenance["path"],
            target_provenance_sha256=provenance["sha256"],
        )
        contract_ref = self.put(contract)
        proof_contract = copy.deepcopy(contract)
        proof_contract_ref = self.put(proof_contract)

        def result(mode, steps, c):
            rows = []
            commands = []
            for step in steps:
                ref = self.put(b"native", True)
                log = self.put(dict(status="PASS-strict-native-readonly", step=step))
                rows.append(
                    {
                        **ref,
                        "step": step,
                        "actual_fresh_process": True,
                        "log_path": log["path"],
                    }
                )
                commands.append(
                    dict(
                        command=[
                            "python",
                            "train.py",
                            "--resume",
                            ref["path"],
                            "--resume-sha256",
                            ref["sha256"],
                            "--stop",
                            str(step),
                            "--audit-only",
                        ],
                        returncode=0,
                        finished=19,
                        log_sha256=log["sha256"],
                    )
                )
            return dict(
                commands=commands,
                status=b.STATUS,
                mode=mode,
                own_updates=8 if mode == "proof" else 64,
                first=10,
                finished=20,
                deadline=30,
                weights_only_initializer={"sha256": "1" * 64},
                contract_sha256=c["sha256"],
                native_payloads=rows,
                full_payload_bits_equal=True,
            )

        proof = result("proof", [0, 8, 0, 4, 4, 8], proof_contract_ref)
        fit = result("fresh-fit", [0, 64], contract_ref)
        return dict(
            schema="NNUE-own-generation-parent-admission-seal-v3",
            status="registered",
            generation=2,
            seed=20262905,
            weights_only=True,
            parent_contract=contract_ref,
            parent_proof_contract=proof_contract_ref,
            parent_proof_result=self.put(proof),
            parent_fit_result=self.put(fit),
            parent_native={k: fit["native_payloads"][1][k] for k in ("path", "sha256")},
            parent_native_helper=helper,
            parent_model_helper=model,
            parent_candidate=candidate,
            parent_dataset=data,
        )

    def test_closed_historical_clock_is_not_new_training_clock(self):
        spec = self.fixture()
        self.assertEqual(b.validate_metadata(spec)["updates"], 64)
        # Historical timestamps long expired; no comparison to time.now and no rewriting.
        self.assertEqual(b.read(spec["parent_fit_result"])["deadline"], 30)

    def test_full_resume_cannot_be_claimed_or_generation_skipped(self):
        for key, value in [
            ("weights_only", False),
            ("generation", 3),
            ("seed", 20262906),
        ]:
            spec = self.fixture()
            spec[key] = value
            with self.assertRaises(ValueError):
                b.validate_metadata(spec)

    def test_changed_data_source_and_binary_rejected(self):
        for key in ("parent_native", "parent_dataset", "parent_model_helper"):
            spec = self.fixture()
            Path(spec[key]["path"]).write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                b.validate_metadata(spec)

    def test_proof_failure_or_missing_fresh_load_rejected(self):
        for mutate in (
            lambda r: r.update(full_payload_bits_equal=False),
            lambda r: r["native_payloads"].pop(),
            lambda r: r.update(finished=31),
        ):
            spec = self.fixture()
            r = b.read(spec["parent_proof_result"])
            mutate(r)
            spec["parent_proof_result"] = self.put(r)
            with self.assertRaises(ValueError):
                b.validate_metadata(spec)

    def test_admission_uses_original_contract_and_exact_candidate_bits(self):
        spec = self.fixture()
        contract = b.read(spec["parent_contract"])
        calls = []
        state = {"embedding": b"\x00\x80"}
        learner = SimpleNamespace(step=64, native=lambda: {"model": state})

        def load(path, actual_contract):
            calls.append((str(path), actual_contract))
            return learner

        native = SimpleNamespace(
            __file__=spec["parent_native_helper"]["path"],
            MODEL_SCHEMA="fixed-model",
            load_native=load,
            bits_equal=lambda a, c: a == c,
            validate_weights=lambda _: None,
        )
        packet = {"schema": "fixed-model", "contract": contract, "model": state}
        weights, receipt = b.admit(spec, native, lambda _: packet)
        self.assertEqual(weights, state)
        self.assertEqual(calls, [(spec["parent_native"]["path"], contract)])
        self.assertFalse(receipt["outgoing_bridge_full_resume"])
        packet["model"] = {"embedding": b"\x00\x00"}
        with self.assertRaises(ValueError):
            b.admit(spec, native, lambda _: packet)

    def test_teacher_cannot_be_admitted_as_own_final(self):
        spec = self.fixture()
        c = b.read(spec["parent_contract"])
        c["phase"] = "teacher-bootstrap"
        spec["parent_contract"] = self.put(c)
        with self.assertRaises(ValueError):
            b.validate_metadata(spec)


if __name__ == "__main__":
    unittest.main()

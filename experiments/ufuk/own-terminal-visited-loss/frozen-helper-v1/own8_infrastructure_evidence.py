"""Source7 proof admission; ancestral source428 receipts never qualify new source."""

from pathlib import Path

from qualify_source_cli import FILES, LEDGER, expected_artifacts


def check_unit_receipt(receipt, source, cases):
    assert len(cases) == len(set(cases)) and len(cases) > 0
    assert receipt["source_commit"] == source
    assert receipt["schema"] == "ufuk-clean-producer-actualCUDA-unit-suite-v1"
    assert receipt["status"] == f"pass-actualCUDA{len(cases)}-no-skip"
    assert receipt["cases"] == cases
    assert receipt["tests_passed"] == len(cases)
    assert receipt["tests_skipped"] == receipt["tests_failed"] == 0
    assert receipt["actualCUDA"] is True and receipt["source_clean"] is True
    assert "A100" in receipt["device_name"] and str(receipt["torch_version"]).startswith("2.11.")


def check_cli_receipt(receipt, directory, source, sha):
    assert receipt["schema"] == "actual-CUDA-certificate-ledger-CLI-qualification-v1"
    assert receipt["status"] == "pass" and receipt["source_commit"] == source
    assert receipt["expected_ledger_schema"] == LEDGER
    assert set(receipt["byte_exact_artifacts"]) == set(expected_artifacts())
    assert receipt["both_all_native_epochs_freshprocess_strictload"] == [0, 1, 2]
    assert receipt["per_run_epochs"] == 2
    assert receipt["started_epoch"] < receipt["finished_epoch"] < receipt["absolute_deadline_epoch"]
    assert receipt["absolute_deadline_epoch"] - receipt["original_started_epoch"] == 600
    for name, digest in receipt["byte_exact_artifacts"].items():
        for arm in ("whole", "split"):
            assert sha(Path(directory) / arm / name) == digest
    for name, digest in receipt["native_manifest_sha256"].items():
        arm, epoch = name.split("/")
        assert sha(Path(directory) / arm / "checkpoints" / epoch / "checkpoint.json") == digest
    assert len(receipt["native_manifest_sha256"]) == 6 and len(FILES) == 6
    assert all(
        receipt[name] > 0
        for name in (
            "last_epoch_policy_target_rows",
            "last_epoch_known_terminal_rows",
            "last_epoch_UNKNOWN_excluded_value_rows",
        )
    )

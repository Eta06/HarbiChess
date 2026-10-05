"""Exact published new-producer curriculum CLI evidence; no hardware job is started."""

import hashlib
import json
from pathlib import Path

SOURCE = "428a30f5e1658f3cf159844db547ff0147ade5a9"
BOOK = "1a5ca17664a828d669e58cf2bd5d9eeb7f980b83f20d3a4031d36e4ff0930ccb"
E8 = "e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03"
PAYLOADS = (
    "model.safetensors",
    "base.safetensors",
    "behavior.safetensors",
    "training.pt",
    "actor.json",
    "last-frozen-epoch.json.gz",
)


def check_cli_receipt(receipt, run, sha):
    assert receipt["schema"] == "actual-CUDA-search-acting-CLI-qualification-v2"
    assert receipt["status"] == "pass" and receipt["source_commit"] == SOURCE
    assert receipt["input_sha256"]["weights"] == E8
    assert receipt["input_sha256"]["book"] == BOOK
    assert receipt["both_final_full_native_freshprocess_strictload"] is True
    assert receipt["per_run_epochs"] == 2
    assert receipt["finished_epoch"] <= receipt["absolute_deadline_epoch"]
    for name in (
        "last_epoch_policy_target_rows",
        "last_epoch_known_terminal_rows",
        "last_epoch_UNKNOWN_excluded_value_rows",
    ):
        assert receipt[name] > 0
    expected = {f"journal/epoch-{i:08d}.json.gz" for i in (1, 2)}
    expected |= {f"checkpoints/epoch-00000002/{name}" for name in PAYLOADS}
    assert set(receipt["byte_exact_artifacts"]) == expected
    run = Path(run)
    for relative, digest in receipt["byte_exact_artifacts"].items():
        for arm in ("whole", "split"):
            assert sha(run / arm / relative) == digest
    for arm in ("whole", "split"):
        native = json.loads((run / arm / "checkpoints/epoch-00000002/checkpoint.json").read_text())
        assert native["schema"] == "torch-search-acting-native-cuda-v2"
        assert native["source_commit"] == SOURCE
        assert native["runtime"]["torch"] == "2.11.0+cu130"
        assert native["inputs"]["book"]["sha256"] == BOOK
        assert set(native["artifacts"]) == set(PAYLOADS)
        for filename, digest in native["artifacts"].items():
            assert sha(run / arm / "checkpoints/epoch-00000002" / filename) == digest


CUDA34_CASES = (
    (
        "tests/test_search_acting_stage.py::test_schedule_before_outcomes_uses_separ"
        "ate_rng_and_both_parities"
    ),
    (
        "tests/test_search_acting_stage.py::test_unknown_and_illegal_zero_gradient_t"
        "erminal_white_black_pov"
    ),
    "tests/test_search_acting_stage.py::test_actual_e8_native_partial_rejection_and_unused_storage",
    (
        "tests/test_search_acting_stage.py::test_real_fresh_process_two_vs_pause_one"
        "_resume_two_native_bytes[cpu]"
    ),
    (
        "tests/test_search_acting_stage.py::test_real_fresh_process_two_vs_pause_one"
        "_resume_two_native_bytes[cuda:0]"
    ),
    (
        "tests/test_search_acting_stage.py::test_rejected_pass_restores_model_adam_g"
        "lobals_and_both_sampler_rngs[cpu]"
    ),
    (
        "tests/test_search_acting_stage.py::test_rejected_pass_restores_model_adam_g"
        "lobals_and_both_sampler_rngs[cuda:0]"
    ),
    "tests/test_search_acting_stage.py::test_closed_ledger_schedule_tamper_rejected",
    (
        "tests/test_search_acting_stage.py::test_zero_search_selection_still_trains_"
        "fresh_known_outcomes"
    ),
    (
        "tests/test_search_acting_stage.py::test_known_terminal_root_never_calls_net"
        "work_both_movers[7k/6Q1/5K2/8/8/8/8/8 b - - 0 1]"
    ),
    (
        "tests/test_search_acting_stage.py::test_known_terminal_root_never_calls_net"
        "work_both_movers[8/8/8/8/8/5k2/6q1/7K w - - 0 1]"
    ),
    (
        "tests/test_search_acting_stage.py::test_same_record_policy_CE_zero_ablation"
        "_changes_only_policy_term"
    ),
    "tests/test_search_acting_stage.py::test_preaction_masked_search_and_actual_mu_raw_pi_archive",
    "tests/test_search_acting_stage.py::test_v1_native_and_non_e8_fresh_rejected",
    (
        "tests/test_search_acting_stage.py::test_completed_own_outcomes_and_caps_use"
        "_rules_not_search_value"
    ),
    (
        "tests/test_search_acting_stage.py::test_real_e8_raw_reference_kl_zero_despi"
        "te_search_mu_at_same_point02_gate"
    ),
    (
        "tests/test_torch_array_encoder.py::test_all104_float32_storage_both_perspec"
        "tives_history_and_rules[rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq "
        "- 0 1-e2e4 a7a6 e4e5 d7d5 e5d6]"
    ),
    (
        "tests/test_torch_array_encoder.py::test_all104_float32_storage_both_perspec"
        "tives_history_and_rules[r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1-e1g1 e8c8]"
    ),
    (
        "tests/test_torch_array_encoder.py::test_all104_float32_storage_both_perspec"
        "tives_history_and_rules[7k/P7/8/8/8/8/8/7K w - - 0 1-a7a8n]"
    ),
    (
        "tests/test_torch_array_encoder.py::test_all104_float32_storage_both_perspec"
        "tives_history_and_rules[7k/8/8/8/8/8/8/R6K w - - 99 1-a1a2 h8g8]"
    ),
    (
        "tests/test_torch_array_encoder.py::test_all104_float32_storage_both_perspec"
        "tives_history_and_rules[rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq "
        "- 0 1-g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8]"
    ),
    "tests/test_torch_array_encoder.py::test_real_e8_masked_cpu_outputs_storage_and_rng_unchanged",
    (
        "tests/test_torch_array_encoder.py::test_private_array_learner_counterfactua"
        "l_and_fresh_resume_all_payloads[cpu]"
    ),
    (
        "tests/test_torch_array_encoder.py::test_private_array_learner_counterfactua"
        "l_and_fresh_resume_all_payloads[cuda:0]"
    ),
    "tests/test_opening_validation_dedup.py::test_duplicate_metadata_still_validated[fen]",
    "tests/test_opening_validation_dedup.py::test_duplicate_metadata_still_validated[root_ply]",
    "tests/test_opening_validation_dedup.py::test_duplicate_metadata_still_validated[source_game]",
    "tests/test_opening_validation_dedup.py::test_illegal_duplicate_not_cached_as_valid",
    (
        "tests/test_opening_validation_dedup.py::test_fullhistory_not_fen_identity_a"
        "nd_claim_repetition"
    ),
    "tests/test_opening_validation_dedup.py::test_order_aliases_rng_restore_and_board_isolation",
    "tests/test_opening_validation_dedup.py::test_invalid_root_is_not_hidden_by_other_valid_root",
    "tests/test_opening_validation_dedup.py::test_duplicate_terminal_state_rejected",
    (
        "tests/test_opening_validation_dedup.py::test_duplicate_special_state_preser"
        "ves_exact_legal_history[moves0]"
    ),
    (
        "tests/test_opening_validation_dedup.py::test_duplicate_special_state_preser"
        "ves_exact_legal_history[moves1]"
    ),
)


def check_unit34_receipt(receipt):
    assert receipt["schema"] == "ufuk-clean-producer-actualCUDA-unit-suite-v1"
    assert receipt["status"] == "pass-actualCUDA34-no-skip"
    assert receipt["source_commit"] == SOURCE and receipt["source_clean"] is True
    assert receipt["tests_passed"] == 34
    assert receipt["tests_skipped"] == receipt["tests_failed"] == 0
    assert receipt["actualCUDA"] is True
    assert "A100" in receipt["device_name"]
    assert receipt["torch_version"] == "2.11.0+cu130" and receipt["cuda_version"] == "13.0"
    assert receipt["initial_e8_sha256"] == E8
    assert tuple(receipt["cases"]) == CUDA34_CASES
    assert receipt["stderr_bytes"] == 0
    assert hashlib.sha256(receipt["stdout"].encode()).hexdigest() == receipt["stdout_sha256"]
    assert "34 passed" in receipt["stdout"] and "skipped" not in receipt["stdout"]
    assert "failed" not in receipt["stdout"]
    assert receipt["whole_seconds"] > 0

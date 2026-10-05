import importlib.util
import json
import sys
from pathlib import Path

import torch
from train import bind_common_rows, extract_full_history_rows, load_common_trainer, load_pinned, sha

ROOT = Path("/workspace/HarbiChess")
SOURCE = ROOT / "src"
RUN = Path("/workspace/work/harbichess/cpu-fresh-selfplay-v2-actual/registration")
JOURNAL_HELPER = ROOT / "experiments/ufuk/cpu-fresh-selfplay-v2/journal_v2.py"
FEATURE_HELPER = ROOT / "experiments/ufuk/cpu-fresh-learning-v1/features.py"
COMMON = ROOT / "experiments/ufuk/cpu-fresh-learning-v1/train.py"
SEARCH = Path(
    "/workspace/work/harbichess/cpu-own-search-consistency-proposal/own_search_consistency.py"
)
ARRAY = ROOT / "src/harbichess/training/torch_array_encoder.py"
EXPECTED = {
    20262805: "8109c074a9e084f33ec8d9a6c4b062eb3c57e8b6e8d897e03bd39dc8261dea8f",
    20262806: "df515e393e97065cd4828ceacccc7628ea04091643c16296009f48f1b9e01645",
}

sys.path.insert(0, str(SOURCE))
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
feature_module = load_pinned(FEATURE_HELPER, sha(FEATURE_HELPER), "smoke_features")
common = load_common_trainer(COMMON, sha(COMMON), FEATURE_HELPER, sha(FEATURE_HELPER))
journal = load_pinned(JOURNAL_HELPER, sha(JOURNAL_HELPER), "smoke_journal")
search = load_pinned(SEARCH, sha(SEARCH), "smoke_search")
encoder_spec = importlib.util.spec_from_file_location("smoke_array_encoder", ARRAY)
encoder_module = importlib.util.module_from_spec(encoder_spec)
sys.modules[encoder_spec.name] = encoder_module
encoder_spec.loader.exec_module(encoder_module)
results = []
for seed in (20262805, 20262806):
    journal_path = Path(f"/dev/shm/harbichess-fresh-E0-{seed}/actions-00002048.json.gz")
    config_path = RUN / f"{seed}-E0-actor-config.json"
    journal_sha = sha(journal_path)
    if journal_sha != EXPECTED[seed]:
        raise ValueError("fixed 2048 journal SHA changed")
    cfg = json.loads(config_path.read_text())
    common_data = common.prepare_fresh(
        journal_path,
        journal_sha,
        config_path,
        sha(config_path),
        JOURNAL_HELPER,
        sha(JOURNAL_HELPER),
        FEATURE_HELPER,
        sha(FEATURE_HELPER),
        protected_position_keys=cfg["excluded_training_position_keys"],
    )
    extracted = search.extract_verified(
        journal_path, journal_sha, cfg, journal, FEATURE_HELPER, sha(FEATURE_HELPER)
    )
    binding = search.bind_common_data(extracted, common_data)
    state = journal.read(journal_path)
    dense = extract_full_history_rows(
        state,
        cfg,
        journal,
        encoder_module.TorchArrayBoardEncoder(),
        feature_module.invariants,
        len(common_data[1]),
    )
    common_binding = bind_common_rows(common_data, dense, binding)
    results.append(
        {
            "seed": seed,
            "journal_sha256": journal_sha,
            "actor_config_sha256": sha(config_path),
            "actor_original_deadline_epoch": cfg["original_deadline_epoch"],
            "actor_actions": state["actions"],
            "known_rows": len(common_data[1]),
            "complete_trajectory_groups": len(common_data[3]),
            "train_rows": len(common_data[7]),
            "validation_rows": len(common_data[8]),
            "unknown_actions": state["actions"] - len(common_data[1]),
            "fullhistory_dense_shape": list(dense["x104"].shape),
            "common_dataset_sha256": common_data[10],
            "search_extended_data_sha256": binding["search_extended_data_sha256"],
            "row_binding_sha256": common_binding["search_rows_sha256"],
            "teacher_labels": False,
            "NN_forward_or_fit_run": False,
        }
    )
receipt = {
    "status": "PASS-actual-2048-data-loader-only",
    "results": results,
    "scope": (
        "read-only CPU data conversion and row-binding smoke; "
        "no E8 forward, fitting, match, or strength query"
    ),
}
Path(__file__).with_name("actual-2048-loader-smoke.json").write_text(
    json.dumps(receipt, sort_keys=True, indent=2) + "\n"
)
print(json.dumps(receipt, sort_keys=True))

"""TEST ONLY explicit synthetic common-data seam; not real-journal CLI qualification.

Executes production CLI/update/checkpoint/native-load paths. Does not generate
selfplay or invoke a neural actor/search. Its synthetic protocol cannot match any
real-data native contract and is never a strength candidate.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import train

CONFIG = None


def common_data(*args, **kwargs):
    global CONFIG
    CONFIG = json.loads(Path(args[2]).read_text())
    x = np.zeros((1024, 840), dtype=np.float32)
    x[:, 0] = 1
    x[:, 833] = 1
    y = np.arange(1024, dtype=np.int64) % 3
    base = np.tile(np.array([0.2, 0.3, 0.5], dtype=np.float32), (1024, 1))
    groups = tuple(tuple(range(i * 64, (i + 1) * 64)) for i in range(16))
    trajectories = tuple(f"{i:064x}" for i in range(16))
    train_groups, val_groups = tuple(range(8)), tuple(range(8, 16))
    train_rows, val_rows = tuple(range(512)), tuple(range(512, 1024))
    receipts = [
        {
            "source_config_sha256": train.sha(Path(args[2])),
            "test_scope": "SYNTHETIC-COMMON-DATA-SEAM-NOT-REAL-ACTOR-DATA",
        }
    ]
    data_sha = hashlib.sha256(x.tobytes() + y.tobytes() + base.tobytes()).hexdigest()
    return (
        torch.from_numpy(x),
        torch.from_numpy(y),
        torch.from_numpy(base),
        groups,
        trajectories,
        train_groups,
        val_groups,
        train_rows,
        val_rows,
        receipts,
        data_sha,
        CONFIG,
    )


def search_columns(*args, **kwargs):
    # Values are test columns, not alleged real E8 search/anchor inference.
    config_path = Path(sys.argv[sys.argv.index("--actor-config") + 1])
    common = common_data(None, None, config_path)
    raw = np.linspace(-2.0, 2.0, 1024, dtype=np.float64)
    normalized = np.clip(raw, -1, 1).astype(np.float32)
    return {
        "x": common[0].numpy(),
        "y": common[1].numpy(),
        "anchors": common[2].numpy(),
        "raw_search": raw,
        "normalized_search": normalized,
        "rows": [{"synthetic_row": i, "selected_value": float(raw[i])} for i in range(1024)],
        "receipt": {
            "config_sha256": hashlib.sha256(train.canonical(CONFIG)).hexdigest(),
            "test_scope": "SYNTHETIC-COMMON-DATA-SEAM-NOT-REAL-DATA",
        },
    }


if __name__ == "__main__":
    train.prepare_fresh = common_data
    train.extract_verified = search_columns
    train.main()

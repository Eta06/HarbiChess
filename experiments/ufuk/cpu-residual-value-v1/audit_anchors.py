"""Prospective fixed64 full-history e8 anchor checks; no optimizer or strength query."""

import argparse
import gzip
import hashlib
import json
import random
import time
from pathlib import Path

import chess
import numpy as np
import torch

from harbichess.backends.torch_network import load_weights
from harbichess.training.cgroup_budget import CgroupMemoryBudget
from harbichess.training.torch_array_encoder import TorchArrayBoardEncoder
from harbichess.training.torch_search_acting_run import clean_source


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--first-epoch", type=float, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    clean_source(protocol["source_commit"])
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        raise ValueError("CPU only")
    first, end = args.first_epoch, args.first_epoch + 900
    assert first <= time.time() < end
    assert sha(args.weights) == protocol["initial_e8_sha256"]
    sampler = random.Random(20262765)
    model = load_weights(args.weights).eval().requires_grad_(False)
    encoder = TorchArrayBoardEncoder()
    budget = CgroupMemoryBudget(15 * 1024**3)
    rows = []
    try:
        for record in protocol["journals"]:
            if time.time() >= end:
                raise TimeoutError("original900s anchorclock expired")
            budget.check()
            path = Path(record["path"])
            assert sha(path) == record["sha256"]
            data = json.loads(gzip.decompress(path.read_bytes()))
            canonical = json.dumps(
                {k: v for k, v in data.items() if k != "sample_chain_sha256"},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
            assert (
                hashlib.sha256(
                    bytes.fromhex(data["previous_sample_chain_sha256"]) + canonical
                ).hexdigest()
                == data["sample_chain_sha256"]
            )
            actions = data["collection"]["actions"]
            indices = sorted(sampler.sample(range(len(actions)), 8))
            for index in indices:
                row = actions[index]
                pre = row["transition"]["pre"]
                board = chess.Board(pre["root_fen"])
                assert board.is_valid()
                for uci in pre["moves"]:
                    move = chess.Move.from_uci(uci)
                    assert board.is_legal(move)
                    board.push(move)
                x = torch.from_numpy(encoder.encode_board(board).values.copy()).reshape(
                    1, 8, 8, 104
                )
                with torch.inference_mode():
                    logits = model.masked_policy_value(x, torch.zeros((1, 1), dtype=torch.long))[1]
                    actual = torch.softmax(logits, 1)[0].numpy()
                stored = np.asarray(row["base_wdl"], dtype=np.float32)
                error = float(np.max(np.abs(actual - stored)))
                assert error <= 3e-5, (record["source_seed"], record["epoch"], index, error)
                rows.append(
                    {
                        "journal_sha256": record["sha256"],
                        "source_seed": record["source_seed"],
                        "epoch": record["epoch"],
                        "action_index": index,
                        "fullhistory_sha256": hashlib.sha256(
                            json.dumps(pre, sort_keys=True, separators=(",", ":")).encode()
                        ).hexdigest(),
                        "maximum_probability_absolute_error": error,
                    }
                )
            assert sha(path) == record["sha256"]
            del canonical, actions, data
        assert len(rows) == 64
        status = "PASS-fixed64-fullhistory-anchor-not-strength"
    except Exception as exc:
        status = "failed-preserved"
        rows.append({"error": repr(exc)})
    result = {
        "status": status,
        "rows": rows,
        "helper_sha256": sha(Path(__file__)),
        "protocol_sha256": sha(args.protocol),
        "e8_sha256": sha(args.weights),
        "original_first_epoch": first,
        "original_deadline_epoch": end,
        "finished_epoch": time.time(),
        "GPU_used": False,
        "scope": (
            "64 preregistered sampled fullhistories; "
            "not exhaustive old131072-row anchor recomputation"
        ),
        "strength_success_claimed": False,
    }
    with args.output.open("x") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    print(json.dumps({"status": status, "rows": len(rows)}), flush=True)
    if status.startswith("failed"):
        raise RuntimeError(rows[-1])


if __name__ == "__main__":
    main()

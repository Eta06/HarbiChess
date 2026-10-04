"""Explicit BOTH fixed OWN-v1 CURRENT_E sameweights full/masked18history parity.

Modes actualCUDA vs TorchCPU, or MLXCPU vs TorchCPU; no hardware simulation.
"""

import argparse
import json
import time
from pathlib import Path

import chess
import numpy as np
import torch
from ownv1_strength_config import SEEDS, bind_cli, validate_binding

from harbichess.backends.torch_network import load_weights, sha256
from harbichess.chess.actions import legal_action_indices
from harbichess.chess.encoding import BoardEncoder
from harbichess.training.torch_online_learner import _prepare_device

Q = {}


PROBE_SHA = "1089fd0cca308c24bb040a840352d0ad18aa85a7bdcd17608133456bf08bed9f"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--probes", type=Path, required=True)
    p.add_argument("--backend", choices=("cuda", "mlx"), required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    validate_binding(a, Q)
    started = time.time()
    _prepare_device("cuda:0" if a.backend == "cuda" else "cpu")
    if a.backend == "cuda":
        assert torch.cuda.is_available(), "Actual CUDA required; no fallback"
    else:
        import mlx.core as mx

        from harbichess.backends.pairwise_network import HarbiChessPairwiseNetwork

        mx.set_default_device(mx.cpu)
    assert sha256(a.probes) == PROBE_SHA
    spec = json.loads(a.manifest.read_text())
    assert [entry["seed"] for entry in spec["seeds"]] == list(SEEDS)
    boards = []
    for row in json.loads(a.probes.read_text())["positions"]:
        board = chess.Board(row["root_fen"])
        for uci in row["moves"]:
            move = chess.Move.from_uci(uci)
            assert move in board.legal_moves
            board.push(move)
        assert board.fen() == row["fen"]
        boards.append(board)
    assert len(boards) == 18
    encoded = torch.tensor(
        [BoardEncoder().encode_board(board).values for board in boards], dtype=torch.float32
    ).reshape(-1, 8, 8, 104)
    reports = []
    with torch.inference_mode():
        for entry in spec["seeds"]:
            path = Path(entry["candidate"])
            assert sha256(path) == entry["candidate_sha256"]
            cpu = load_weights(path).eval()
            other = (
                load_weights(path).to("cuda:0").eval()
                if a.backend == "cuda"
                else HarbiChessPairwiseNetwork.from_portable(path)
            )
            other.eval()
            ca = cpu(encoded)
            ga = (
                other(encoded.to("cuda:0"))
                if a.backend == "cuda"
                else other(mx.array(encoded.numpy()))
            )
            if a.backend == "mlx":
                mx.eval(ga)
            errors = []
            for x, y in zip(ca, ga, strict=True):
                xn = x.numpy()
                yn = y.cpu().numpy() if a.backend == "cuda" else np.array(y)
                assert np.isfinite(xn).all() and np.isfinite(yn).all()
                np.testing.assert_allclose(xn, yn, atol=2e-05, rtol=2e-05)
                errors.append(float(np.max(np.abs(xn - yn))))
            masked = []
            for index, board in enumerate(boards):
                actions = torch.tensor([legal_action_indices(board)])
                ca = cpu.masked_policy_value(encoded[index : index + 1], actions)
                ga = (
                    other.masked_policy_value(
                        encoded[index : index + 1].to("cuda:0"), actions.to("cuda:0")
                    )
                    if a.backend == "cuda"
                    else other.masked_policy_value(
                        mx.array(encoded[index : index + 1].numpy()), mx.array(actions.numpy())
                    )
                )
                if a.backend == "mlx":
                    mx.eval(ga)
                for x, y in zip(ca, ga, strict=True):
                    xn = x.numpy()
                    yn = y.cpu().numpy() if a.backend == "cuda" else np.array(y)
                    assert np.isfinite(xn).all() and np.isfinite(yn).all()
                    np.testing.assert_allclose(xn, yn, atol=2e-05, rtol=2e-05)
                    masked.append(float(np.max(np.abs(xn - yn))))
            assert sha256(path) == entry["candidate_sha256"]
            reports.append(
                {
                    "seed": entry["seed"],
                    "weights_sha256": sha256(path),
                    "full4672_policy_logit_max_error": errors[0],
                    "full_value_logit_max_error": errors[1],
                    "legal_masked_policy_and_value_max_error": max(masked),
                }
            )
    receipt = {
        "schema": "ufuk-ownv1-final-portable18parity-v1",
        "status": "pass-sameweights-full-and-legalmasked18fullhistories",
        "backend": a.backend,
        "device": "cuda:0 actual" if a.backend == "cuda" else "MLXCPU actual; AppleMetal untested",
        "models": reports,
        "probes_sha256": PROBE_SHA,
        "manifest_sha256": sha256(a.manifest),
        "script_sha256": sha256(Path(__file__)),
        "atol": 2e-05,
        "rtol": 2e-05,
        "torch_version": torch.__version__,
        "whole_seconds_after_import": time.time() - started,
        "scope": (
            "Raw policy/value logits and legal masked heads parity; not stren"
            "gth, speed or game outcome evidence."
        ),
    }
    receipt.update(
        source_commit=Q["source_commit"],
        fixed_epochs=Q["fixed_epochs"],
        qualification_ledger_slot=4,
    )
    with a.output.open("x") as f:
        json.dump(receipt, f, indent=2)
        f.write("\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    bind_cli(globals())
    main()

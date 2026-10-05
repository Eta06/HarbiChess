"""Explicit BOTH fixed SEARCH-ACTING-v2 CURRENT_E sameweights full/masked18history parity.

Modes actualCUDA vs TorchCPU, or MLXCPU vs TorchCPU; no hardware simulation.
"""

import argparse
import json
import time
from pathlib import Path

import chess
import numpy as np
import torch
from cpu_contingency_strength_config import SEEDS, bind_cli, validate_binding

from harbichess.backends.torch_network import load_weights, sha256
from harbichess.chess.actions import legal_action_indices
from harbichess.chess.encoding import BoardEncoder
from harbichess.training.torch_online_learner import _prepare_device

Q = {}


PROBE_SHA = "1089fd0cca308c24bb040a840352d0ad18aa85a7bdcd17608133456bf08bed9f"


def restore_native(entry):
    from cpu_contingency_adapter_controls import validate_spec

    from harbichess.selfplay.online_actor import OnlineActorConfig
    from harbichess.training.ownsearch_targets import OwnSearchConfig
    from harbichess.training.torch_fullgame_ppo import FullGamePPOTrainConfig
    from harbichess.training.torch_ownsearch_core import OwnSearchObjective
    from harbichess.training.torch_search_acting_learner import (
        TorchSearchActingConfig,
        TorchSearchActingLearner,
    )

    manifest_path = Path(entry["native_audit_manifest"])
    assert sha256(manifest_path) == entry["native_audit_manifest_sha256"]
    manifest = json.loads(manifest_path.read_text())
    validate_spec(manifest)
    cfg = dict(manifest["frozen_config"])
    for key, cls in (
        ("actors", OnlineActorConfig),
        ("search", OwnSearchConfig),
        ("objective", OwnSearchObjective),
        ("schedule", FullGamePPOTrainConfig),
    ):
        cfg[key] = cls(**cfg[key])
    config = TorchSearchActingConfig(**cfg)
    assert config.seed == entry["seed"] and config.device == "cpu"
    native = Path(manifest["run"]) / "checkpoints" / "epoch-00000008"
    assert sha256(native / "model.safetensors") == entry["candidate_sha256"]
    learner = TorchSearchActingLearner.resume(
        native,
        config=config,
        input_paths={k: Path(v["path"]) for k, v in manifest["inputs"].items()},
        source_commit=manifest["source_commit"],
    )
    assert learner.epoch == 8 and learner.closed
    return learner.online.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--probes", type=Path, required=True)
    p.add_argument("--backend", choices=("cpu", "mlx"), required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    validate_binding(a, Q)
    started = time.time()
    _prepare_device("cpu")
    if a.backend == "cpu":
        assert torch.__version__ == "2.14.1+cpu"
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
        [BoardEncoder().encode_board(board).values for board in boards],
        dtype=torch.float32,
    ).reshape(-1, 8, 8, 104)
    reports = []
    with torch.inference_mode():
        for entry in spec["seeds"]:
            path = Path(entry["candidate"])
            assert sha256(path) == entry["candidate_sha256"]
            cpu = load_weights(path).eval()
            other = (
                restore_native(entry)
                if a.backend == "cpu"
                else HarbiChessPairwiseNetwork.from_portable(path)
            )
            other.eval()
            ca = cpu(encoded)
            ga = (
                other(encoded.to("cpu")) if a.backend == "cpu" else other(mx.array(encoded.numpy()))
            )
            if a.backend == "mlx":
                mx.eval(ga)
            errors = []
            for x, y in zip(ca, ga, strict=True):
                xn = x.numpy()
                yn = y.cpu().numpy() if a.backend == "cpu" else np.array(y)
                assert np.isfinite(xn).all() and np.isfinite(yn).all()
                np.testing.assert_allclose(xn, yn, atol=2e-05, rtol=2e-05)
                errors.append(float(np.max(np.abs(xn - yn))))
            masked = []
            for index, board in enumerate(boards):
                actions = torch.tensor([legal_action_indices(board)])
                ca = cpu.masked_policy_value(encoded[index : index + 1], actions)
                ga = (
                    other.masked_policy_value(
                        encoded[index : index + 1].to("cpu"), actions.to("cpu")
                    )
                    if a.backend == "cpu"
                    else other.masked_policy_value(
                        mx.array(encoded[index : index + 1].numpy()),
                        mx.array(actions.numpy()),
                    )
                )
                if a.backend == "mlx":
                    mx.eval(ga)
                for x, y in zip(ca, ga, strict=True):
                    xn = x.numpy()
                    yn = y.cpu().numpy() if a.backend == "cpu" else np.array(y)
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
        "schema": "ufuk-cpu-contingency-final-portable18parity-v1",
        "status": "pass-sameweights-full-and-legalmasked18fullhistories",
        "backend": a.backend,
        "device": "CPU native->portable actual"
        if a.backend == "cpu"
        else "MLXCPU actual; AppleMetal untested",
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
        qualification_ledger_slot=8,
    )
    with a.output.open("x") as f:
        json.dump(receipt, f, indent=2)
        f.write("\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    bind_cli(globals())
    main()

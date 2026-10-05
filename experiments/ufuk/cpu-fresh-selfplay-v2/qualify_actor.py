"""Root-invoked actual e8 whole9/pause4/freshresume9; never synthetic qualification."""

import argparse
import json
import time
from pathlib import Path

from controller_support import (
    bindings,
    command,
    guarded_child,
    journal_module,
    runtime,
    sha,
    validate_clock,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--first-epoch", type=float, required=True)
    parser.add_argument("--ram-stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    manifest, config = bindings(args.manifest, args.manifest_sha256, 9)
    deadline = config["original_deadline_epoch"]
    validate_clock(args.first_epoch, deadline, 600, time.time())
    owned, guard = runtime(manifest, args.workspace, args.ram_stage)
    guard(deadline)
    journal = journal_module(manifest)
    journal.validate_config(config)
    if not args.execute:
        print(
            json.dumps({"status": "preflight-only-no-actor-executed", "deadline_epoch": deadline})
        )
        return
    args.output.mkdir(parents=True, exist_ok=False)
    args.ram_stage.mkdir(parents=True, exist_ok=False)
    contract = {
        "schema": "fresh-qsearch-realCPU-qualification-owner-v1",
        "manifest_path": str(args.manifest),
        "manifest_sha256": args.manifest_sha256,
        "source_commit": config["source_commit"],
        "original_first_epoch": args.first_epoch,
        "original_deadline_epoch": deadline,
        "helper_sha256": sha(__file__),
        "support_sha256": sha(Path(__file__).with_name("controller_support.py")),
        "GPU_used": False,
        "registered_cpu_core": manifest["cpu_core"],
        "synthetic_mode_allowed": False,
    }
    owned.publish(args.output / "owner-contract.json", contract)
    try:
        whole, pause, resumed = [
            args.ram_stage / name
            for name in ("whole9.json.gz", "pause4.json.gz", "resumed9.json.gz")
        ]
        for label, target, output, parent in [
            ("whole9", 9, whole, None),
            ("pause4", 4, pause, None),
            ("resumed9", 9, resumed, pause),
        ]:
            bindings(args.manifest, args.manifest_sha256, 9)
            guarded_child(
                command(manifest, config, target, output, parent, sha(parent) if parent else None),
                manifest,
                deadline,
                guard,
                args.output,
                label,
                owned.publish,
            )
        states = [journal.read(path) for path in (whole, pause, resumed)]
        if whole.read_bytes() != resumed.read_bytes() or journal.canonical(
            states[0]
        ) != journal.canonical(states[2]):
            raise ValueError("whole versus fresh-process resume journal/state/RNG/anchors differs")
        for state in states:
            guard(deadline)
            journal.replay(state, config)
        # Independently use full masked network forward for every stored pre-action anchor.
        import torch

        from harbichess.backends.torch_network import load_weights
        from harbichess.training.torch_array_encoder import TorchArrayBoardEncoder

        torch.set_num_threads(1)
        network = load_weights(Path(manifest["anchor_model"]["path"])).eval().requires_grad_(False)
        encoder = TorchArrayBoardEncoder()
        cache, checked = {}, 0
        with torch.inference_mode():
            for state in states:
                games = state["games"] + ([state["active"]] if state["active"] else [])
                for game in games:
                    board = journal.board_for(config["roots"][game["root_index"]])
                    for row in game["moves"]:
                        guard(deadline)
                        key = (board.root().fen(), tuple(m.uci() for m in board.move_stack))
                        if key not in cache:
                            x = torch.from_numpy(encoder.encode_board(board).values.copy()).reshape(
                                1, 8, 8, 104
                            )
                            logits = network.masked_policy_value(
                                x, torch.zeros((1, 1), dtype=torch.long)
                            )[1]
                            cache[key] = torch.softmax(logits, 1)[0]
                        recorded = torch.tensor(row["e8_anchor_wdl"], dtype=torch.float32)
                        torch.testing.assert_close(recorded, cache[key], atol=3e-7, rtol=3e-7)
                        board.push_uci(row["action"])
                        checked += 1
        guard(deadline)
        bindings(args.manifest, args.manifest_sha256, 9)
        result = {
            "status": "PASS-actualCPU-realE8-freshactor-whole9-pause4-resume9-allstate-RNG-anchors",
            "schema": "fresh-qsearch-realCPU-actor-qualification-v1",
            "contract": contract,
            "journal_sha256": {p.name: sha(p) for p in (whole, pause, resumed)},
            "journal_paths": {p.name: str(p) for p in (whole, pause, resumed)},
            "all3_journals_independently_rule_replayed": True,
            "full_WDL_anchor_rows_checked": checked,
            "distinct_fullhistory_anchor_NN_positions": len(cache),
            "new_generation_actions": 18,
            "old_online_training_resume_claimed": False,
            "Torch_version": torch.__version__,
            "GPU_used": False,
            "finished_epoch": time.time(),
            "strength_success_claimed": False,
        }
    except BaseException as exc:
        result = {
            "status": "failed-preserved",
            "contract": contract,
            "error": repr(exc),
            "failure_snapshot": getattr(exc, "snapshot", None),
            "finished_epoch": time.time(),
        }
        owned.publish(args.output / "result.json", result)
        raise
    owned.publish(args.output / "result.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()

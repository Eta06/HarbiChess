"""One frozen real-e88192-action epoch, cumulative1024 milestones; no training."""

import argparse
import json
import time
from pathlib import Path

from controller_support import (
    bindings,
    command,
    guarded_child,
    journal_module,
    read_bound,
    runtime,
    sha,
    validate_clock,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--qualification-sha256", required=True)
    parser.add_argument("--first-epoch", type=float, required=True)
    parser.add_argument("--ram-stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    manifest, config = bindings(args.manifest, args.manifest_sha256, 8192)
    deadline = config["original_deadline_epoch"]
    validate_clock(args.first_epoch, deadline, 14400, time.time())
    q = read_bound(args.qualification, args.qualification_sha256)
    if q["status"] != "PASS-actualCPU-realE8-freshactor-whole9-pause4-resume9-allstate-RNG-anchors":
        raise ValueError("actual real-e8 actor qualification required")
    if q["contract"]["source_commit"] != config["source_commit"]:
        raise ValueError("qualification source differs")
    if q["contract"]["support_sha256"] != sha(Path(__file__).with_name("controller_support.py")):
        raise ValueError("qualified controller guard/support differs")
    if (
        not q["all3_journals_independently_rule_replayed"]
        or q["full_WDL_anchor_rows_checked"] != 22
    ):
        raise ValueError("complete real actor/anchor qualification missing")
    if q["GPU_used"] or q["old_online_training_resume_claimed"]:
        raise ValueError("qualification hardware/native scope differs")
    for name, digest in q["journal_sha256"].items():
        if sha(q["journal_paths"][name]) != digest:
            raise ValueError("actual qualification journal artifact changed")
    if q["journal_sha256"]["whole9.json.gz"] != q["journal_sha256"]["resumed9.json.gz"]:
        raise ValueError("whole/fresh-resume proof differs")
    # Qualification has max9/deadline600, with the same actor implementation.
    qualified = read_bound(q["contract"]["manifest_path"], q["contract"]["manifest_sha256"])
    if (
        qualified["helpers"] != manifest["helpers"]
        or qualified["model"] != manifest["model"]
        or qualified["anchor_model"] != manifest["anchor_model"]
    ):
        raise ValueError("qualified actor/model/helper closure differs")
    qualified_config = read_bound(qualified["config"]["path"], qualified["config"]["sha256"])
    for key in (
        "nodes",
        "qdepth",
        "max_depth",
        "exploration",
        "total_ply_cap",
        "actors",
        "source_commit",
        "torch_version",
        "evaluator_identity",
        "model_sha256",
        "anchor_model_sha256",
        "anchor_helper_sha256",
        "anchor_target",
        "search_helper_sha256",
        "value_helper_sha256",
        "producer_sha256",
        "runner_sha256",
    ):
        if qualified_config[key] != config[key]:
            raise ValueError("qualified fixed search/runtime/anchor config differs")
    owned, guard = runtime(manifest, args.workspace, args.ram_stage)
    guard(deadline)
    journal = journal_module(manifest)
    journal.validate_config(config)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "preflight-only-no-actor-executed",
                    "fixed_max_actions": 8192,
                    "deadline_epoch": deadline,
                }
            )
        )
        return
    args.output.mkdir(parents=True, exist_ok=False)
    args.ram_stage.mkdir(parents=True, exist_ok=False)
    contract = {
        "schema": "fresh-qsearch-fixedE8-8192-epoch-owner-v1",
        "manifest_path": str(args.manifest),
        "manifest_sha256": args.manifest_sha256,
        "qualification_sha256": args.qualification_sha256,
        "source_commit": config["source_commit"],
        "seed": config["seed"],
        "registered_cpu_core": manifest["cpu_core"],
        "max_actions": 8192,
        "original_first_epoch": args.first_epoch,
        "original_deadline_epoch": deadline,
        "helper_sha256": sha(__file__),
        "support_sha256": sha(Path(__file__).with_name("controller_support.py")),
    }
    owned.publish(args.output / "owner-contract.json", contract)
    parent, parent_sha = None, None
    rows = []
    try:
        for target in range(1024, 8193, 1024):
            guard(deadline)
            bindings(args.manifest, args.manifest_sha256, 8192)
            checkpoint = args.ram_stage / f"actions-{target:08d}.json.gz"
            invocation = command(manifest, config, target, checkpoint, parent, parent_sha)
            process = guarded_child(
                invocation,
                manifest,
                deadline,
                guard,
                args.output,
                f"actions-{target:08d}",
                owned.publish,
            )
            state = journal.read(checkpoint)
            journal.replay(state, config)
            if state["actions"] != target:
                raise ValueError("actual cumulative action cursor differs")
            row = {
                "target_actions": target,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": sha(checkpoint),
                "parent_checkpoint_sha256": parent_sha,
                "process_result": process,
                "config_sha256": manifest["config"]["sha256"],
                "completed_games": len(state["games"]),
                "active_tail_excluded": state["active"] is not None,
            }
            owned.publish(args.output / f"actions-{target:08d}.milestone.json", row)
            rows.append(row)
            parent, parent_sha = checkpoint, row["checkpoint_sha256"]
        guard(deadline)
        bindings(args.manifest, args.manifest_sha256, 8192)
        result = {
            "status": "completed-fixed8192-freshown-actions-not-strength",
            "contract": contract,
            "milestones": rows,
            "finished_epoch": time.time(),
            "training_updates": 0,
            "candidate_selection": False,
            "GPU_used": False,
        }
    except BaseException as exc:
        result = {
            "status": "failed-preserved",
            "contract": contract,
            "milestones": rows,
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

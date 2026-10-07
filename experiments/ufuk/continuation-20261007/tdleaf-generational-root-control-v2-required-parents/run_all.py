"""ROOT-only prospective fixed6banks+3generations, bothseeds, no strength jobs."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from control import (
    END,
    SEEDS,
    Owner,
    descendants,
    pinned,
    proc_identity,
    publish,
    read,
    ref,
    setup,
    stop_owned,
)
from learn import next_parent
from seal_plan import seal


def pair(owner, root, commands):
    """Parallel CPU1/3; identity-owned children, technical END, no name kills."""
    root = Path(root)
    root.mkdir(exist_ok=False)
    children = []
    tracked = {}
    record = dict(first=time.time(), deadline=END, commands=commands, status="registered")
    publish(root / "commands.json", record)
    try:
        for seed, cmd in commands.items():
            out = (root / f"{seed}.stdout.log").open("xb")
            err = (root / f"{seed}.stderr.log").open("xb")
            proc = subprocess.Popen(
                cmd,
                stdout=out,
                stderr=err,
                start_new_session=True,
                env={
                    **os.environ,
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                },
            )
            identity = proc_identity(proc.pid)
            if identity is None:
                raise RuntimeError("child ended before identity receipt; preserve failure")
            children.append((proc, identity, out, err))
            tracked[identity["pid"]] = identity
            publish(
                root / f"{seed}-process.json",
                dict(pid=proc.pid, identity=identity, first=record["first"], deadline=END),
            )
        while any(p.poll() is None for p, _, _, _ in children):
            for p, i, _, _ in children:
                descendants(i, tracked)
                if p.poll() not in (None, 0):
                    raise RuntimeError("paired owned worker failed")
            owner.guard(END)
            time.sleep(0.25)
        if any(p.returncode != 0 for p, _, _, _ in children):
            raise RuntimeError("paired worker exit")
        owner.guard(END)
        record["status"] = "PASS-owned-pair-not-strength"
    except BaseException as e:
        record.update(status="FAILED-preserved", error=repr(e))
        stop_owned(tracked)
        for p, _, _, _ in children:
            p.wait()
        raise
    finally:
        for _, _, out, err in children:
            out.close()
            err.close()
        record.update(finished=time.time(), tracked_owned=list(tracked.values()))
        publish(root / "result.json", record)


def execute(config_ref, output):
    config = read(config_ref)
    if config["status"] != "registered":
        raise ValueError("ROOT registered before any execution")
    owner = Owner(config, output)
    if not owner.root.resolve().is_relative_to("/dev/shm"):
        raise ValueError("full registration/source/phase output in RAM")
    publish(
        owner.root / "scope.json",
        dict(
            schema="fixed3generation-own-learning-scope-v1",
            first=time.time(),
            operator_end_epoch=END,
            config=config_ref,
            strength_scope=config["strength_scope"],
            endpoint_generation=3,
            generation_parents="g0->g1->g2;weights-only-fresh-Adam-RNG",
            no_actual_strength_jobs=True,
        ),
    )
    banks = {}
    for seed in SEEDS:
        banks[str(seed)] = [
            sys.executable,
            str(Path(__file__).with_name("control.py")),
            "--config",
            str(pinned(config_ref)),
            "--config-sha256",
            config_ref["sha256"],
            "--mode",
            "banks",
            "--seed",
            str(seed),
            "--output",
            str(owner.root / f"banks-{seed}"),
            "--execute",
        ]
    pair(owner, owner.root / "bank-pair", banks)
    config_for_plan = dict(config, config_path=str(pinned(config_ref)))
    bank_sets = {str(s): ref(owner.root / f"banks-{s}/banks-{s}.json") for s in SEEDS}
    plan_path = owner.root / "generation-plan.json"
    seal(config_for_plan, bank_sets, plan_path)
    plan_ref = ref(plan_path)
    parents = config["zero_parent_seals"].copy()
    final = {}
    for generation in (1, 2, 3):
        commands = {}
        for seed in SEEDS:
            commands[str(seed)] = [
                sys.executable,
                str(Path(__file__).with_name("generation_entry.py")),
                "--config",
                str(pinned(config_ref)),
                "--config-sha256",
                config_ref["sha256"],
                "--plan",
                str(plan_path),
                "--plan-sha256",
                plan_ref["sha256"],
                "--parent-seal",
                str(pinned(parents[str(seed)])),
                "--parent-seal-sha256",
                parents[str(seed)]["sha256"],
                "--seed",
                str(seed),
                "--generation",
                str(generation),
                "--output",
                str(owner.root / f"g{generation}-{seed}"),
            ]
        pair(owner, owner.root / f"generation{generation}-pair", commands)
        for seed in SEEDS:
            record_ref = ref(owner.root / f"g{generation}-{seed}/generation-complete.json")
            record = read(record_ref)
            final[str(seed)] = record_ref
            if generation < 3:
                path = owner.root / f"parent-g{generation + 1}-{seed}.json"
                publish(path, next_parent(record, owner.h))
                parents[str(seed)] = ref(path)
    owner.guard(END)
    result = dict(
        status="PASS-fixed-generation3-full-own-data-native-only-not-strength",
        config=config_ref,
        plan=plan_ref,
        final_generation3=final,
        original_literalzero_parents=config["zero_parent_seals"],
        required_current_generation2={
            str(s): ref(owner.root / f"g2-{s}/generation-complete.json") for s in SEEDS
        },
        strength_scope=config["strength_scope"],
        finished=time.time(),
    )
    publish(owner.root / "result.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--config-sha256", required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    a = p.parse_args()
    r = dict(path=str(a.config), sha256=a.config_sha256)
    if a.validate_only:
        setup(read(r), allow_draft=True)
        print(
            json.dumps(
                dict(status="PASS-readonly-original-zero-and-immutable-source-preflight-no-compute")
            )
        )
    elif a.execute and a.output:
        execute(r, a.output)
    else:
        p.error("--validate-only or ROOT --execute --output required")

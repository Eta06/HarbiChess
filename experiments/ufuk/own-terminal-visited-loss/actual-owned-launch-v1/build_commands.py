"""Read-only root command planner: emits argv, never invokes processes or samples clocks."""

import argparse
import hashlib
import json
import time
from pathlib import Path

SHA_FLAGS = {
    "template",
    "profile-receipt",
    "auditor-receipt",
    "mc-completion-barrier",
    "books-provenance",
    "curriculum-provenance",
    "curriculum-qualification",
    "unit59-receipt",
    "unit-case-inventory",
    "ancestor-auditor-receipt",
    "failed4-dependency-supplement",
    "original45-cohort",
    "terminal45-barrier-config",
    "standalone8-prior-failures",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def complete(value):
    text = json.dumps(value)
    assert "ROOT_FILL" not in text and "ROOT_SAMPLE" not in text
    return value


def factory(plan):
    complete(plan)
    paths = plan["factory_inputs"]
    assert type(plan["earliest_training_epoch"]) in (int, float)
    assert plan["earliest_training_epoch"] <= time.time() <= 1791169800
    for name, digest in plan["expected_helper_sha256"].items():
        assert sha(Path(paths["helpers"]) / name) == digest, name
    argv = [plan["python"], str(Path(paths["helpers"]) / "own8_formal_config_factory.py")]
    for name, path in paths.items():
        if name == "output":
            assert not Path(path).exists()
        else:
            assert Path(path).exists(), path
        argv.extend(["--" + name, str(path)])
        if name in SHA_FLAGS:
            argv.extend(["--" + name + "-sha256", sha(path)])
        if name in plan["expected_sha256"]:
            assert sha(path) == plan["expected_sha256"][name], name
    profile = json.loads(Path(paths["profile-receipt"]).read_text())
    assert (
        profile["status"]
        == "pass-one-search-acting-v4-development-epoch-and-fullchronological-audit"
    )
    assert profile["source_commit"] == plan["source_commit"]
    argv.extend(
        [
            "--fixed-epochs",
            "8",
            "--whole-training-seconds",
            "6000",
            "--whole-audit-seconds",
            "9000",
            "--earliest-training-epoch",
            str(plan["earliest_training_epoch"]),
            "--posttraining-reserve-seconds",
            str(plan["posttraining_reserve_seconds"]),
            "--freeze",
        ]
    )
    return argv


def owners(plan):
    complete(plan)
    frozen = Path(plan["factory_inputs"]["output"])
    helpers = Path(plan["factory_inputs"]["helpers"])
    paths = json.loads(Path(plan["owner_paths_config"]).read_text())
    complete(paths)
    argv = [plan["python"], str(helpers / "own8_owner_config_factory.py")]
    for name, path in {
        "registration": frozen / "registration.json",
        "three-input-manifest": frozen / "three-input-manifest.json",
        "qualification-config": frozen / "strength-bindings.json",
        "paths-config": Path(plan["owner_paths_config"]),
    }.items():
        argv.extend(["--" + name, str(path), "--" + name + "-sha256", sha(path)])
    assert not Path(plan["owner_output"]).exists()
    argv.extend(["--helpers", str(helpers), "--output", plan["owner_output"]])
    return argv


def launches(plan):
    complete(plan)
    helpers = Path(plan["factory_inputs"]["helpers"])
    frozen = Path(plan["factory_inputs"]["output"])
    owner = Path(plan["owner_output"])
    config = json.loads((owner / "posttraining-config.json").read_text())
    complete(config)
    q = frozen / "strength-bindings.json"
    reg = frozen / "registration.json"
    commands = []
    for row in config["seeds"]:
        seed, first = row["seed"], row["original_training_started_epoch"]
        assert first <= time.time() < first + 6000
        assert seed in (20261825, 20261826)
        trainroot = Path(row["run"]).parent
        assert not trainroot.exists()
        common = ["--repo", plan["producer_repo"], "--python", plan["python"]]
        commands.append(
            [
                plan["python"],
                str(helpers / "own8_training_controller.py"),
                *common,
                "--registration",
                str(reg),
                "--registration-sha256",
                sha(reg),
                "--input-manifest",
                str(frozen / "three-input-manifest.json"),
                "--root",
                str(trainroot),
                "--seed",
                str(seed),
                "--started-epoch",
                str(first),
                "--deadline-epoch",
                str(first + 6000),
            ]
        )
        manifest = owner / f"audit-manifest-{seed}.json"
        commands.append(
            [
                plan["python"],
                str(helpers / "own8_audit_controller.py"),
                *common,
                "--manifest",
                str(manifest),
                "--manifest-sha256",
                sha(manifest),
                "--audit-script",
                str(helpers / "own8_full_audit.py"),
                "--audit-script-sha256",
                sha(helpers / "own8_full_audit.py"),
                "--root",
                str(Path(row["audit_controller_result"]).parent),
            ]
        )
        baseline_first = row["baseline_deadline_epoch"] - 3600
        commands.append(
            [
                plan["python"],
                str(helpers / "own8_baseline.py"),
                *common,
                "--qualification-config",
                str(q),
                "--qualification-config-sha256",
                sha(q),
                "--registration",
                str(reg),
                "--registration-sha256",
                sha(reg),
                "--weights",
                config["initial"],
                "--book",
                row["book"],
                "--stockfish",
                config["stockfish"],
                "--root",
                row["baseline_root"],
                "--seed",
                str(seed),
                "--started-epoch",
                str(baseline_first),
                "--deadline-epoch",
                str(row["baseline_deadline_epoch"]),
            ]
        )
    post = owner / "posttraining-config.json"
    commands.append(
        [
            plan["python"],
            str(helpers / "own8_posttraining.py"),
            "--config",
            str(post),
            "--config-sha256",
            sha(post),
            "--execute",
        ]
    )
    return commands


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--phase", choices=["factory", "owners", "launches"], required=True)
    args = parser.parse_args()
    assert sha(args.plan) == args.plan_sha256
    plan = json.loads(args.plan.read_text())
    result = {
        "phase": args.phase,
        "argv": globals()[args.phase](plan),
        "processes_launched": 0,
        "clocks_sampled_or_reset": 0,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

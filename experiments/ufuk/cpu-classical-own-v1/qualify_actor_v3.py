"""REAL classical Qsearch whole9/pause4/fresh-process-resume9, shared600s proof."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from journal_v3 import load_module, read, replay, sha

HELPER_CLOSURE = frozenset(
    {
        "run_epoch_v3.py",
        "qualify_actor_v3.py",
        "journal_v3.py",
        "produce_v3.py",
        "value.py",
        "train_v3.py",
        "learner.py",
        "runtime.py",
        "transfer.py",
    }
)


def run_owned(command, deadline, root, name, resource_guard=None):
    if time.time() >= deadline:
        raise TimeoutError("original owned deadline already exhausted")
    if resource_guard is not None:
        resource_guard()
    with (root / (name + ".stdout")).open("x") as out, (root / (name + ".stderr")).open("x") as err:
        process = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
        try:
            while process.poll() is None:
                if resource_guard is not None:
                    resource_guard()
                if time.time() >= deadline:
                    raise TimeoutError("original qualification deadline")
                time.sleep(0.1)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=3)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("config", "search-helper", "value-helper", "model", "source-repo", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--config-sha256", required=True)
    p.add_argument("--first", type=float, required=True)
    p.add_argument("--deadline", type=float, required=True)
    p.add_argument("--cpu-core", type=int, required=True)
    a = p.parse_args()
    if (
        a.first > time.time()
        or a.deadline != a.first + 600
        or a.deadline > 1791273600.0
        or time.time() >= a.deadline
    ):
        raise ValueError("actual original600/hard08clock required")
    helpers = {n: sha(Path(__file__).with_name(n)) for n in sorted(HELPER_CLOSURE)}
    a.output.mkdir(exist_ok=False)
    base = [sys.executable, str(Path(__file__).with_name("produce_v3.py"))]
    for k in ("config", "search-helper", "value-helper", "model", "source-repo"):
        base += ["--" + k, str(getattr(a, k.replace("-", "_")))]
    base += [
        "--config-sha256",
        a.config_sha256,
        "--original-deadline",
        str(a.deadline),
        "--cpu-core",
        str(a.cpu_core),
    ]
    from runtime import guard

    def parent_guard():
        return guard(a.deadline, a.source_repo, a.output)

    whole = a.output / "whole9.json.gz"
    pause = a.output / "pause4.json.gz"
    resumed = a.output / "resumed9.json.gz"
    run_owned(
        [*base, "--target-actions", "9", "--output", str(whole)],
        a.deadline,
        a.output,
        "whole9",
        parent_guard,
    )
    run_owned(
        [*base, "--target-actions", "4", "--output", str(pause)],
        a.deadline,
        a.output,
        "pause4",
        parent_guard,
    )
    run_owned(
        [
            *base,
            "--target-actions",
            "9",
            "--output",
            str(resumed),
            "--resume",
            str(pause),
            "--resume-sha256",
            sha(pause),
        ],
        a.deadline,
        a.output,
        "resume9",
        parent_guard,
    )
    if whole.read_bytes() != resumed.read_bytes():
        raise ValueError("full native journal/RNG gzip mismatch")
    c = json.loads(a.config.read_text())
    v = load_module(a.value_helper, c["value_helper_sha256"])
    prior = v.ClassicalValue()
    counts = []
    for path in (whole, pause, resumed):
        state = read(path)
        packets = replay(state, c)
        # Replay includes active UNKNOWN rows; exact prior recomputation covers every stored action.
        checked = 0
        from journal_v3 import board_for

        for g in state["games"] + ([state["active"]] if state["active"] else []):
            b = board_for(c["roots"][g["root_index"]])
            for row in g["moves"]:
                if prior.nonterminal(b) != row["human_prior_scalar"]:
                    raise ValueError("actual frozen human-prior packet mismatch")
                b.push_uci(row["action"])
                checked += 1
        counts.append(
            dict(
                path=str(path),
                sha256=sha(path),
                actions=state["actions"],
                prior_packets=checked,
                completed_known_games=len(packets),
            )
        )
    if helpers != {n: sha(Path(__file__).with_name(n)) for n in sorted(HELPER_CLOSURE)}:
        raise ValueError("qualified helper closure changed during actual proof")
    parent_guard()
    if time.time() >= a.deadline:
        raise TimeoutError("original qualification600 includes independent audit")
    receipt = dict(
        helper_sha256=helpers,
        schema="classical-own-realCPU-actor-qualification-v3",
        status="PASS-new-classical-actor-native-whole9-pause4-freshresume9-not-strength",
        finished_epoch=time.time(),
        first=a.first,
        deadline=a.deadline,
        source_commit=c["source_commit"],
        config_sha256=a.config_sha256,
        all_journals=counts,
        whole_resumed_gzip_exact=True,
        old_E8_native_resume_claimed=False,
    )
    from train_v3 import publish

    publish(a.output / "result.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()

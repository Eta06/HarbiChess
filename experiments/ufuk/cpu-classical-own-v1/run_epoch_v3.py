"""One owned frozen16384 epoch, cumulative1024 segments; no training/cycling."""

import argparse
import json
import sys
import time
from pathlib import Path

from journal_v3 import digest, read, replay, sha
from qualify_actor_v3 import HELPER_CLOSURE, run_owned
from train_v3 import publish
from transfer import validate_transfer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--qualification", type=Path, required=True)
    p.add_argument("--qualification-sha256", required=True)
    p.add_argument("--first", type=float, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if sha(a.manifest) != a.manifest_sha256 or sha(a.qualification) != a.qualification_sha256:
        raise ValueError("manifest/proof SHA differs")
    m = json.loads(a.manifest.read_text())
    q = json.loads(a.qualification.read_text())
    if (
        m["schema"] != "classical-own-epoch-controller-manifest-v3"
        or q["status"] != "PASS-new-classical-actor-native-whole9-pause4-freshresume9-not-strength"
    ):
        raise ValueError("actual classical proof/manifest required")
    if set(m["helper_sha256"]) != HELPER_CLOSURE or m["helper_sha256"] != q["helper_sha256"]:
        raise ValueError("entire qualified owner helper closure differs")
    for name, expected in m["helper_sha256"].items():
        if sha(Path(__file__).with_name(name)) != expected:
            raise ValueError("qualified helper code bytes differ")
    for role in ("config", "model", "search_helper", "value_helper"):
        if sha(m[role]["path"]) != m[role]["sha256"]:
            raise ValueError("actual input SHA differs")
    c = json.loads(Path(m["config"]["path"]).read_text())
    if len(q["all_journals"]) != 3 or [r["actions"] for r in q["all_journals"]] != [9, 4, 9]:
        raise ValueError("all three genuine qualification journals required")
    if q["deadline"] != q["first"] + 600 or q["finished_epoch"] > q["deadline"]:
        raise ValueError("qualification original deadline violated")
    if (
        Path(q["all_journals"][0]["path"]).read_bytes()
        != Path(q["all_journals"][2]["path"]).read_bytes()
    ):
        raise ValueError("actual whole/resumed gzip bytes differ")
    proof_input = m["qualification_config"]
    proof_path = Path(proof_input["path"])
    actual_proof_sha = sha(proof_path)
    if actual_proof_sha != proof_input["sha256"]:
        raise ValueError("sealed actual qualification config file differs")
    proofcfg = json.loads(proof_path.read_text())
    validate_transfer(proofcfg, c, m["qualification_transfer"], q, actual_proof_sha)
    for item in q["all_journals"]:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("proof journal bytes differ")
        state = read(item["path"])
        if state["config"] != proofcfg or state["config_sha256"] != digest(proofcfg):
            raise ValueError("journal embedded exact qualification config differs")
        if state["actions"] != item["actions"]:
            raise ValueError("qualification action count differs")
        replay(state, proofcfg)
    if not q["whole_resumed_gzip_exact"] or q["source_commit"] != c["source_commit"]:
        raise ValueError("qualification source/resume differs")
    if (
        c["max_actions"] != 16384
        or a.first != c["original_first_epoch"]
        or not a.first <= time.time() < c["original_deadline_epoch"] <= 1791273600.0
    ):
        raise ValueError("fixed epoch originalclock differs")
    a.output.mkdir(exist_ok=False)
    publish(
        a.output / "owner-contract.json",
        dict(
            manifest_sha256=a.manifest_sha256,
            qualification_sha256=a.qualification_sha256,
            original_first=a.first,
            original_deadline=c["original_deadline_epoch"],
            max_actions=16384,
            automatic_training=False,
        ),
    )
    base = [sys.executable, str(Path(__file__).with_name("produce_v3.py"))]
    for flag, role in [
        ("config", "config"),
        ("model", "model"),
        ("search-helper", "search_helper"),
        ("value-helper", "value_helper"),
    ]:
        base += ["--" + flag, m[role]["path"]]
    base += [
        "--config-sha256",
        m["config"]["sha256"],
        "--source-repo",
        m["source_repo"],
        "--cpu-core",
        str(m["cpu_core"]),
        "--original-deadline",
        str(c["original_deadline_epoch"]),
    ]
    from runtime import guard

    def parent_guard():
        return guard(c["original_deadline_epoch"], m["source_repo"], a.output)

    previous = None
    for target in range(1024, 16385, 1024):
        path = a.output / f"actions-{target:08d}.json.gz"
        command = [*base, "--target-actions", str(target), "--output", str(path)]
        if previous:
            command += ["--resume", str(previous), "--resume-sha256", sha(previous)]
        run_owned(
            command, c["original_deadline_epoch"], a.output, f"segment-{target}", parent_guard
        )
        state = read(path)
        replay(state, c)
        parent_guard()
        if time.time() >= c["original_deadline_epoch"]:
            raise TimeoutError("original collection clock includes full replay")
        if state["actions"] != target:
            raise ValueError("cumulative cursor differs")
        publish(
            a.output / f"receipt-{target:08d}.json",
            dict(
                actions=target,
                journal_sha256=sha(path),
                parent_journal_sha256=sha(previous) if previous else None,
                original_deadline=c["original_deadline_epoch"],
                finished=time.time(),
            ),
        )
        previous = path
    parent_guard()
    if time.time() >= c["original_deadline_epoch"]:
        raise TimeoutError("original collection clock includes final audit")
    publish(
        a.output / "result.json",
        dict(
            status="PASS-fixed16384-classical-own-data-not-trained",
            final_journal=str(previous),
            final_journal_sha256=sha(previous),
            original_first=a.first,
            original_deadline=c["original_deadline_epoch"],
        ),
    )


if __name__ == "__main__":
    main()

"""ROOT-only one local build/seal. No API POST, workflow or asset publication."""

import argparse
import base64
import json
import os
import shutil
import sys
import time
from pathlib import Path

import raw_capsule_v9 as capsule
import receiver_v9 as receiver
from prepare_v9 import files, sha


def publish(path, value):
    with path.open("xb") as f:
        f.write(json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    a = p.parse_args()
    original = a.manifest.read_bytes()
    m = json.loads(original)
    capsule.checked_rows(m)
    clock = m["new_scope_clock"]
    if not a.output.resolve().is_relative_to(Path("/dev/shm")):
        raise ValueError("RAM build only")
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    sys.path.insert(0, str(a.source_root / "src"))
    path = a.source_root / "src/harbichess/training/cgroup_budget.py"
    if (
        capsule.sha(path.read_bytes())
        != "a53079851ab9fb63b65dd5bef0d0e8fb01d6864ebfbf2389dcdbcab39b30289c"
    ):
        raise ValueError("actual cgroup source")
    from harbichess.training.cgroup_budget import CgroupMemoryBudget

    memory = CgroupMemoryBudget(15 * 2**30)

    def guard(check_integrity=True):
        if (
            not clock["first"]
            <= time.time()
            < clock["deadline"]
            <= min(clock["first"] + 5400, 1791273600)
        ):
            raise TimeoutError("new original5400")
        memory.check()
        if shutil.disk_usage("/workspace").free < 256 * 2**20:
            raise RuntimeError("workspacefloor")
        if not check_integrity:
            return
        if a.manifest.read_bytes() != original:
            raise ValueError("manifest changed")
        for name, digest in m["source_bindings"].items():
            if capsule.sha(Path(__file__).with_name(name).read_bytes()) != digest:
                raise ValueError("frozen helper changed")
        for name, digest in m["producer_bindings"].items():
            if sha(Path(__file__).with_name(name)) != digest:
                raise ValueError("frozen producer changed")
        observed = files({"regular_file_roots": m["scope_roots"], "exact_files": m["exact_files"]})
        if [str(p) for p in observed] != [r["path"] for r in m["rows"]]:
            raise ValueError("scope changed; no unrecorded tail omission")
        q = m["scope_quiescence_receipt"]
        if sha(q["path"]) != q["sha256"]:
            raise ValueError("quiescence receipt changed")

    guard()
    a.output.mkdir(exist_ok=False)
    raw = capsule.encode(m, guard=lambda: guard(check_integrity=False))
    guard()
    proof = capsule.verify(raw, m)
    guard()
    capsha = capsule.sha(raw)
    blob = receiver.git_sha(raw)
    with (a.output / "capsule.bundle.zlib").open("xb") as f:
        f.write(raw)
    publish(
        a.output / "blob-POST-body.json",
        dict(content=base64.b64encode(raw).decode(), encoding="base64"),
    )
    control = dict(
        schema="classical-raw-blob-release-control-v9",
        approval="ROOT-approved",
        manifest_sha256=capsule.sha(original),
        release_id=404068972,
        release_tag="ufuk-cpu-ownplay-state-20261005",
        protected_assets=[
            receiver.PROTECTED,
            "classical-state-v7-sha256-06104b5fb2332a94176f7b4dbeb121acf8912cc8a9cbfb69b2a15d8f954e2ef6.bundle.zlib",
        ],
        asset_name=f"classical-state-v9-sha256-{capsha}.bundle.zlib",
        capsule_sha256=capsha,
        capsule_bytes=len(raw),
        blob_sha1=blob,
        first=clock["first"],
        deadline=clock["deadline"],
        source_commit=m["source_commit"],
        helper_sha256={
            k: m["source_bindings"][k]
            for k in ["raw_capsule_v9.py", "receiver_v9.py", "release_transport_v9.py"]
        },
        workflow_sha256=m["workflow_sha256"],
        clock=dict(
            schema="classical-raw-blob-release-clock-v9",
            status="ROOT-approved-frozen-transport",
            operator_end_epoch=1791273600,
            started_epoch=clock["first"],
            deadline_epoch=clock["deadline"],
        ),
    )
    receiver.validate_control(control, original)
    publish(a.output / "control.json", control)
    approval = dict(
        schema="classical-native-state-blob-ROOT-approval-v9",
        status="ROOT-approved-frozen-transport",
        manifest_sha256=capsule.sha(original),
        control_sha256=capsule.sha((a.output / "control.json").read_bytes()),
        blob_sha1=blob,
        release_id=404068972,
        scope="V9-explicit-original-byte-public-preservation-not-runtime-strength",
    )
    publish(a.output / "ROOT-approval.json", approval)
    publish(a.output / "allowlist.json", {capsule.sha(original): "manifest.json"})
    guard()
    publish(
        a.output / "local-build-proof.json",
        dict(
            status="PASS-local-byte-build-NOT-public",
            files=len(proof),
            originals=proof,
            capsule_sha256=capsha,
            capsule_bytes=len(raw),
            blob_sha1=blob,
            finished=time.time(),
            deadline=clock["deadline"],
            POST_attempts=0,
            native_payloads_decoded=False,
        ),
    )


if __name__ == "__main__":
    main()

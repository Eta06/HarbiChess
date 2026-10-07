"""ROOT registered literalzero0, native/candidate plus two FRESH strict opens."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from model import FEATURE_SCHEMA
from native import MATH, Learner, load_native
from support import clock, guard, pinned, pins_tree, publish, ref

HERE = Path(__file__).resolve().parent


def build(seal):
    clock(seal, 600)
    pins_tree(seal)
    from compiled_evaluator import load_compiled, load_prior

    load_compiled(seal["compiled_refs"])
    load_prior(seal["prior_helper"])
    if (
        seal["schema"] != "antisymmetric-zero-initialization-seal-v3"
        or type(seal["seed"]) is not int
        or seal["seed"] not in (20262905, 20262906)
        or seal["status"] != "registered"
        or seal["teacher_labels_used"] is not False
        or seal["search_helper"]["sha256"]
        != "de53c14728a67b7772f18b396ac8ef5c35a144d4e4e616fec40099cd461a6670"
    ):
        raise ValueError("new typed teacher-free zero family original search")
    sources = {
        n: ref(HERE / n)
        for n in [
            "model.py",
            "native.py",
            "initialize.py",
            "zero_parent.py",
            "support.py",
            "compiled_evaluator.py",
        ]
    }
    return dict(
        seal,
        phase="human-prior-zero-antisymmetric-init-v3",
        updates=0,
        initializer="literalzero-antisymmetric-new-Adam-RNG-v1",
        feature_schema=FEATURE_SCHEMA,
        math=MATH,
        source_refs=sources,
    )


def execute(seal, out):
    torch.set_num_threads(1)
    c = build(seal)
    check = guard(c, 600)
    if not out.resolve().is_relative_to("/dev/shm"):
        raise ValueError("RAM output")
    out.mkdir(parents=True, exist_ok=False)
    publish(out / "contract.json", c)
    learner = Learner(c)
    for name, value in [("native.pt", learner.native()), ("candidate.pt", learner.candidate())]:
        with (out / name).open("xb") as f:
            torch.save(value, f)
    check()
    commands = []
    result = dict(
        schema="antisymmetric-zero-two-fresh-opens-result-v3",
        status="FAILED-preserved",
        first=c["first"],
        deadline=c["deadline"],
        seed=c["seed"],
        contract=ref(out / "contract.json"),
        native=ref(out / "native.pt"),
        candidate=ref(out / "candidate.pt"),
        commands=commands,
    )
    try:
        for i in range(2):
            check()
            log = out / f"fresh-{i}.json"
            cmd = [
                sys.executable,
                str(HERE / "initialize.py"),
                "--audit-only",
                "--contract",
                str(out / "contract.json"),
                "--native",
                str(out / "native.pt"),
                "--native-sha256",
                ref(out / "native.pt")["sha256"],
            ]
            started = time.time()
            with log.open("xb") as f:
                proc = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.PIPE,
                    timeout=max(0.01, c["deadline"] - time.time()),
                )
            with (out / f"fresh-{i}.stderr").open("xb") as stderr:
                stderr.write(proc.stderr)
            commands.append(
                dict(
                    command=cmd,
                    started=started,
                    finished=time.time(),
                    returncode=proc.returncode,
                    log_path=str(log),
                    log_sha256=ref(log)["sha256"],
                )
            )
            if proc.returncode:
                raise RuntimeError("fresh zero native failed; raw stderr preserved")
            check()
        result["status"] = "PASS-typedzero-fullnative-two-fresh-opens-not-playing-strength"
    finally:
        result["finished"] = time.time()
        publish(out / "result.json", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seal", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--contract", type=Path)
    p.add_argument("--native", type=Path)
    p.add_argument("--native-sha256")
    a = p.parse_args()
    torch.set_num_threads(1)
    if a.audit_only:
        c = json.loads(a.contract.read_bytes())
        guard(c, 600)()
        state = load_native(pinned(dict(path=str(a.native), sha256=a.native_sha256)), c).native()
        print(json.dumps(dict(status="PASS-strict-antisymmetric-zero-native", step=state["step"])))
    else:
        execute(json.loads(a.seal.read_bytes()), a.output)

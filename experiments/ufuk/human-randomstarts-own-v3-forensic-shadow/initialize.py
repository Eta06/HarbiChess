"""ROOT600 literal-zero initialization and two fresh strict loads; no forward/SGD."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from parent_bridge import END, canonical, pin_tree, pinned, sha
from zero_parent import PHASE

HERE = Path(__file__).resolve().parent


def build_contract(seal):
    from model import FEATURE_SCHEMA
    from native import MATH

    required_keys = {
        "schema",
        "status",
        "seed",
        "first",
        "deadline",
        "operator_end_epoch",
        "core_repo",
        "core_commit",
        "prior_helper",
        "search_helper",
        "inference_source_sha256",
        "teacher_labels_used",
    }
    if set(seal) not in (required_keys, required_keys | {"search_admission"}):
        raise ValueError("exact zero-init metadata scope, no arbitrary data/teacher inputs")
    if (
        seal["teacher_labels_used"] is not False
        or seal["core_commit"] != "6fcc8b476d25495d1c9c413e55b2c7ba4794013e"
        or seal["schema"] != "human-prior-zero-initialization-seal-v1"
        or seal["status"] != "registered"
        or type(seal["seed"]) is not int
        or seal["search_helper"]["sha256"]
        not in (
            "de53c14728a67b7772f18b396ac8ef5c35a144d4e4e616fec40099cd461a6670",
            "90c85ba8e3401cba57c82e6c2a34764d25aa999a03b244a4f2838124a8b46357",
        )
        or not seal["first"]
        < seal["deadline"]
        <= min(seal["first"] + 600, seal["operator_end_epoch"], END)
    ):
        raise ValueError("ROOT registered teacher-free zero-phase source/search/clock")
    pin_tree(seal)
    if (
        seal["search_helper"]["sha256"]
        != "de53c14728a67b7772f18b396ac8ef5c35a144d4e4e616fec40099cd461a6670"
    ):
        admission = json.loads(pinned(seal["search_admission"]).read_bytes())
        if (
            admission["status"] != "PASS-24-paired-TRAIN-searches-not-strength"
            or admission["source_pins"]["advanced_search"] != seal["search_helper"]
            or admission["advanced_incheck_extensions"] != 1
            or admission["latency_within1p10"] is not True
            or not admission["first"] < admission["finished"] <= admission["deadline"]
        ):
            raise ValueError("actual default advanced-human latency admission required")
    sources = {
        str(HERE / n): sha(HERE / n)
        for n in ("model.py", "native.py", "initialize.py", "zero_parent.py")
    }
    inference = seal["inference_source_sha256"]
    required = {
        "model.py": "c04914bc5a560ad51de81c6d0aaeb3fae4d04b45c07cd8e39bbd31f655d36565",
        "native.py": "f60a1f9814f48691af1e2d1f1db1b548cff3e7df0a7cae9718051de1865b8b69",
        "evaluator.py": "5317e736f872a2e237e45942c8df9e797c5acee8f626bb12009464869fab8d95",
        "forward.c": "454f8c8f2b5d177f0ff31fd520b9554636a278b10ddb2f1eedcd92a0f9ad44e0",
        "_kingbucket16.cpython-312-x86_64-linux-gnu.so": (
            "495505cdebd6dbc029ee314846dbac79d5cca6d445de26c804a43821808d644b"
        ),
        "build-receipt.json": "d998140e9505cb667f91b8295f8aae0cd68091b03ddc387633e79fc3fb326e80",
    }
    for name, digest in required.items():
        matches = [h for path, h in inference.items() if Path(path).name == name]
        if matches != [digest]:
            raise ValueError("exact original NNUE C/Python/model/build closure required: " + name)
    if (
        seal["prior_helper"]["sha256"]
        != "a99cddfc397bd9221b99a70691427fdd663488b033fb239d9f48240786b2f277"
    ):
        raise ValueError("exact original unchanged human18 helper")
    if seal["prior_helper"]["sha256"] not in inference.values():
        raise ValueError("prior included in inference closure")
    for path, digest in inference.items():
        pinned(dict(path=path, sha256=digest))
    return dict(
        phase=PHASE,
        updates=0,
        generation=0,
        seed=seal["seed"],
        math=MATH,
        feature_schema=FEATURE_SCHEMA,
        core_source_repo=seal["core_repo"],
        core_source_commit=seal["core_commit"],
        execution_scope_schema="human-prior-zero-initialization-contract-v1",
        original_first_epoch=seal["first"],
        original_deadline_epoch=seal["deadline"],
        operator_end_epoch=seal["operator_end_epoch"],
        source_sha256=sources,
        inference_source_sha256=inference,
        prior_helper_path=seal["prior_helper"]["path"],
        prior_helper_sha256=seal["prior_helper"]["sha256"],
        search_helper=seal["search_helper"],
        lineage_origin=dict(
            phase=PHASE,
            seed=seal["seed"],
            teacher_labels_used=False,
            prior_sha256=seal["prior_helper"]["sha256"],
            initializer_sha256=sha(__file__),
        ),
        initializer_seal=seal,
    )


def execute(seal, output):
    import torch
    from native import Learner, bits_equal, load_native
    from zero_parent import zero_head

    torch.set_num_threads(1)
    if torch.cuda.is_available():
        raise ValueError("CPU-only initialization")
    contract = build_contract(seal)
    core = Path(contract["core_source_repo"])
    if subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=core, text=True
    ).strip() != contract["core_source_commit"] or subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=core, text=True
    ):
        raise ValueError("clean core source")
    sys.path.insert(0, str(core / "src"))
    from harbichess.training.cgroup_budget import CgroupMemoryBudget

    memory = CgroupMemoryBudget(15 * 2**30)
    monotonic_end = time.monotonic() + seal["deadline"] - time.time()

    def guard():
        memory.check()
        if not seal["first"] <= time.time() < seal["deadline"] or time.monotonic() >= monotonic_end:
            raise TimeoutError("original initialization clock")
        if shutil.disk_usage("/workspace").free < 256 * 2**20:
            raise RuntimeError("diskfloor256MiB")

    guard()
    if not output.resolve().is_relative_to("/dev/shm"):
        raise ValueError("RAM-only immutable output")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "contract.json").open("xb") as f:
        f.write(canonical(contract) + b"\n")
    learner = Learner(contract)
    zero_head(learner.model.state_dict())
    if learner.optimizer.state or learner.step != 0:
        raise ValueError("empty newAdam/step0 required")
    for name, value in [("native.pt", learner.native()), ("candidate.pt", learner.candidate())]:
        with (output / name).open("xb") as f:
            torch.save(value, f)
    checked = load_native(output / "native.pt", contract)
    if not bits_equal(checked.native(), learner.native()):
        raise ValueError("exact native0 state/RNG readback")

    def ref(p):
        return dict(path=str(p.resolve()), sha256=sha(p))

    result = dict(
        schema="human-prior-zero-initialization-result-v1",
        status="FAILED-preserved",
        first=seal["first"],
        deadline=seal["deadline"],
        seed=seal["seed"],
        contract=ref(output / "contract.json"),
        contract_sha256=sha(output / "contract.json"),
        native=ref(output / "native.pt"),
        candidate=ref(output / "candidate.pt"),
        source_sha256=contract["source_sha256"],
        teacher_labels_used=False,
        optimizer_updates=0,
        commands=[],
        native_payloads=[],
    )
    try:
        for ordinal in range(2):
            guard()
            log = output / f"fresh-native0-{ordinal}.json"
            cmd = [
                sys.executable,
                str(HERE / "initialize.py"),
                "--audit-only",
                "--contract",
                str(output / "contract.json"),
                "--native",
                str(output / "native.pt"),
                "--native-sha256",
                sha(output / "native.pt"),
            ]
            started = time.time()
            with log.open("xb") as stream:
                child = subprocess.run(
                    cmd,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=max(0.001, seal["deadline"] - time.time()),
                    check=False,
                )
            owner = dict(
                command=cmd,
                first=started,
                finished=time.time(),
                returncode=child.returncode,
                log_sha256=sha(log),
            )
            result["commands"].append(owner)
            result["native_payloads"].append(
                dict(
                    **ref(output / "native.pt"),
                    step=0,
                    actual_fresh_process=True,
                    log_path=str(log),
                )
            )
            if child.returncode or json.loads(log.read_bytes()) != dict(
                status="PASS-strict-native-readonly", step=0
            ):
                raise ValueError("fresh strict initializer load failed")
        guard()
        result["status"] = "PASS-literal-zero-parent-and-two-fresh-native-loads-not-strength"
    except BaseException as exc:
        result["error"] = repr(exc)
        raise
    finally:
        result["finished"] = time.time()
        with (output / "result.json").open("xb") as f:
            f.write(canonical(result) + b"\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seal", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--contract", type=Path)
    p.add_argument("--native", type=Path)
    p.add_argument("--native-sha256")
    a = p.parse_args()
    if a.audit_only:
        import torch
        from native import load_native
        from zero_parent import zero_head

        torch.set_num_threads(1)
        c = json.loads(a.contract.read_bytes())
        if (
            not c["original_first_epoch"] <= time.time() < c["original_deadline_epoch"]
            or sha(a.native) != a.native_sha256
        ):
            raise ValueError("original zero-init clock/native SHA")
        for path, h in c["source_sha256"].items():
            pinned(dict(path=path, sha256=h))
        learner = load_native(a.native, c)
        zero_head(learner.model.state_dict())
        if learner.step != 0 or learner.optimizer.state:
            raise ValueError("literal zero native0")
        print(json.dumps(dict(status="PASS-strict-native-readonly", step=0)))
    else:
        execute(json.loads(a.seal.read_bytes()), a.output)


if __name__ == "__main__":
    main()

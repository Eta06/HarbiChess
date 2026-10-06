"""Explicit synthetic proof: whole8/pause4/fresh8 and six strict fresh loads."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from model import FEATURE_SCHEMA
from native import MATH, bits_equal

HERE = Path(__file__).parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--first", type=float, required=True)
    p.add_argument("--seconds", type=int, default=120)
    a = p.parse_args()
    torch.set_num_threads(1)
    first, deadline = a.first, a.first + a.seconds
    if not first <= time.time() < deadline or not 0 < a.seconds <= 120:
        raise ValueError("prospective synthetic clock120 maximum")
    a.output.mkdir(exist_ok=False)
    dataset = dict(
        schema="own-kingbucket-sparse-training-data-v1",
        phase="synthetic-test",
        rows=[
            dict(indices=sorted({4, 8, 50 + i}), prior_logit=(i - 8) / 10, target=(i % 3 - 1) * 0.5)
            for i in range(16)
        ],
    )
    data = a.output / "synthetic-data.json"
    data.write_text(json.dumps(dataset, sort_keys=True))
    contract = dict(
        phase="synthetic-test",
        updates=8,
        seed=17,
        math=MATH,
        feature_schema=FEATURE_SCHEMA,
        dataset_sha256=sha(data),
        source_sha256={
            str(HERE / name): sha(HERE / name) for name in ["model.py", "native.py", "train.py"]
        },
        original_first_epoch=first,
        original_deadline_epoch=deadline,
        core_source_repo="/workspace/work/harbichess/cpu-additive-source-6fcc8b4",
        actual_data_or_teacher_used=False,
        core_source_commit="6fcc8b476d25495d1c9c413e55b2c7ba4794013e",
    )
    contract_path = a.output / "contract.json"
    contract_path.write_text(json.dumps(contract, sort_keys=True))
    commands = []

    def run(out, stop, native=None, audit=False):
        cmd = [
            sys.executable,
            str(HERE / "train.py"),
            "--contract",
            str(contract_path),
            "--dataset",
            str(data),
            "--output",
            str(out),
            "--stop",
            str(stop),
        ]
        if native:
            cmd += ["--resume", str(native), "--resume-sha256", sha(native)]
        if audit:
            cmd += ["--audit-only"]
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError("original synthetic clock expired")
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=remaining,
            env={
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
        commands.append(
            dict(
                command=cmd,
                code=proc.returncode,
                stdout=proc.stdout.decode(),
                stderr=proc.stderr.decode(),
            )
        )
        if proc.returncode:
            raise RuntimeError("fresh synthetic child failed; captured logs preserved")

    try:
        run(a.output / "whole", 8)
        run(a.output / "pause", 4)
        run(a.output / "resume", 8, a.output / "pause/native.pt")
        whole = torch.load(a.output / "whole/native.pt", weights_only=False)
        resumed = torch.load(a.output / "resume/native.pt", weights_only=False)
        if not bits_equal(whole, resumed):
            raise ValueError("all-model/Adam/global/private sampler RNG storage mismatch")
        paths = [
            a.output / "whole/initial.pt",
            a.output / "whole/native.pt",
            a.output / "pause/initial.pt",
            a.output / "pause/native.pt",
            a.output / "resume/initial.pt",
            a.output / "resume/native.pt",
        ]
        for i, path in enumerate(paths):
            state = torch.load(path, weights_only=False)
            run(a.output / ("audit-" + str(i)), state["step"], path, True)
        result = dict(
            status="PASS-synthetic-whole8-pause4-fresh8-six-native-loads",
            commands=commands,
            first=first,
            deadline=deadline,
            finished=time.time(),
            full_native_bytes=(a.output / "whole/native.pt").stat().st_size,
            all_storagebits_equal=True,
            actual_teacher_or_chess_data_used=False,
            raw_torch_zip_byte_identity_claimed=False,
        )
        if result["finished"] > deadline:
            raise TimeoutError("proof final deadline exceeded")
    except BaseException as error:
        result = dict(
            status="FAILED-preserved",
            error=repr(error),
            commands=commands,
            first=first,
            deadline=deadline,
            finished=time.time(),
        )
        (a.output / "result.json").write_text(json.dumps(result, sort_keys=True, indent=2))
        raise
    (a.output / "result.json").write_text(json.dumps(result, sort_keys=True, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "commands"}))


if __name__ == "__main__":
    main()

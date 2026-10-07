"""ROOT actual same-parent search instrumentation check; no actor or training."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
QUALIFIER = BASE / "tdleaf-search-equivalence-v4-protected-source-cohort"
PRODUCER = BASE / "tdleaf-human-prior-own-v2-qualified-producer-v3"
END = 1791448916.685839


def ref(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(20262905, 20262906), required=True)
    parser.add_argument("--cpu-core", type=int, choices=(0, 2), required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu_core})
    records = BASE / f"TDLeaf-v5-equivalence-actual-{args.seed}"
    records.mkdir(exist_ok=False)
    output = Path("/dev/shm/harbichess-TDLeaf-search-equivalence-v5") / str(args.seed)
    output.mkdir(exist_ok=False, parents=True)
    human_reg = (
        BASE
        / f"human-random-collect-actual-{args.seed}"
        / "collection-registration.json"
    )
    ancestral_reg = BASE / "nnue-own-producer-v2/registrations" / f"{args.seed}.json"
    current = json.loads(human_reg.read_bytes())
    ancestral = json.loads(ancestral_reg.read_bytes())
    first = time.time()
    deadline = min(first + 600, END)
    reg = dict(current)
    reg.update(
        schema="tdleaf-search-equivalence-registration-v4",
        status="registered",
        first_epoch=first,
        deadline_epoch=deadline,
        cpu_core=args.cpu_core,
        source=dict(
            baseline_search=ref(QUALIFIER / "search_original.py"),
            pv_search=ref(QUALIFIER / "search_pv.py"),
            parent_runner=ref(PRODUCER / "run_collection.py"),
        ),
        root_pool=ref(ancestral["root_pool"]["path"]),
        protected_aliases=ref(
            "/dev/shm/harbichess-continuation-20261007/metadata-20262906/protected-aliases.bin"
        ),
        root_count=24,
        max_attempted_roots=128,
        selection_rule="first24-source-order-current-query-PV-protected-clear-v1",
        search=dict(nodes=8192, qdepth=2, max_depth=8),
        teacher_labels_used=False,
        updates=0,
        output_path=str(output / "report.json"),
        packet_dir=str(output / "packets"),
        current_parent_registration=ref(human_reg),
        ancestral_registration=ref(ancestral_reg),
        qualifier_source_inventory=ref(QUALIFIER / "source-inventory.json"),
    )
    registration = records / "qualification-registration.json"
    write(registration, reg)
    command = [
        sys.executable,
        str(QUALIFIER / "qualify.py"),
        "--registration",
        str(registration),
    ]
    receipt = dict(
        schema="ROOT-TDLeaf-v3-actual-equivalence-owner-v1",
        status="running",
        first=first,
        deadline=deadline,
        operator_end_epoch=END,
        seed=args.seed,
        cpu_core=args.cpu_core,
        helper=ref(__file__),
        registration=ref(registration),
        command=command,
        teacher_labels_used=False,
        updates=0,
    )
    write(records / "registration.json", receipt)
    try:
        with (
            (output / "stdout.log").open("xb") as out,
            (output / "stderr.log").open("xb") as err,
        ):
            subprocess.run(
                command,
                stdout=out,
                stderr=err,
                check=True,
                timeout=max(0.001, deadline - time.time()),
            )
        report = json.loads((output / "report.json").read_bytes())
        if report["status"] != "PASS":
            raise ValueError("actual same24 protected/search/wall qualification failed")
        if time.time() >= deadline:
            raise TimeoutError("original600-second qualification clock")
        receipt.update(
            status="PASS-actual24-fullquery-history-HEX-and-search-instrumentation-not-strength",
            report=ref(output / "report.json"),
        )
    except BaseException as error:
        receipt.update(status="FAILED-preserved", error=repr(error))
        raise
    finally:
        receipt["finished"] = time.time()
        write(records / "result.json", receipt)


if __name__ == "__main__":
    main()

"""ROOT qualified same-parent teacher-free TDLeaf collection, new original clocks."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
STAGE = BASE / "tdleaf-human-prior-own-v2-qualified-producer-v3"
END = 1791448916.685839


def ref(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(20262905, 20262906), required=True)
    parser.add_argument("--cpu-core", type=int, choices=(0, 2), required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu_core})
    qualifications = []
    for seed in (20262905, 20262906):
        result_path = BASE / f"TDLeaf-v5-equivalence-actual-{seed}/result.json"
        result = json.loads(result_path.read_bytes())
        report_path = Path(result["report"]["path"])
        if ref(report_path) != result["report"]:
            raise ValueError("actual qualification report changed")
        report = json.loads(report_path.read_bytes())
        if (
            result["status"]
            != "PASS-actual24-fullquery-history-HEX-and-search-instrumentation-not-strength"
            or report["status"] != "PASS"
            or report["root_count"] != 24
            or not report["all_attempts_in_search_equality_and_wall"]
            or report["source"]["parent_runner"] != ref(STAGE / "run_collection.py")
            or report["source"]["pv_search"] != ref(STAGE / "search_pv.py")
            or report["source"]["baseline_search"] != ref(STAGE / "search_original.py")
        ):
            raise ValueError("both same-source actual qualifications must PASS")
        qualifications.append(
            dict(seed=seed, result=ref(result_path), report=ref(report_path))
        )
    human_path = (
        BASE / f"human-random-collect-actual-{args.seed}/collection-registration.json"
    )
    human = json.loads(human_path.read_bytes())
    ancestral_path = BASE / f"nnue-own-producer-v2/registrations/{args.seed}.json"
    ancestral = json.loads(ancestral_path.read_bytes())
    records = BASE / f"TDLeaf-collection-v3-actual-{args.seed}"
    records.mkdir(exist_ok=False)
    logs = Path(
        f"/dev/shm/harbichess-continuation-20261007/TDLeaf-collection-v3-logs-{args.seed}"
    )
    logs.mkdir(parents=True, exist_ok=False)
    pool = Path(
        f"/dev/shm/harbichess-continuation-20261007/TDLeaf-collection-v3-pool-{args.seed}.json"
    )
    first = time.time()
    deadline = min(first + 7200, END)
    seal = dict(
        schema="human-prior-tdleaf-collection-build-seal-v2",
        status="registered",
        first=first,
        deadline=deadline,
        operator_end_epoch=END,
        cpu_core=args.cpu_core,
        parent_admission_seal=human["parent_admission_seal"],
        parent_admission_result=human["parent_admission_result"],
        parent_helpers=human["parent_helpers"],
        search_helper=human["search_helper"],
        ancestral_root_pool=ref(ancestral["root_pool"]["path"]),
        ancestral_selection=ref(ancestral["root_pool"]["selection_path"]),
        protected_aliases=human["protected_aliases"],
        root_pool_output=str(pool),
        actual_both_parent_search_qualifications=qualifications,
        qualified_source_inventory=ref(STAGE / "source-inventory.json"),
    )
    seal_path = records / "collection-build-seal.json"
    write(seal_path, seal)
    registration = records / "collection-registration.json"
    commands = [
        [
            sys.executable,
            str(STAGE / "metadata_factory.py"),
            "--seal",
            str(seal_path),
            "--output",
            str(registration),
        ],
        [
            sys.executable,
            str(STAGE / "run_collection.py"),
            "--registration",
            str(registration),
        ],
    ]
    receipt = dict(
        schema="ROOT-actual-TDLeaf-current-parent-collection-v3",
        status="running",
        seed=args.seed,
        cpu_core=args.cpu_core,
        first=first,
        deadline=deadline,
        operator_end_epoch=END,
        helper=ref(__file__),
        seal=ref(seal_path),
        source_inventory=ref(STAGE / "source-inventory.json"),
        parent_registration=ref(human_path),
        ancestral_registration=ref(ancestral_path),
        both_actual_qualifications=qualifications,
        commands=commands,
        completed=[],
        teacher_weights_used=False,
        teacher_labels_used=False,
    )
    write(records / "registration.json", receipt)
    try:
        for i, command in enumerate(commands):
            with (
                (logs / f"{i}.stdout.log").open("xb") as out,
                (logs / f"{i}.stderr.log").open("xb") as err,
            ):
                subprocess.run(
                    command,
                    stdout=out,
                    stderr=err,
                    check=True,
                    timeout=max(0.001, deadline - time.time()),
                )
            receipt["completed"].append(i)
        reg = json.loads(registration.read_bytes())
        raw_path = Path(reg["output_path"]) / "receipt.json"
        raw = json.loads(raw_path.read_bytes())
        if (
            raw["status"] != "PASS-exact-row-budget"
            or raw["registration_sha256"] != ref(registration)["sha256"]
            or time.time() >= deadline
        ):
            raise ValueError("exact original-clock collection and registration binding")
        receipt.update(
            status="PASS-actual1024-TDLeaf-current-parent-selfplay-not-strength",
            collection_registration=ref(registration),
            raw_receipt=ref(raw_path),
        )
    except BaseException as error:
        receipt.update(status="FAILED-preserved", error=repr(error))
        raise
    finally:
        receipt["finished"] = time.time()
        write(records / "result.json", receipt)


if __name__ == "__main__":
    main()

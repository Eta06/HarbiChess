"""ROOT-only single owned CPU formal worker; fresh five arms, no resume/retry."""

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

from bindings import SEEDS, TASKS, read, sha
from runtime_support import guard_factory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("registration", "study", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    a = parser.parse_args()
    reg = json.loads(a.registration.read_bytes())
    if (reg["schema"] != "ONE-NNUE-formal960-runtime-registration-v2"
            or reg["status"] != "registered-no-games-yet"
            or not reg["first"] <= time.time() < reg["deadline"] <= reg["operator_end_epoch"]):
        raise ValueError("original formal owner clock/ONE registration")
    views = {seed: read(reg["views"][str(seed)]) for seed in SEEDS}
    core = Path(views[SEEDS[0]]["core_repo"])
    spec = importlib.util.spec_from_file_location(
        "formal_owned", core / "experiments/ufuk/cpu-fix-v1/qualify_cli.py")
    owned = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owned)
    owned.check_source(core, views[SEEDS[0]]["source_commit"])
    for name, digest in reg["helper_sha256"].items():
        if sha(a.study / name) != digest:
            raise ValueError("complete formal helper closure SHA")
    guard = guard_factory(core, reg["cpu_core"], reg["deadline"],
                          a.output, reg["reserved_output_bytes"])
    import torch
    from admission import child, import_nnue

    torch.set_num_threads(1)
    _, native, _, _ = import_nnue(views[SEEDS[0]])
    admissions = [child(views[s], s, native)[2] for s in SEEDS]
    if time.time() >= reg["deadline"]:
        raise TimeoutError("original formal clock includes admission")
    a.output.mkdir(exist_ok=False)
    owned.publish(a.output / "native-admission.json", admissions)
    rows = []
    for arm, opponent in TASKS:
        for seed in SEEDS:
            guard()
            q = views[seed]
            tag = f"{seed}-{arm}-vs-{opponent}"
            first = time.time()
            deadline = min(reg["deadline"], first + reg["per_arm_seconds"])
            cmd = [os.sys.executable, str(a.study / "tournament.py"),
                   "--protocol", reg["views"][str(seed)]["path"],
                   "--candidate", q["models"][str(seed)][arm]["path"],
                   "--opponent", opponent if opponent == "SF512" else
                   q["models"][str(seed)][opponent]["path"],
                   "--seed", str(seed), "--book", q["book_path"],
                   "--stockfish", q["stockfish_path"], "--deadline-epoch", str(deadline),
                   "--progress", str(a.output / f"{tag}.progress.jsonl"),
                   "--output", str(a.output / f"{tag}.json")]
            row = dict(seed=seed, arm=arm, opponent=opponent, command=cmd,
                       original_first_epoch=first, original_deadline_epoch=deadline,
                       view_sha256=reg["views"][str(seed)]["sha256"])
            owned.publish(a.output / f"{tag}.invocation.json", row)
            try:
                for name, digest in reg["helper_sha256"].items():
                    if sha(a.study / name) != digest:
                        raise ValueError("immutable helper changed")
                with (a.output / f"{tag}.stdout.log").open("xb") as out, (
                        a.output / f"{tag}.stderr.log").open("xb") as err:
                    owned.run_owned(cmd, cwd=core,
                                    env={**os.environ, "PYTHONPATH": str(core / "src"),
                                         "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                                         "OPENBLAS_NUM_THREADS": "1"}, stdout=out, stderr=err,
                                    deadline=deadline)
                packet = json.loads((a.output / f"{tag}.json").read_bytes())
                if len(packet["games"]) != 96 or packet["finished_epoch"] > deadline:
                    raise ValueError("exact96 closedgames within original phase")
                row.update(status="completed-games-awaiting-independent-audit",
                           result_sha256=sha(a.output / f"{tag}.json"))
            except Exception as exc:
                row.update(status="failed-preserved-no-retry", error=repr(exc))
            row["finished_epoch"] = time.time()
            owned.publish(a.output / f"{tag}.owner-result.json", row)
            rows.append(row)
            if row["status"].startswith("failed"):
                break
        if rows[-1]["status"].startswith("failed"):
            break
    guard()
    owned.publish(a.output / "cohort-result.json",
                  dict(schema="ONE-formal960-owned-cohort-v2",
                       status="completed-960-awaiting-independent-audit"
                       if len(rows) == 10 and all(r["status"].startswith("completed") for r in rows)
                       else "INCOMPLETE-preserved-no-retry",
                       registration_sha256=sha(a.registration), rows=rows,
                       helper_sha256=sha(__file__), finished_epoch=time.time()))


if __name__ == "__main__":
    main()

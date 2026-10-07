"""ROOT-only completed fixed64 admission and ONE observed clock freeze; no forwards/search."""

import argparse
import json
import math
import time
from pathlib import Path

from admission import child, import_nnue
from teacher_admission import sha
from wiring import validate_wiring


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--draft", type=Path, required=True)
    p.add_argument("--draft-sha256", required=True)
    p.add_argument("--first", type=float, required=True)
    p.add_argument("--operator-end", type=float, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if sha(a.draft) != a.draft_sha256:
        raise ValueError("exact ROOT-reviewed completed-input draft")
    if (
        not all(math.isfinite(x) for x in [a.first, a.operator_end])
        or not a.first <= time.time() < a.first + 600
        or a.first + 7200 + 900 > a.operator_end
    ):
        raise ValueError("actual first, no future first/reset; original profile+whole+audit fits")
    q = json.loads(a.draft.read_bytes())
    validate_wiring(q)
    q["variant_admission_schema"] = "NNUE-typed-own64-known160-admission-v1"
    q["ROOToperator_end_epoch"] = a.operator_end
    q["profile_first_epoch"] = q["original_first_epoch"] = a.first
    q["profile_deadline_epoch"] = a.first + 600
    q["original_deadline_epoch"] = a.first + 7200
    q["audit_deadline_epoch"] = a.first + 7200 + 900
    q["clock_accounting"] = (
        "ONE observed originalfirst; native admission+profile INCLUDED in whole7200"
    )
    q["whole_seconds"] = 7200
    for key in q["models"]:
        if q["models"][key]["parent"] != q["teachers"][key]["candidate"]:
            raise ValueError("SAME exact named teacher comparator; no endpoint substitution")
        for role, row in q["models"][key].items():
            actual = sha(row["path"])
            if role == "learned":
                row["sha256"] = actual
                q["children"][key]["candidate"] = row
            elif row["sha256"] != actual:
                raise ValueError("frozen parent/E8 SHA changed")
    q["arena_helper_sha256"] = {
        x.name: sha(x) for x in a.draft.parent.glob("*.py") if not x.name.startswith("test_")
    }
    _, native, _, _ = import_nnue(q)
    admissions = [child(q, seed, native)[2] for seed in q["match_seeds"]]
    if time.time() >= q["profile_deadline_epoch"]:
        raise TimeoutError("same profile600 includes admission")
    q["status"] = "ROOT-sealed-fixed64-awaiting-real48-parity48-profile-and160-games"
    with a.output.open("x") as stream:
        json.dump(q, stream, sort_keys=True, indent=2, allow_nan=False)
    print(
        json.dumps(
            dict(
                protocol_sha256=sha(a.output),
                admissions=admissions,
                native_metadata_reads_only=True,
                search_calls=0,
                forwards=0,
                optimizer_updates=0,
            )
        )
    )


if __name__ == "__main__":
    main()

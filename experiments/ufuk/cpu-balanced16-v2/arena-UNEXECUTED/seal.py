"""ROOT-only completed fresh48 arena seal; no search/SGD/forward/process launch."""

import argparse
import json
import math
import time
from pathlib import Path

from native_admission import admit
from support import sha

HERE = Path(__file__).parent
END = 1791273600
SEEDS = (20262905, 20262906)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--draft-sha256", required=True)
    p.add_argument("--profile-first", type=float, required=True)
    p.add_argument("--arena-first", type=float, required=True)
    p.add_argument("--output", type=Path, required=True)
    for suffix in ["05", "06"]:
        p.add_argument("--fit" + suffix + "-registration", type=Path, required=True)
        p.add_argument("--fit" + suffix + "-registration-sha256", required=True)
    a = p.parse_args()
    if sha(HERE / "protocol-DRAFT.json") != a.draft_sha256:
        raise ValueError("approved exact draft required")
    now = time.time()
    if (
        not all(math.isfinite(v) for v in [a.profile_first, a.arena_first])
        or not a.profile_first <= now < a.profile_first + 600
        or a.arena_first < a.profile_first
        or a.arena_first + 7200 + 900 > END
    ):
        raise ValueError("new prospective600/7200/900 clocks, not original fit reset")
    q = json.loads((HERE / "protocol-DRAFT.json").read_bytes())
    for seed, suffix in zip(SEEDS, ["05", "06"], strict=True):
        path = getattr(a, "fit" + suffix + "_registration")
        expected = getattr(a, "fit" + suffix + "_registration_sha256")
        if sha(path) != expected:
            raise ValueError("actual fit registration hash")
        reg = json.loads(path.read_bytes())
        if (
            reg["schema"] != "own-balanced-risk-ROOT-registration-v2"
            or reg["status"] != "registered"
            or reg["seed"] != seed
            or reg["source_commit"] != q["source_commit"]
        ):
            raise ValueError("exact actual registered fit seed/source")
        q["fit_inputs"][str(seed)] = dict(path=str(path), sha256=expected)
        row = q["models"][str(seed)]["learned"]
        for k, h in [
            ("path", "sha256"),
            ("native_path", "native_sha256"),
            ("fresh0_path", "fresh0_sha256"),
            ("fit_result_path", "fit_result_sha256"),
        ]:
            row[h] = sha(row[k])
    q["profile_first_epoch"], q["profile_deadline_epoch"] = a.profile_first, a.profile_first + 600
    q["original_first_epoch"], q["original_deadline_epoch"] = a.arena_first, a.arena_first + 7200
    q["audit_deadline_epoch"] = a.arena_first + 7200 + 900
    q["status"] = "ROOT-sealed-fresh48-awaiting-trained24-profile-and192-arena"
    q["native_qualification"] = (
        "exact actual12 fresh proof loads plus both final48 strict admission"
    )
    q.pop("fit_cohort_path", None)
    q.pop("fit_cohort_sha256", None)
    q["arena_helper_sha256"] = {n: sha(HERE / n) for n in q["arena_helper_sha256"]}
    q["seal_helper_sha256"] = sha(Path(__file__))
    admissions = [admit(q, seed) for seed in SEEDS]
    with a.output.open("x") as stream:
        json.dump(q, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            dict(
                status="sealed-awaiting-profile-not-strength",
                protocol_sha256=sha(a.output),
                admissions=admissions,
                optimizer_steps=0,
                forward_calls=0,
                search_calls=0,
            )
        )
    )


if __name__ == "__main__":
    main()

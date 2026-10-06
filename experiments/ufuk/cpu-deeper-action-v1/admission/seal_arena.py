"""ROOT-only immutable ACTION arena seal; no search, forward or optimizer step."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

HERE = Path(__file__).parent
SEEDS = (20262905, 20262906)
END = 1791273600


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def checked_json(path, expected):
    if sha(path) != expected:
        raise ValueError("sealed source/receipt SHA differs: " + Path(path).name)
    return json.loads(Path(path).read_bytes())


def validate_receipts(proof, fits, fit_protocol):
    if (proof.get("status")
            != "PASS-two-ACTION-realdata-proofs-8-4-fresh8-and12-fresh-native-loads-not-strength"
            or len(proof.get("rows", [])) != 2
            or proof["finished_epoch"] > proof["deadline_epoch"]
            or fits.get("status")
            != "PASS-two-ACTION-FRESH64-fits-and-four-fresh0-final64-native-loads-not-strength"
            or fits["finished_epoch"] > fits["deadline_epoch"]
            or fits.get("qualification_state_imported") is not False
            or fits.get("teacher_labels") != 0):
        raise ValueError("actual completed proof/fits required, original clocks preserved")
    rows = {row["seed"]: row for row in fits["rows"]}
    if len(fits["rows"]) != 2 or set(rows) != set(SEEDS):
        raise ValueError("both exact seeds, no duplicates")
    for seed in SEEDS:
        if (fits["first_epoch"] != fit_protocol["first_by_seed"][str(seed)]
                or fits["deadline_epoch"] != fit_protocol["deadline_by_seed"][str(seed)]):
            raise ValueError("original fixed fit clock differs")
        loads = rows[seed]["fresh_interpreter_loads"]
        if (len(loads) != 2 or [r["step"] for r in loads] != [0, 64]
                or any(r["receipt"] != {"status": "strict-action-native-load-PASS",
                                       "step": r["step"]} for r in loads)):
            raise ValueError("four actual fresh0/final64 strict interpreter loads required")
    for row in proof["rows"]:
        if (row.get("schema") != "own-action-native-qualification-v1"
                or row.get("status") != "PASS-six-native-freshloads-and-8-4-fresh8-not-strength"
                or row["finished"] > row["deadline"]
                or sum("--audit-only" in c for c in row["commands"]) != 6):
            raise ValueError("original twelve fresh proof loads required")
    return rows


def seal(args):
    q = checked_json(HERE / "arena/protocol-DRAFT.json", args.draft_sha256)
    reg = args.registration_directory.resolve()
    fit = checked_json(reg / "fit-protocol.json", args.fit_protocol_sha256)
    proof_protocol = checked_json(reg / "proof-protocol.json", args.proof_protocol_sha256)
    if any(proof_protocol[k] != fit[k] for k in ["inputs", "helpers", "math", "source_commit"]):
        raise ValueError("proof and fresh fit must bind identical source/data/math")
    proof = checked_json(reg / "proofs-finished.json", args.proof_sha256)
    for row in proof["rows"]:
        for command in row["commands"]:
            if ("--protocol-sha256" not in command
                    or command[command.index("--protocol-sha256") + 1]
                    != args.proof_protocol_sha256):
                raise ValueError("actual proof command source/protocol binding differs")
    fits = checked_json(reg / "fits-and-freshloads-finished.json", args.fits_sha256)
    rows = validate_receipts(proof, fits, fit)
    if (fit["schema"] != "own-deeper-bestmove-training-protocol-v1"
            or fit["source_commit"] != q["source_commit"]
            or fit.get("fixed_updates") != 64 or fit.get("phase") != "fit"
            or fit["helpers"] != q["action_helpers"]):
        raise ValueError("same frozen eight-source/action64 protocol")
    now = time.time()
    if (not all(math.isfinite(v) for v in [args.profile_first, args.arena_first])
            or not args.profile_first <= now < args.profile_first + 600
            or args.arena_first < args.profile_first
            or args.arena_first + 7200 + 900 > END):
        raise ValueError("new prospective profile/arena/audit clocks")
    for seed in SEEDS:
        s = str(seed)
        incoming = fit["inputs"][s]
        for path_key, hash_key in [("labels_path", "labels_sha256"),
                                   ("producer_registration_path", "producer_registration_sha256")]:
            if sha(incoming[path_key]) != incoming[hash_key]:
                raise ValueError("actual source/own-label binding changed")
        q["ACTION_fit_inputs"][s] = {
            "labels_sha256": incoming["labels_sha256"],
            "producer_registration_sha256": incoming["producer_registration_sha256"],
            "protocol_sha256": args.fit_protocol_sha256,
        }
        role = q["models"][s]["learned"]
        role["sha256"] = sha(role["path"])
        role["native_sha256"] = sha(role["native_path"])
        zero_path = Path(role["native_path"]).with_name("step-00000000.json")
        zero, final = [json.loads(p.read_bytes()) for p in [zero_path, Path(role["native_path"])]]
        candidate = json.loads(Path(role["path"]).read_bytes())
        if (zero["step"] != 0 or final["step"] != 64
                or zero["contract"] != final["contract"]
                or final["contract"]["deadline"] != fits["deadline_epoch"]
                or final["contract"]["protocol_sha256"] != args.fit_protocol_sha256
                or canonical(final["model"]) != canonical(candidate)):
            raise ValueError("full actual fresh0/final64/model/clock binding differs")
        if [r["native_sha256"] for r in rows[seed]["fresh_interpreter_loads"]] != [
                sha(zero_path), role["native_sha256"]]:
            raise ValueError("actual fresh loads do not bind these native bytes")
        role["fresh0_path"], role["fresh0_sha256"] = str(zero_path), sha(zero_path)
    for name, expected in q["action_helpers"].items():
        if sha(Path(q["action_helper_directory"]) / name) != expected:
            raise ValueError("qualified frozen action source changed")
    q["profile_first_epoch"] = args.profile_first
    q["profile_deadline_epoch"] = args.profile_first + 600
    q["original_first_epoch"] = args.arena_first
    q["original_deadline_epoch"] = args.arena_first + 7200
    q["audit_deadline_epoch"] = args.arena_first + 7200 + 900
    q["fit_cohort_path"] = str(reg / "fits-and-freshloads-finished.json")
    q["fit_cohort_sha256"] = args.fits_sha256
    q["status"] = "ROOT-sealed-actual-ACTION64-native-awaiting-new-profile-and-arena"
    q["native_qualification"] = {
        "proof_path": str(reg / "proofs-finished.json"), "proof_sha256": args.proof_sha256,
        "fits_path": str(reg / "fits-and-freshloads-finished.json"),
        "fits_sha256": args.fits_sha256, "fit_protocol_path": str(reg / "fit-protocol.json"),
        "fit_protocol_sha256": args.fit_protocol_sha256,
        "proof_protocol_sha256": args.proof_protocol_sha256,
        "original_fit_first": fits["first_epoch"], "original_fit_deadline": fits["deadline_epoch"],
        "read_only_seal_helper_sha256": sha(Path(__file__)),
    }
    q["arena_helper_sha256"] = {n: sha(HERE / "arena" / n) for n in q["arena_helper_sha256"]}
    import sys

    sys.path.insert(0, str(HERE / "arena"))
    try:
        from native_admission import admit

        admissions = [admit(q, seed) for seed in SEEDS]
    finally:
        sys.path.pop(0)
    with args.output.open("xb") as stream:
        stream.write(canonical(q) + b"\n")
    return {"status": "sealed-awaiting-profile-not-strength", "path": str(args.output),
            "protocol_sha256": sha(args.output), "native_admissions": admissions,
            "optimizer_steps": 0, "search_calls": 0, "forward_calls": 0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["registration-directory", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    for name in ["draft-sha256", "fit-protocol-sha256", "proof-protocol-sha256",
                 "proof-sha256", "fits-sha256"]:
        p.add_argument("--" + name, required=True)
    p.add_argument("--profile-first", type=float, required=True)
    p.add_argument("--arena-first", type=float, required=True)
    print(json.dumps(seal(p.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()

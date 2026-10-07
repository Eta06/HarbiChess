"""Strict trained NNUE admission; original fit clocks are evidence, not reset."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(row):
    path = Path(row["path"])
    if sha(path) != row["sha256"]:
        raise ValueError("pinned artifact/source changed: " + path.name)
    return path


def read(row):
    return json.loads(pin(row).read_bytes())


def module(row, name):
    path = pin(row)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def check_contract(c):
    for key in ("source_sha256", "inference_source_sha256"):
        for path, digest in c[key].items():
            pin(dict(path=path, sha256=digest))
    for stem in ("target_provenance", "prior_helper"):
        pin(dict(path=c[stem + "_path"], sha256=c[stem + "_sha256"]))
    if c["teacher_bootstrap_is_selflearning"] is not False:
        raise ValueError("teacher provenance must not claim own learning")


def receipt(r, contract_ref, expected_steps):
    if (r["status"] != "PASS-readonly-loads-and-fixed-phase-not-strength"
            or r["contract_sha256"] != contract_ref["sha256"]
            or not r["first"] < r["finished"] < r["deadline"] <= 1791273600
            or r["raw_zip_identity_claimed"] is not False):
        raise ValueError("actual successful original proof/fit receipt")
    audits = [c for c in r["commands"] if "--audit-only" in c["command"]]
    if len(audits) != len(expected_steps) or any(c["code"] != 0 for c in r["commands"]):
        raise ValueError("all fresh processes required")
    for c, step in zip(audits, expected_steps, strict=True):
        cmd = c["command"]
        packet = json.loads(c["stdout"])
        if (packet != dict(status="PASS-strict-native-readonly", step=step)
                or c["stderr"] or cmd[cmd.index("--contract") + 1] != contract_ref["path"]
                or int(cmd[cmd.index("--stop") + 1]) != step):
            raise ValueError("fresh native audit output/contract/counter")
        path = cmd[cmd.index("--resume") + 1]
        h = cmd[cmd.index("--resume-sha256") + 1]
        if r["native_sha256"].get(path) != h:
            raise ValueError("fresh audited native binding")
        pin(dict(path=path, sha256=h))


def teacher(info, native):
    fit, proof = read(info["contract"]), read(info["proof_contract"])
    check_contract(fit)
    check_contract(proof)
    if (fit["phase"] != "teacher-bootstrap" or fit["updates"] != 256
            or fit["math"] != native.MATH or fit["seed"] != info["seed"]):
        raise ValueError("fixed actual teacher256 contract")
    ignored = {"original_first_epoch", "original_deadline_epoch", "inference_source_sha256",
               "proof_receipt_path", "proof_receipt_sha256"}
    if {k: v for k, v in fit.items() if k not in ignored} != {
            k: v for k, v in proof.items() if k not in ignored}:
        raise ValueError("proof/fit identical model/data/math/source/baseline")
    extra = set(fit["inference_source_sha256"]) - set(proof["inference_source_sha256"])
    if (extra != {info["proof_result"]["path"]}
            or any(fit["inference_source_sha256"].get(k) != v
                   for k, v in proof["inference_source_sha256"].items())
            or fit["proof_receipt_path"] != info["proof_result"]["path"]
            or fit["proof_receipt_sha256"] != info["proof_result"]["sha256"]):
        raise ValueError("production exact proof binding")
    data = pin(info["dataset"])
    if sha(data) != fit["dataset_sha256"]:
        raise ValueError("exact teacher data")
    provenance = json.loads(Path(fit["target_provenance_path"]).read_bytes())
    if (provenance["inputs"]["labels"] != info["labels"]
            or provenance["dataset_sha256"] != fit["dataset_sha256"]):
        raise ValueError("profile roots exact teacher-data provenance")
    for ref in provenance["inputs"].values():
        if isinstance(ref, dict) and "path" in ref:
            pin(ref)
    pr, fr = read(info["proof_result"]), read(info["fit_result"])
    receipt(pr, info["proof_contract"], [0, 8, 0, 4, 4, 8])
    receipt(fr, info["contract"], [0, 256])
    if not pr["full_payload_bits_equal"] or pr["mode"] != "proof" or fr["mode"] != "fresh-fit":
        raise ValueError("fresh proof/fresh production distinct scopes")
    for c, r in ((proof, pr), (fit, fr)):
        if (c["original_first_epoch"], c["original_deadline_epoch"]) != (r["first"], r["deadline"]):
            raise ValueError("original clocks exact")
        for path, h in r["native_sha256"].items():
            pin(dict(path=path, sha256=h))
            native.load_native(path, c)
    final = native.load_native(pin(info["native"]), fit).native()
    initial = native.load_native(pin(info["initial"]), fit).native()
    import torch

    candidate = torch.load(pin(info["candidate"]), map_location="cpu", weights_only=False)
    if (set(candidate) != {"schema", "contract", "model"} or candidate["contract"] != fit
            or candidate["schema"] != "own-kingbucket-nnue16-model-v1"
            or final["step"] != 256 or initial["step"] != 0
            or not native.bits_equal(candidate["model"], final["model"])
            or not native.bits_equal(final["baseline"], initial["model"])
            or native.bits_equal(final["model"], initial["model"])):
        raise ValueError("full teacher model/immutable zero baseline/candidate exact")
    return candidate["model"], initial["model"], dict(
        teacher_candidate_sha256=info["candidate"]["sha256"], strict_native_loads=8,
        proof_fresh_loads=6, fit_fresh_loads=2, teacher_is_selflearning=False)

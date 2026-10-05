"""Read-only strict actual PST initial+fixed-final native/fit/model admission."""

import hashlib
import json
from pathlib import Path

from value import pst_module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_pst_fit(protocol):
    module = pst_module(protocol)
    import pst_native

    info = protocol["PST_fit_result"]
    if sha(info["path"]) != info["sha256"]:
        raise ValueError("immutable actual PST fit result differs")
    fit = json.loads(Path(info["path"]).read_text())
    if fit["status"] != "PASS-fixed-updates-and-strict-native-loads-not-strength":
        raise ValueError("actual fixed-fit and strict native PASS required")
    if fit["protocol_sha256"] != protocol["PST_training_protocol"]["sha256"] or (
        fit["finished_epoch"] > fit["phase_deadline_epoch"]
    ):
        raise ValueError("original fit protocol/clock differs")
    original = json.loads(Path(protocol["PST_training_protocol"]["path"]).read_text())
    for seed in protocol["match_seeds"]:
        models = protocol["models"][str(seed)]
        r = models["learned"]
        row = fit["seeds"][str(seed)]
        for name, field in [
            ("path", "sha256"),
            ("native_path", "native_sha256"),
            ("initial_native_path", "initial_native_sha256"),
            ("fit_contract_path", "fit_contract_sha256"),
        ]:
            if sha(r[name]) != r[field]:
                raise ValueError("PST actual model/native/contract SHA differs")
        if (
            row["candidate_sha256"] != r["sha256"]
            or row["final_native_sha256"] != r["native_sha256"]
        ):
            raise ValueError("original actual fit receipt payload binding differs")
        if row["fit_contract_sha256"] != r["fit_contract_sha256"]:
            raise ValueError("original actual fit contract differs")
        contract = json.loads(Path(r["fit_contract_path"]).read_text())
        if contract["schema"] != "classical-own-pst-fit-contract-v1" or (
            contract["source_commit"] != protocol["source_commit"]
            or contract["seed"] != seed
            or contract["protocol_sha256"] != fit["protocol_sha256"]
            or contract["original_deadline_epoch"] != fit["phase_deadline_epoch"]
            or contract["journal_sha256"] != original["inputs"][str(seed)]["journal_sha256"]
            or contract["config_sha256"] != original["inputs"][str(seed)]["config_sha256"]
            or contract["dataset_sha256"] != original["inputs"][str(seed)]["PST_dataset_sha256"]
            or contract["updates"] != min(1024, 4 * row["training_rows"] // 256)
            or contract["updates"] != row["updates"]
            or contract["updates"] != r["accepted_updates"]
        ):
            raise ValueError("fixed final input/source/update contract differs")
        for field, step in [("initial_native_path", 0), ("native_path", contract["updates"])]:
            state = pst_native.decode(Path(r[field]).read_bytes(), contract)
            if state["step"] != step:
                raise ValueError("initial/fixed-final strict native counter differs")
            if step == 0 and state["theta"] != [0.0] * 242:
                raise ValueError("original PST initialization not freshzero")
            if step and state["candidate"] != json.loads(Path(r["path"]).read_text()):
                raise ValueError("learned candidate differs from actual final native")
        module.load_pst(r["path"])
        prior = models["prior"]
        if sha(prior["path"]) != prior["sha256"]:
            raise ValueError("SAMEhumanprior SHA differs")
        packet = json.loads(Path(prior["path"]).read_text())
        if packet["schema"] != "classical-own-linear-value-v1" or packet["theta"] != [0.0] * 18:
            raise ValueError("explicit SAMEhumanprior model required")
        if sha(models["e8"]["path"]) != models["e8"]["sha256"]:
            raise ValueError("strongest E8 model differs")
    return True

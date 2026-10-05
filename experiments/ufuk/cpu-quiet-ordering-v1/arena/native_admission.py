"""No learned-value swap; strict quiet ordering native/source/data admission."""

import json
import random
from pathlib import Path

from value import quiet_model, sha


def verify_quiet_fit(protocol):
    model = quiet_model(protocol)
    fitpath = Path(protocol["ordering_fit_result"]["path"])
    if sha(fitpath) != protocol["ordering_fit_result"]["sha256"]:
        raise ValueError("actual fixed quiet fit result differs")
    fit = json.loads(fitpath.read_text())
    if (
        fit["status"] != "PASS-both-fixed-final-quiet-fits-not-strength"
        or fit["finished"] > fit["deadline"]
    ):
        raise ValueError("actual fixed-final quiet fit within original1800 required")
    training = protocol["ordering_training_protocol"]
    q = json.loads(Path(training["path"]).read_text())
    for seed in protocol["match_seeds"]:
        r = protocol["models"][str(seed)]["learned"]
        if sha(r["path"]) != r["sha256"] or fit["seed_results"][str(seed)]["sha256"] != r["sha256"]:
            raise ValueError("actual learned ordering candidate differs")
        packet = json.loads(Path(r["path"]).read_text())
        model.load_model(packet)
        for role, step in [("initial_native_path", 0), ("native_path", r["accepted_updates"])]:
            if sha(r[role]) != r[role.removesuffix("_path") + "_sha256"]:
                raise ValueError("initial/final ordering native SHA differs")
            n = json.loads(Path(r[role]).read_text())
            c = n["contract"]
            if (
                n["schema"] != "own-quiet-ordering-native-v1"
                or n["step"] != step
                or c["seed"] != seed
            ):
                raise ValueError("quiet native schema/counter/seed differs")
            if (
                c["source_commit"] != protocol["source_commit"]
                or c["protocol_sha256"] != training["sha256"]
            ):
                raise ValueError("original source/protocol differs")
            if (
                c["deadline"] != fit["deadline"]
                or c["updates"] != r["accepted_updates"]
                or (c["updates"] != min(1024, 4 * c["receipt"]["training_rows"] // 256))
            ):
                raise ValueError("fixed derived final endpoint/originalclock differs")
            if c["dataset_sha256"] != r["dataset_sha256"] or (
                c["journal_sha256"] != q["inputs"][str(seed)]["journal_sha256"]
                or c["config_sha256"] != q["inputs"][str(seed)]["config_sha256"]
            ):
                raise ValueError("actual original complete dataset differs")
            if model.model_dict(n["weights"]) != n["model"]:
                raise ValueError("native model storage differs")
            if step == 0 and not n["weights"] == n["m"] == n["v"]:
                raise ValueError("zero native arrays differ")
            if step == 0 and any(n["weights"]):
                raise ValueError("initialordering nonzero")
            if step and n["model"] != packet:
                raise ValueError("final candidate/native differs")
            import math

            for key in ["m", "v"]:
                if len(n[key]) != 268 or not all(math.isfinite(x) for x in n[key]):
                    raise ValueError("all Adam storage required")
            if any(x < 0 for x in n["v"]):
                raise ValueError("negative Adam variance")
            for key in ["global_rng", "sampler_rng"]:
                version, words, gaussian = n[key]
                random.Random().setstate((version, tuple(words), gaussian))
        prior = protocol["models"][str(seed)]["prior"]
        if prior != protocol["fixed_prior_model"]:
            raise ValueError("SAME humanprior control required")
    return True

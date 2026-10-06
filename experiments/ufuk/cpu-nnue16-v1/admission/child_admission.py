"""Future own64 admission, named teacher baseline and new data mandatory."""
import json
from pathlib import Path

from admission import pin, read, receipt, sha, teacher


def child(info, parent_info, native):
    parent, _, parent_receipt = teacher(parent_info, native)
    c, proof = read(info["contract"]), read(info["proof_contract"])
    if (c["phase"] != "own-learning" or c["updates"] != 64 or c["math"] != native.MATH
            or c["seed"] != parent_info["seed"]
            or c["bootstrap_candidate_path"] != parent_info["candidate"]["path"]
            or c["bootstrap_candidate_sha256"] != parent_info["candidate"]["sha256"]):
        raise ValueError("explicit same-seed teacher weights-only bridge; fixed own64")
    for contract in (c, proof):
        for key in ("source_sha256", "inference_source_sha256"):
            for path, h in contract[key].items():
                pin(dict(path=path, sha256=h))
        for stem in ("target_provenance", "prior_helper"):
            pin(dict(path=contract[stem + "_path"], sha256=contract[stem + "_sha256"]))
    for key in ("phase", "updates", "seed", "math", "dataset_sha256", "feature_schema",
                "source_sha256", "bootstrap_candidate_path", "bootstrap_candidate_sha256",
                "prior_helper_path", "prior_helper_sha256", "target_provenance_sha256"):
        if c[key] != proof[key]:
            raise ValueError("own proof/fit shared bindings")
    if sha(pin(info["dataset"])) != c["dataset_sha256"]:
        raise ValueError("exact new own dataset")
    collection = read(info["collection_receipt"])
    if (collection["schema"] != "own-nnue-ownq-collection-receipt-v1"
            or collection["status"] != "PASS-exact-row-budget"
            or collection["train_rows"] != 1024 or collection["teacher_labels_used"] is not False
            or collection["parent_candidate_sha256"] != parent_info["candidate"]["sha256"]
            or not collection["original_first_epoch"] < collection["finished_epoch"]
            <= collection["original_deadline_epoch"] <= 1791273600):
        raise ValueError("closed teacher-free own-search production receipt")
    provenance = json.loads(Path(c["target_provenance_path"]).read_bytes())
    if provenance["collection_receipt_sha256"] != info["collection_receipt"]["sha256"]:
        raise ValueError("converted own dataset bound to exact collection receipt")
    pr, fr = read(info["proof_result"]), read(info["fit_result"])
    receipt(pr, info["proof_contract"], [0, 8, 0, 4, 4, 8])
    receipt(fr, info["contract"], [0, 64])
    if not pr["full_payload_bits_equal"]:
        raise ValueError("own proof complete payload equality")
    for contract, result in ((proof, pr), (c, fr)):
        for path, h in result["native_sha256"].items():
            pin(dict(path=path, sha256=h))
            native.load_native(path, contract)
    initial = native.load_native(pin(info["initial"]), c).native()
    final = native.load_native(pin(info["native"]), c).native()
    import torch

    candidate = torch.load(pin(info["candidate"]), map_location="cpu", weights_only=False)
    if (set(candidate) != {"schema", "contract", "model"}
            or candidate["schema"] != "own-kingbucket-nnue16-model-v1"
            or candidate["contract"] != c or initial["step"] != 0 or final["step"] != 64
            or not native.bits_equal(initial["model"], parent)
            or initial["optimizer"]["state"]
            or not native.bits_equal(final["baseline"], parent)
            or not native.bits_equal(candidate["model"], final["model"])
            or native.bits_equal(final["model"], parent)):
        raise ValueError("own fullstate/empty initial Adam/frozen namedparent/changed final")
    return candidate["model"], dict(parent=parent_receipt,
                                    child_sha256=info["candidate"]["sha256"], own_updates=64)

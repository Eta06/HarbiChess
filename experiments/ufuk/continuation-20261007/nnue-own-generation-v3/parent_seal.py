"""ROOT metadata-only own-final parent seal factory; no model load."""

import argparse
import json
from pathlib import Path

from parent_bridge import canonical, sha, validate_metadata


def ref(path):
    p = Path(path).resolve()
    return {"path": str(p), "sha256": sha(p)}


def build(a):
    c = json.loads(Path(a.fit_contract).read_bytes())
    result = json.loads(Path(a.fit_result).read_bytes())
    if len(result["native_payloads"]) != 2:
        raise ValueError("two actual0/64 fit fresh loads")
    source = c["source_sha256"]

    def helper(name):
        matches = [p for p in source if Path(p).name == name]
        if len(matches) != 1:
            raise ValueError("unique original helper closure")
        return {"path": matches[0], "sha256": source[matches[0]]}

    spec = dict(
        schema="NNUE-own-generation-parent-admission-seal-v3",
        status="registered",
        generation=a.generation,
        seed=a.seed,
        weights_only=True,
        parent_contract=ref(a.fit_contract),
        parent_proof_contract=ref(a.proof_contract),
        parent_proof_result=ref(a.proof_result),
        parent_fit_result=ref(a.fit_result),
        parent_candidate=ref(a.candidate),
        parent_dataset=ref(a.dataset),
        parent_native={k: result["native_payloads"][1][k] for k in ("path", "sha256")},
        parent_model_helper=helper("model.py"),
        parent_native_helper=helper("native.py"),
    )
    validate_metadata(spec)
    return spec


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "fit-result",
        "fit-contract",
        "proof-result",
        "proof-contract",
        "candidate",
        "dataset",
        "output",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--generation", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args()
    s = build(a)
    with a.output.open("xb") as f:
        f.write(canonical(s) + b"\n")

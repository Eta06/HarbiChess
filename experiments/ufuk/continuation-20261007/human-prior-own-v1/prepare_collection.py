"""Metadata-only current zero/own parent collection seal, no model calls."""

import argparse
from pathlib import Path

from parent_bridge import canonical, sha, validate_admission_result


def ref(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=sha(path))


def build(a):
    parent, contract = validate_admission_result(ref(a.admission_seal), ref(a.admission_result))
    directory = a.inference_directory.resolve()
    h = dict(
        directory=str(directory),
        model_sha256=sha(directory / "model.py"),
        native_sha256=sha(directory / "native.py"),
        evaluator_sha256=sha(directory / "evaluator.py"),
        prior_path=contract["prior_helper_path"],
        prior_sha256=contract["prior_helper_sha256"],
        extension_path=str(a.extension.resolve()),
        extension_sha256=sha(a.extension),
    )
    return dict(
        schema="human-prior-own-collection-build-seal-v1",
        status="registered",
        first=a.first,
        deadline=a.deadline,
        operator_end_epoch=a.operator_end_epoch,
        cpu_core=a.cpu_core,
        parent_admission_seal=ref(a.admission_seal),
        parent_admission_result=ref(a.admission_result),
        ancestral_root_pool=ref(a.ancestral_root_pool),
        ancestral_selection=ref(a.ancestral_selection),
        protected_aliases=ref(a.protected_aliases),
        parent_helpers=h,
        search_helper=contract["search_helper"],
        root_pool_output=str(a.root_pool_output.resolve()),
        generation=parent["generation"],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("first", "deadline", "operator-end-epoch"):
        p.add_argument("--" + name, type=float, required=True)
    p.add_argument("--cpu-core", type=int, choices=(1, 3), required=True)
    for name in (
        "admission-seal",
        "admission-result",
        "inference-directory",
        "extension",
        "ancestral-root-pool",
        "ancestral-selection",
        "protected-aliases",
        "root-pool-output",
        "output",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    with a.output.open("xb") as f:
        f.write(canonical(build(a)) + b"\n")

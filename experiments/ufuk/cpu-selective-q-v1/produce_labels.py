"""ROOT executes actual frozen-prior leaf counterfactuals; template refuses execution."""

import argparse
import gzip
import json
from pathlib import Path

from labels import collect
from search import BASE_SHA
from support import load, publish, registration, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--registration", type=Path, required=True)
    a = p.parse_args()
    r, out, guard = registration(a.registration)
    try:
        with gzip.open(r["inputs"]["ownQ_labels"]["path"], "rb") as stream:
            raw = stream.read(16 * 2**20 + 1)
        if len(raw) > 16 * 2**20:
            raise ValueError("bounded ownQ input")
        data = json.loads(raw)
        if (
            data["schema"] != "own-search-deeper-value-labels-v1"
            or data["seed"] != r["seed"]
            or data["source_commit"] != r["source_commit"]
        ):
            raise ValueError("actual ownQ TRAIN source")
        for key in ["journal_sha256", "config_sha256", "training_dataset_sha256"]:
            if data[key] != r["parent_data_sha256"][key]:
                raise ValueError("original ownTRAIN lineage")
        rows = data["roots"]
        if r["selection"] != {
            "roots": 256,
            "order": "lexicographic-row_id",
            "leaves": 4,
            "leaf_order": "chronological-q2-horizon",
            "base_nodes": 512,
            "counterfactual_nodes": 64,
            "counterfactual_extra_plies": 2,
            "label_absolute_delta_strict_gt": 0.10,
        }:
            raise ValueError("fixed label design")
        base = load(r["base_search"]["path"], BASE_SHA, "selective_q_pinned_base")
        value = load(
            r["prior_value"]["path"], r["prior_value"]["sha256"], "selective_q_pinned_prior"
        )
        prior = value.load_classical(Path(r["inputs"]["prior"]["path"]))
        if data["frozen_prior_model"] != json.loads(Path(r["inputs"]["prior"]["path"]).read_text()):
            raise ValueError("same own-search prior")
        if prior.theta != (0.0,) * 18:
            raise ValueError("humanprior-only frozen labels")
        result = collect(rows, base.BudgetSearch, base.BudgetExhausted, prior.nonterminal, guard)
        result.update(
            seed=r["seed"],
            registration_sha256=sha(a.registration),
            inputs=r["inputs"],
            source_commit=r["source_commit"],
        )
        groups = sorted({x["trajectory_id"] for x in result["roots"]})
        import hashlib

        validation = [x for x in groups if int(hashlib.sha256(x.encode()).hexdigest(), 16) % 5 == 0]
        split = dict(
            schema="own-selective-q-trajectory-split-v1",
            train=[x for x in groups if x not in validation],
            validation=validation,
            rule="trajectory-sha256-mod5-zero-validation",
        )
        guard()
        publish(out / "split.json", split)
        publish(out / "dataset.json", result)
        guard()
        publish(
            out / "result.json",
            dict(
                status="PASS-own-counterfactual-labels-not-admission-or-strength",
                dataset_sha256=sha(out / "dataset.json"),
                finished=__import__("time").time(),
                deadline=r["deadline_epoch"],
            ),
        )
    except Exception as e:
        publish(out / "failure.json", dict(status="INCOMPLETE", error=repr(e)))
        raise


if __name__ == "__main__":
    main()

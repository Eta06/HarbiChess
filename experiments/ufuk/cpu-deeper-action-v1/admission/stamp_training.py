"""ROOT future protocol seal: exact closed labels and original producer, no fits."""

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).parent
END = 1791273600


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["proof", "fit"], required=True)
    p.add_argument("--first", type=float, required=True)
    p.add_argument("--draft-sha256", required=True)
    p.add_argument("--labels05-sha256", required=True)
    p.add_argument("--labels06-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    draft = ROOT / f"{a.phase}-protocol-DRAFT.json"
    if sha(draft) != a.draft_sha256:
        raise ValueError("exact draft SHA")
    q = json.loads(draft.read_bytes())
    seconds = 600 if a.phase == "proof" else 1800
    if not a.first <= time.time() < a.first + seconds <= END:
        raise ValueError("new actual phase clock")
    q["status"] = "ROOT-approved-fixed-ACTION-" + a.phase
    for seed, expected in [(20262905, a.labels05_sha256), (20262906, a.labels06_sha256)]:
        pair = q["inputs"][str(seed)]
        path = Path(pair["labels_path"])
        if sha(path) != expected or path.stat().st_size > 16 * 1024**2:
            raise ValueError("sealed complete action label SHA")
        labels = json.loads(gzip.decompress(path.read_bytes()))
        if (
            labels["schema"] != "own-search-deeper-bestmove-labels-v1"
            or len(labels["roots"]) != 1024
            or labels["seed"] != seed
            or labels["registration_sha256"] != pair["producer_registration_sha256"]
            or sha(pair["producer_registration_path"]) != pair["producer_registration_sha256"]
        ):
            raise ValueError("exact original producer closure/data")
        pair["labels_sha256"] = expected
        q["first_by_seed"][str(seed)] = a.first
        q["deadline_by_seed"][str(seed)] = a.first + seconds
    with a.output.open("x") as stream:
        json.dump(q, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "phase": a.phase,
                "protocol": str(a.output),
                "sha256": sha(a.output),
                "first": a.first,
                "deadline": a.first + seconds,
                "jobs_launched": False,
            }
        )
    )


if __name__ == "__main__":
    main()

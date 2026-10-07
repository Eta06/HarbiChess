"""ROOT-only metadata conversion entry point; never searches or trains."""

import argparse
import hashlib
import json
from pathlib import Path

from convert import canonical, convert_sealed


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path, data, limit):
    path = Path(path)
    if not path.resolve().is_relative_to("/dev/shm"):
        raise ValueError("converted artifacts are RAM-only")
    payload = canonical(data) + b"\n"
    if len(payload) > limit:
        raise ValueError("converted artifact size bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    seal_path = args.seal.resolve()
    if not seal_path.is_file() or seal_path.is_symlink() or seal_path.stat().st_size > 2**20:
        raise ValueError("small immutable conversion seal required")
    seal = json.loads(seal_path.read_bytes())
    dataset, provenance = convert_sealed(seal)
    out = args.output.resolve()
    if not out.is_relative_to("/dev/shm"):
        raise ValueError("output directory must be in RAM")
    out.mkdir(parents=True, exist_ok=False)
    dataset_path, provenance_path = out / "dataset.json", out / "provenance.json"
    write_once(dataset_path, dataset, 8 * 2**20)
    write_once(provenance_path, provenance, 2 * 2**20)
    receipt = {
        "schema": "human-prior-tdleaf-conversion-output-v1",
        "status": "PASS-replayed-closed-PV-leaf-targets",
        "seed": provenance["seed"],
        "conversion_seal": {"path": str(seal_path), "sha256": sha(seal_path)},
        "dataset": {"path": str(dataset_path), "sha256": sha(dataset_path)},
        "provenance": {"path": str(provenance_path), "sha256": sha(provenance_path)},
        "converter_path": str(Path(__file__).with_name("convert.py").resolve()),
        "converter_sha256": sha(Path(__file__).with_name("convert.py")),
        "teacher_labels_used": False,
        "rows": dataset["gradient_rows"],
    }
    write_once(out / "conversion-result.json", receipt, 64 * 2**10)


if __name__ == "__main__":
    main()

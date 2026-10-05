"""Supply the omitted seed export; preserve frozen controller bytes and clocks."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--original-controller", type=Path, required=True)
    p.add_argument("--original-controller-sha256", required=True)
    a, remaining = p.parse_known_args()
    assert remaining and remaining[0] == "--"
    remaining = remaining[1:]
    original = a.original_controller.resolve()
    assert original.name in ("ownv1_baseline.py", "own5_baseline.py")
    assert sha(original) == a.original_controller_sha256
    qpath = Path(remaining[remaining.index("--qualification-config") + 1])
    qsha = remaining[remaining.index("--qualification-config-sha256") + 1]
    assert sha(qpath) == qsha
    q = json.loads(qpath.read_text())
    assert q["helper_sha256"][original.name] == sha(original)
    for name, expected in q["helper_sha256"].items():
        assert Path(name).name == name and sha(original.parent / name) == expected
    sys.path.insert(0, str(original.parent))
    spec = importlib.util.spec_from_file_location("frozen_baseline_seed_export_repair", original)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.argv = [str(original), *remaining]
    module.bind_cli(module.__dict__)
    assert not hasattr(module.c, "SEEDS")
    assert list(module.SEEDS) == q["seeds"]
    module.c.SEEDS = module.SEEDS
    module.main()


if __name__ == "__main__":
    main()

"""ROOT-owned single generation: no candidate selection, no previous actor resume."""

import argparse
from pathlib import Path

from control import Owner, read, ref
from learn import run_chain

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["config", "plan", "parent-seal", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    for name in ["config-sha256", "plan-sha256", "parent-seal-sha256"]:
        p.add_argument("--" + name, required=True)
    p.add_argument("--seed", type=int, choices=[20262905, 20262906], required=True)
    p.add_argument("--generation", type=int, choices=[1, 2, 3], required=True)
    a = p.parse_args()
    owner = Owner(read(dict(path=str(a.config), sha256=a.config_sha256)), a.output)
    parent = read(dict(path=str(a.parent_seal), sha256=a.parent_seal_sha256))
    if parent["seed"] != a.seed or parent["generation"] != a.generation:
        raise ValueError("exact same-seed sequential incoming generation")
    owner.generation(a.seed, a.generation, dict(path=str(a.plan), sha256=a.plan_sha256), parent)
    collection = read(ref(owner.root / f"collection-{a.seed}-g{a.generation}.json"))
    run_chain(owner, collection)

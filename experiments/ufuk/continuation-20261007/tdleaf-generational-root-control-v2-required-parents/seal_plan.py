"""Publish fixed six-bank plan once; ROOT metadata only, no outcome selection."""

import argparse
import importlib
import sys
import time
from pathlib import Path

from control import END, SEEDS, pinned, publish, read, ref, setup


def seal(config, bank_sets, output):
    h, _, _ = setup(config)
    sys.path.insert(0, str(h))
    plan_module = importlib.import_module("generation_plan")
    banks = {}
    for seed in SEEDS:
        slots = read(bank_sets[str(seed)])
        if set(slots) != {"1", "2", "3"}:
            raise ValueError("all three actual banks")
        for g in (1, 2, 3):
            b = read(slots[str(g)])
            if (
                b["status"] != "PASS-rule-only-procedural-root-bank"
                or b["accepted_count"] != 4096
                or b["seed"] != seed + 1000003 * g
                or b["protected_aliases"] != config["protected_aliases"]
            ):
                raise ValueError("registered bank seed/protection/count")
            pinned(b["registration"])
            pinned(b["selection"])
            pinned(b["attempt_trace"])
        banks[str(seed)] = slots
    plan = dict(
        schema=plan_module.SCHEMA,
        status="registered-before-generation1",
        recipe=plan_module.RECIPE,
        seeds=list(SEEDS),
        first=time.time(),
        operator_end_epoch=END,
        protected_aliases=config["protected_aliases"],
        banks=banks,
        bank_seeds={str(s): {str(g): s + 1000003 * g for g in (1, 2, 3)} for s in SEEDS},
        strength_scope=config["strength_scope"],
        control_config=ref(config["config_path"]),
    )
    publish(output, plan)
    plan_module.validate_plan(ref(output), SEEDS[0])
    return plan


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--config-sha256", required=True)
    p.add_argument("--banks05", type=Path, required=True)
    p.add_argument("--banks05-sha256", required=True)
    p.add_argument("--banks06", type=Path, required=True)
    p.add_argument("--banks06-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    config = read(dict(path=str(a.config), sha256=a.config_sha256))
    config["config_path"] = str(a.config)
    seal(
        config,
        {
            str(SEEDS[0]): dict(path=str(a.banks05), sha256=a.banks05_sha256),
            str(SEEDS[1]): dict(path=str(a.banks06), sha256=a.banks06_sha256),
        },
        a.output,
    )

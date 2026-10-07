"""Fixed prospective three-generation schedule; no model/search/clock sampling."""

import json

from parent_bridge import pinned, read

SCHEMA = "procedural-TDLeaf-three-generation-plan-v1"
END = 1791448916.685839
SEEDS = (20262905, 20262906)
RECIPE = dict(
    generations=3,
    eligible_rows=2048,
    root_limit=512,
    plies_per_root=16,
    updates=128,
    batch=256,
    search_nodes=8192,
    qdepth=2,
    max_depth=8,
    endpoint_generation=3,
    transfer="named-parent-weights-only-fresh-Adam-global-and-private-RNG",
    rng_seed="seed+1000003*generation",
    prior_teacher_labels=False,
)


def validate_plan(reference, seed):
    plan = read(reference)
    if (
        plan.get("schema") != SCHEMA
        or plan.get("status") != "registered-before-generation1"
        or plan.get("recipe") != RECIPE
        or plan.get("seeds") != list(SEEDS)
        or seed not in SEEDS
        or not 0 < plan["first"] < plan["operator_end_epoch"] <= END
        or set(plan["banks"]) != {str(s) for s in SEEDS}
    ):
        raise ValueError("fixed prospective plan/endpoint/seeds")
    seen = set()
    for s in SEEDS:
        if set(plan["banks"][str(s)]) != {"1", "2", "3"}:
            raise ValueError("all six precommitted banks required")
        for g in (1, 2, 3):
            r = plan["banks"][str(s)][str(g)]
            bank = read(r)
            if (
                r["sha256"] in seen
                or bank["seed"] != plan["bank_seeds"][str(s)][str(g)]
                or bank["protected_aliases"] != plan["protected_aliases"]
            ):
                raise ValueError("distinct seed-specific bank and same protection")
            seen.add(r["sha256"])
    if len({v for slots in plan["bank_seeds"].values() for v in slots.values()}) != 6:
        raise ValueError("six distinct outcome-blind bank seeds")
    return plan


def previous_start_aliases(plan, seed, generation, protected):
    """Source-order skip only, never outcome/quality selection or reroll."""
    from collector import starts

    seen = set()
    for old_generation in range(1, generation):
        old_bank = read(plan["banks"][str(seed)][str(old_generation)])
        old_selection = json.loads(pinned(old_bank["selection"]).read_bytes())
        pool = dict(
            schema="generational-procedural-train-roots-v1",
            selection_status="pass",
            train_only=True,
            rows=old_selection["rows"],
            previous_start_aliases=sorted(seen),
        )
        previous, _ = starts(pool, seed, protected)
        seen.update(r["final_alias"] for r in previous)
    return sorted(seen)


def validate_bank_slot(plan, seed, generation, bank, selected):
    from collector import starts
    from root_bank import protection

    if not 1 <= generation <= 3 or bank["protected_aliases"] != plan["protected_aliases"]:
        raise ValueError("fixed generation/protection")
    protected = protection(plan["protected_aliases"])
    excluded = previous_start_aliases(plan, seed, generation, protected)
    pool = dict(
        schema="generational-procedural-train-roots-v1",
        selection_status="pass",
        train_only=True,
        rows=selected["rows"],
        previous_start_aliases=excluded,
    )
    chosen, _ = starts(pool, seed, protected)
    return chosen, excluded


def require_generation(reg):
    plan = validate_plan(reg["generation_plan"], reg["seed"])
    if (
        not 1 <= reg["generation"] <= 3
        or reg["procedural_bank_receipt"] != plan["banks"][str(reg["seed"])][str(reg["generation"])]
        or reg["protected_aliases"] != plan["protected_aliases"]
        or reg["operator_end_epoch"] != plan["operator_end_epoch"]
        or reg["original_first_epoch"] < plan["first"]
    ):
        raise ValueError("exact registered generation bank/clock/protection")
    if "root_pool" in reg:
        from root_bank import protection

        bank = read(reg["procedural_bank_receipt"])
        selected = json.loads(pinned(bank["selection"]).read_bytes())
        pool = read(reg["root_pool"])
        expected = previous_start_aliases(
            plan, reg["seed"], reg["generation"], protection(reg["protected_aliases"])
        )
        if pool.get("previous_start_aliases") != expected or pool["rows"] != selected["rows"]:
            raise ValueError("complete bank and exact source-only skip schedule")
    return plan


def final_candidate(record):
    contract = read(record["contract"])
    if (
        contract["generation"] != 3
        or contract["updates"] != 128
        or contract["phase"] != "procedural-generational-tdleaf-own-learning-v2"
    ):
        raise ValueError("only CURRENT generation3 is selectable")
    pinned(record["candidate"])
    pinned(record["native"])
    return contract

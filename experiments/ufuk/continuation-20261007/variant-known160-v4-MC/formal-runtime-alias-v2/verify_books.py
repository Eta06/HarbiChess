"""ROOT independent deterministic replay of sealed draw, never a replacement draw."""

import argparse
import json
import time
from pathlib import Path

from alias_pool import own_aliases
from bindings import SEEDS, read, sha, validate_pair
from exposure import alias, build
from select_books import candidates, publish


def verify(spec, selection, directory, guard=lambda: None):
    eligibility = read(spec["eligibility"])
    if (selection["eligibility_sha256"] != spec["eligibility"]["sha256"]
            or selection["selection_observed_epoch"] < eligibility["finished_epoch"]
            or selection["outcomes_observed"] is not False
            or selection["root_result_rating_filter"] is not False
            or selection["excluded_last_record"] is not True
            or selection["specification_sha256"] != sha(spec["registered_specification_path"])):
        raise ValueError("same single outcome-blind registered draw")
    known = read(spec["known_protocol"])
    excluded, sources, counts = build(known, read(spec["exposure_manifest"]), guard)
    path = Path(spec["human_pgn"]["path"])
    if sha(path) != spec["human_pgn"]["sha256"]:
        raise ValueError("source pool changed")
    groups, _ = candidates(path.read_text(), excluded, guard)
    groups, owners = own_aliases(groups)
    if (selection["alias_ownership"] != "lexicographic-ECO-before-random-draw-v2"
            or selection["alias_owner_count"] != len(owners)):
        raise ValueError("same deterministic PREdraw alias-stratum population")
    books = {seed: read(ref) for seed, ref in selection["book_refs"].items()}
    validate_pair(books)
    if len(selection["draws"]) != 96 or len(set(selection["selected_strata"])) != 96:
        raise ValueError("exact96 oneper distinct preselected strata")
    for number, row in enumerate(selection["draws"]):
        guard()
        seed, index = SEEDS[number // 48], number % 48
        eco = selection["selected_strata"][number]
        pool = groups[eco]
        draw_index = row["selected_index"]
        if (row["seed"] != seed or row["eco"] != eco
                or type(draw_index) is not int or not 0 <= draw_index < len(pool)
                or row["population_size"] != len(pool)
                or row["probability"] != 1 / len(pool)
                or books[str(seed)]["splits"]["arena"][index] != pool[draw_index]):
            raise ValueError("exact uniform source draw/fullhistory/root/probability replay")
        import chess

        board = chess.Board(pool[draw_index]["root_fen"])
        if alias(board) in excluded:
            raise ValueError("selected current root exposure overlap")
    exclusion = read(dict(path=str(directory / "exclusion-pass.json"),
                          sha256=sha(directory / "exclusion-pass.json")))
    inventory = read(exclusion["exposure_inventory"])
    if (inventory["sources"] != sources or inventory["counts"] != counts
            or exclusion["books"] != selection["book_refs"]):
        raise ValueError("full current-DAG exposure reconciliation")
    return dict(schema="ONE-independent-blind-book-draw-verification-v2",
                status="PASS96-fullhistory-draw-source-probability-and-current-DAG-exclusion",
                roots=96, books=selection["book_refs"], new_random_draws=0,
                eligibility_sha256=spec["eligibility"]["sha256"], source_inventory=sources,
                caveat=("randomness assumptions conditional on fixed pool/selected strata, "
                        "not ECO magic"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("spec", "selection", "clock", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    spec = json.loads(a.spec.read_bytes())
    spec["registered_specification_path"] = str(a.spec)
    selection = json.loads(a.selection.read_bytes())
    clock = json.loads(a.clock.read_bytes())
    if (clock["selection_sha256"] != sha(a.selection)
            or not clock["first"] <= time.time() < clock["deadline"]
            <= min(clock["first"] + 600, spec["operator_end_epoch"])):
        raise ValueError("ROOT prospective original bookverify600")

    import os
    import shutil
    import sys

    known = read(spec["known_protocol"])
    os.sched_setaffinity(0, {clock["cpu_core"]})
    sys.path.insert(0, known["core_repo"] + "/src")
    from harbichess.training.cgroup_budget import CgroupMemoryBudget

    memory = CgroupMemoryBudget(15 * 2**30)

    def guard():
        memory.check()
        if shutil.disk_usage("/workspace").free < 256 * 2**20:
            raise RuntimeError("workspace256MiB floor")
        if time.time() >= clock["deadline"]:
            raise TimeoutError("original verifier600")

    result = verify(spec, selection, a.selection.parent, guard)
    guard()
    result.update(first=clock["first"], deadline=clock["deadline"], finished=time.time(),
                  selection_sha256=sha(a.selection), helper_sha256=sha(__file__))
    publish(a.output, result)


if __name__ == "__main__":
    main()

"""ROOT-only single outcome-blind finite-pool stratified source draw AFTER eligibility."""

import argparse
import hashlib
import io
import itertools
import json
import os
import re
import secrets
import shutil
import struct
import sys
import time
from pathlib import Path

import chess
import chess.pgn
from alias_pool import own_aliases
from bindings import SEEDS, read, sha, validate_pair
from exposure import alias, build


def publish(path, packet):
    with Path(path).open("x") as f:
        json.dump(packet, f, sort_keys=True, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())


def candidates(text, excluded, guard=lambda: None):
    """Fixed16ply root; ignore Result/ratings/evaluation/comments. Discard final PGN."""
    offsets = [m.start() for m in re.finditer(r"(?m)^\[Event ", text)]
    groups, ids, counts = {}, set(), dict(records=0, rejected=0, excluded=0)
    # The partial compressed prefix's LAST record is never admitted, even if parsed.
    for start, end in itertools.pairwise(offsets):
        guard()
        counts["records"] += 1
        block = text[start:end]
        game = chess.pgn.read_game(io.StringIO(block))
        if (game is None or game.errors or game.headers.get("SetUp", "0") != "0"
                or "FEN" in game.headers
                or game.headers.get("Variant", "Standard") != "Standard"):
            counts["rejected"] += 1
            continue
        eco, source = game.headers.get("ECO", ""), game.headers.get("Site", "")
        if not re.fullmatch(r"[A-E][0-9]{2}", eco) or not source.startswith("https://lichess.org/"):
            counts["rejected"] += 1
            continue
        if source in ids:
            raise ValueError("duplicated human source game in fixed sampling pool")
        ids.add(source)
        board, prefix = chess.Board(), []
        for move in game.mainline_moves():
            if len(prefix) == 16:
                break
            if move not in board.legal_moves:
                raise ValueError("illegal complete source prefix")
            board.push(move)
            prefix.append(move.uci())
        if len(prefix) != 16 or not board.is_valid() or board.outcome(claim_draw=True):
            counts["rejected"] += 1
            continue
        if alias(board) in excluded:
            counts["excluded"] += 1
            continue
        root_id = hashlib.sha256((source + "\n" + " ".join(prefix)).encode()).hexdigest()
        groups.setdefault(eco, []).append(dict(root_id=root_id, source_family_id=source, eco=eco,
                    root_fen=board.fen(), opening=dict(moves=prefix),
                    human_source_record_sha256=hashlib.sha256(block.encode()).hexdigest()))
    # Duplicate root states across different strata would undermine block uniqueness.
    state_counts = {}
    for rows in groups.values():
        for row in rows:
            state_counts[row["root_fen"]] = state_counts.get(row["root_fen"], 0) + 1
    groups = {eco: [r for r in rows if state_counts[r["root_fen"]] == 1]
              for eco, rows in groups.items()}
    groups = {eco: rows for eco, rows in groups.items() if rows}
    return groups, counts


def draw(groups, randbelow=secrets.randbelow):
    if len(groups) < 96:
        raise ValueError("INCOMPLETE fewer96 eligible distinct ECO strata; no refill/new source")
    strata = sorted(groups)
    selected = []
    for _ in range(96):
        index = randbelow(len(strata))
        if type(index) is not int or not 0 <= index < len(strata):
            raise ValueError("random index")
        selected.append(strata.pop(index))
    books, draws = {}, []
    for n, seed in enumerate(SEEDS):
        rows = []
        for eco in selected[n * 48:(n + 1) * 48]:
            pool = groups[eco]
            index = randbelow(len(pool))
            if type(index) is not int or not 0 <= index < len(pool):
                raise ValueError("independent source-game random index")
            rows.append(pool[index])
            draws.append(dict(seed=seed, eco=eco, selected_index=index, population_size=len(pool),
                              probability=1 / len(pool),
                              source_family_id=pool[index]["source_family_id"]))
        books[str(seed)] = dict(schema="ONE-independent-source-ECO-root48-book-v2",
                               scope="ancestry-conditional finite-pool stratified confirmation",
                               splits=dict(arena=rows))
    validate_pair(books)
    return books, draws, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    spec = json.loads(a.spec.read_bytes())
    eligibility = read(spec["eligibility"])
    if (spec["schema"] != "ONE-after-eligibility-blind-book-selection-registration-v2"
            or spec["status"] != "registered-no-book-draw-yet"
            or eligibility["status"] != "PASS-fixed-own64-bothseeds-eligible-no-formal-results"
            or eligibility["formal_campaigns_used"] != 0
            or not eligibility["finished_epoch"] <= spec["first"] <= time.time()
            < spec["deadline"] <= min(spec["first"] + 600, spec["operator_end_epoch"])):
        raise ValueError("ONE prospective registered selection after fixed eligible candidate")
    known = read(spec["known_protocol"])
    manifest = read(spec["exposure_manifest"])
    if eligibility["known_protocol"] != spec["known_protocol"]:
        raise ValueError("same selected immutable DAG")
    os.sched_setaffinity(0, {spec["cpu_core"]})
    sys.path.insert(0, known["core_repo"] + "/src")
    from harbichess.training.cgroup_budget import CgroupMemoryBudget

    memory = CgroupMemoryBudget(15 * 2**30)

    def guard():
        memory.check()
        if time.time() >= spec["deadline"]:
            raise TimeoutError("original bookstage600 includes ALL exclusion and parsing work")
        if shutil.disk_usage("/workspace").free < 256 * 2**20:
            raise RuntimeError("workspace floor256MiB")

    for name, digest in spec["helper_sha256"].items():
        if sha(Path(__file__).with_name(name)) != digest:
            raise ValueError("registered frozen selection/helper source")
    guard()
    excluded, sources, counts = build(known, manifest, guard)
    if any(sha(path) != digest for path, digest in sources.items()):
        raise ValueError("exclusion source changed before draw")
    source = spec["human_pgn"]
    if sha(source["path"]) != source["sha256"] or Path(source["path"]).stat().st_size > 32 * 2**20:
        raise ValueError("registered finite public humanPGN bytepool")
    binding, decoded = read(spec["public_source_binding"]), read(spec["decode_binding"])
    if (binding["sha256"] != spec["compressed_source"]["sha256"]
            or sha(spec["compressed_source"]["path"]) != binding["sha256"]
            or decoded["decoded_sha256"] != source["sha256"]
            or decoded["decoded_bytes"] != Path(source["path"]).stat().st_size
            or not binding["url"].startswith("https://database.lichess.org/standard/")):
        raise ValueError("exact public compressed-prefix/decode SHA provenance")
    groups, pool_counts = candidates(Path(source["path"]).read_text(), excluded, guard)
    groups, alias_owners = own_aliases(groups)
    # Reserve the publish-once stage BEFORE random draw; no repeated draw on existing output.
    a.output.mkdir(exist_ok=False)
    publish(a.output / "draw-stage-started.json", dict(
        specification_sha256=sha(a.spec), observed_epoch=time.time(),
        retries_permitted=False, selection_only_after_eligibility=True))
    # Genuine independent OS-random choices per stratum, no candidate result read/filter.
    books, draws, strata = draw(groups)
    guard()
    publish(a.output / "draw-once-receipt.json", dict(
        draws=draws, selected_strata=strata, specification_sha256=sha(a.spec),
        observed_epoch=time.time(), retries_permitted=False))
    if (any(sha(path) != digest for path, digest in sources.items())
            or sha(source["path"]) != source["sha256"]):
        raise ValueError("all exposure/public pool inputs immutable after single draw")
    refs = {}
    for seed, book in books.items():
        path = a.output / f"book-{seed}.json"
        publish(path, book)
        refs[seed] = dict(path=str(path), sha256=sha(path))
    binary = a.output / "current-DAG-exposure-aliases.bin"
    with binary.open("xb") as f:
        for value in sorted(excluded):
            f.write(struct.pack("<q", value))
        f.flush()
        os.fsync(f.fileno())
    inventory = dict(schema="ONE-current-selected-DAG-exposure-inventory-v2", sources=sources,
                     recorded_aliases=len(excluded), counts=counts,
                     alias_binary=dict(path=str(binary), sha256=sha(binary)),
                     manifest=spec["exposure_manifest"],
                     excluded_scope="heldout ROOT/state aliases, not every shared START ancestor",
                     unresolved=("unlogged SF-internal nodes/old6and8/E8 pretraining; "
                                 "no global coverage"))
    ip = a.output / "exposure-inventory.json"
    publish(ip, inventory)
    exclusion = dict(schema="ONE-ancestry-selected-DAG-book-exclusion-audit-v2",
                     status="PASS-zero-current-ancestry-and-known-book-overlap", candidate_roots=96,
                     matched_roots=0, books=refs, eligibility_sha256=spec["eligibility"]["sha256"],
                     current_DAG_coverage_complete=True, global_historical_coverage_claimed=False,
                     exposure_inventory=dict(path=str(ip), sha256=sha(ip)))
    publish(a.output / "exclusion-pass.json", exclusion)
    selection = dict(schema="ONE-after-eligible-outcome-blind-source-selection-v2",
                     eligibility_sha256=spec["eligibility"]["sha256"],
                     selection_observed_epoch=time.time(), book_refs=refs, outcomes_observed=False,
                     independence_design="independent-distinct-source-ECO-clusters",
                     sampling_population="fixed SHA public-prefix, legal16ply, nonexposed roots; "
                     "condition on selected96strata; each stratum one uniform "
                     "independent human game",
                     alias_ownership="lexicographic-ECO-before-random-draw-v2",
                     alias_owner_count=len(alias_owners),
                     draws=draws, selected_strata=strata, pool_counts=pool_counts,
                     stratum_sizes={eco: len(groups[eco]) for eco in strata},
                     OS_random_draws=True, pgn_source=source,
                     public_source=spec["public_source_binding"],
                     decode_binding=spec["decode_binding"], root_result_rating_filter=False,
                     excluded_last_record=True, specification_sha256=sha(a.spec),
                     first=spec["first"], deadline=spec["deadline"], finished=time.time())
    guard()
    publish(a.output / "book-selection.json", selection)


if __name__ == "__main__":
    main()

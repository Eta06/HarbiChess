"""Outcome-blind prospective ONE confirmation bindings; no model/search calls."""

import copy
import hashlib
import json
import re
from pathlib import Path

import chess

SEEDS = (20262905, 20262906)
TASKS = (("learned", "e8"), ("learned", "parent"), ("learned", "SF512"),
         ("parent", "SF512"), ("e8", "SF512"))
SEARCH_SHA = "de53c14728a67b7772f18b396ac8ef5c35a144d4e4e616fec40099cd461a6670"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(ref):
    path = Path(ref["path"])
    if sha(path) != ref["sha256"]:
        raise ValueError("immutable input SHA")
    return json.loads(path.read_bytes())


def validate_book(book):
    rows = book["splits"]["arena"]
    if len(rows) != 48:
        raise ValueError("exact48 paired source-root blocks")
    result = []
    for row in rows:
        if any(not isinstance(row.get(k), str) or not row[k]
               for k in ("root_id", "source_family_id", "eco")):
            raise ValueError("explicit root/source-family/ECO provenance")
        if not re.fullmatch(r"[A-E][0-9]{2}", row["eco"]):
            raise ValueError("actual standard ECO stratum code")
        moves = row["opening"]["moves"]
        board = chess.Board()
        for uci in moves:
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                raise ValueError("illegal complete opening history")
            board.push(move)
        if not board.is_valid() or board.outcome(claim_draw=True) or board.ply() >= 400:
            raise ValueError("nonterminal legal fullhistory root before400total cap")
        if row["root_fen"] != board.fen():
            raise ValueError("exact complete root FEN")
        result.append((row["root_id"], row["source_family_id"], row["eco"],
                       tuple(moves), board.fen()))
    for col in range(5):
        if len({x[col] for x in result}) != 48:
            raise ValueError("distinct root/source/ECO blocks within seed")
    return result


def validate_pair(books):
    if set(books) != set(map(str, SEEDS)):
        raise ValueError("exact both registered seed books")
    records = [x for seed in SEEDS for x in validate_book(books[str(seed)])]
    for col in range(5):
        if len({x[col] for x in records}) != 96:
            raise ValueError("both seed books must have disjoint root/source/ECO blocks")
    return records


def validate_supplement(ref, original_draft, approval, october_prereg, betting_sha):
    supplement = read(ref)
    # ROOT's prospective registered supplement must expose exact these bindings.
    if (supplement["schema"] != "ONE-confirmation-additional-fixed-n-betting-preregistration-v1"
            or supplement["status"] !=
            "REGISTERED-prospective-additional-requirement-before-any-new-strength-outcome"
            or supplement["fraction_grid"] != [f"{i}/10" for i in range(1, 10)] + ["1"]
            or supplement["data_driven_bets_allowed"] is not False
            or supplement["exact_new_union_alpha"] != "1/160"
            or supplement["original_confirmation_DRAFT"] != original_draft
            or supplement["user_v2_approval"] != approval
            or supplement["current_experiment_preregistration"] != october_prereg
            or supplement["original_confirmation_registration_sha256"] is not None
            or supplement["new_code_sha256"] != betting_sha
            or supplement["root_blocks_fixed_n"] != 48
            or supplement["new_required_bounds"] != 8
            or supplement["exact_alpha_each"] != "1/1280"
            or supplement["optional_stopping_allowed"] is not False
            or supplement["original_bounds_unchanged_required"] is not True):
        raise ValueError("exact prospective additive supplement/draft/USER scope binding")
    for binding in (original_draft, approval, october_prereg):
        if sha(binding["path"]) != binding["sha256"]:
            raise ValueError("original draft/approval/Markdown preregistration SHA")
    return supplement


def validate_value_wiring(known):
    # Exact fields traversed by NNUE MixedValue, original MixedValue and NeuralValue.
    for key in ("original_mixed_value", "E8_value_helper", "prior_helper", "binary"):
        ref = known.get(key)
        if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
                or not isinstance(ref["path"], str)
                or not isinstance(ref["sha256"], str) or len(ref["sha256"]) != 64):
            raise ValueError("complete explicit value adapter dependency: " + key)
    if (not known.get("nnue_directory")
            or set(known.get("nnue_helpers", {})) != {"model.py", "native.py", "evaluator.py"}):
        raise ValueError("complete same-directory NNUE helpers")
    for seed in SEEDS:
        roles = known["models"][str(seed)]
        if (set(roles) != {"learned", "parent", "e8"}
                or not all(Path(roles[r]["path"]).suffix == ".pt" for r in ("learned", "parent"))
                or Path(roles["e8"]["path"]).suffix != ".safetensors"):
            raise ValueError("no JSON/zero/prior endpoint substituted for own/teacher/E8")
        if (roles["parent"] != known["teachers"][str(seed)]["candidate"]
                or roles["learned"] != known["children"][str(seed)]["candidate"]
                or roles["e8"]["sha256"] !=
                "e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03"):
            raise ValueError("role references must exactly equal actual native admitted endpoints")


def protocol_views(known, book_refs):
    validate_value_wiring(known)
    if (known["match_seeds"] != list(SEEDS)
            or {tuple(x) for x in known["tasks"]} != set(TASKS)
            or (known["search_nodes"], known["quiescence_plies"], known["max_depth"],
                known["max_plies"], known["stockfish_nodes"])
            != (512, 2, 8, 400, 512)
            or known["original_search"]["sha256"] != SEARCH_SHA):
        raise ValueError("same frozen recipe/search/5arms/settings")
    validate_pair({s: read(ref) for s, ref in book_refs.items()})
    result = {}
    for seed in SEEDS:
        q = copy.deepcopy(known)
        q.update(schema="ONE-ancestry-conditional-NNUE-confirmation-seed-view-v2",
                 book_path=book_refs[str(seed)]["path"],
                 book_sha256=book_refs[str(seed)]["sha256"], opening_pairs=48,
                 games_per_tournament=96, total_games=960,
                 scope="ONE approved ancestry-conditional confirmation; five arms, frozen root48")
        result[str(seed)] = q
    return result

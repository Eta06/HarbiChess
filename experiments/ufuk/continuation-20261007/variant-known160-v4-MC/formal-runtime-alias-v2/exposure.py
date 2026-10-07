"""Recorded current-candidate DAG state aliases only; no inference/search/teacher."""

import gzip
import hashlib
import json
import struct
from pathlib import Path

import chess
from bindings import SEEDS, read, sha


def alias(board):
    placement = min(board.board_fen(), board.mirror().board_fen())
    return int.from_bytes(hashlib.sha256(placement.encode()).digest()[:8], "little", signed=True)


def history(root_fen, moves, values, guard):
    board = chess.Board(root_fen)
    values.add(alias(board))
    for uci in moves:
        guard()
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError("illegal recorded DAG fullhistory")
        board.push(move)
        values.add(alias(board))
    return board


def packed(ref):
    path = Path(ref["path"])
    if sha(path) != ref["sha256"]:
        raise ValueError("sealed alias sidecar SHA")
    data = path.read_bytes()
    if len(data) % 8:
        raise ValueError("int64 alias binary")
    return set(struct.unpack(f"<{len(data)//8}q", data))


def build(known, manifest, guard=lambda: None):
    """Manifest names ALL raw ancestors/known selection records, never TRAIN subset."""
    if (manifest["schema"] != "ONE-current-selected-DAG-exposure-manifest-v2"
            or set(manifest["classical_parent_lineages"]) != set(map(str, SEEDS))
            or manifest["all_known_selection_inventory_complete"] is not True):
        raise ValueError("complete current DAG and all known-selection inventory required")
    inventory_ref = manifest["known_selection_inventory"]
    inventory = read(inventory_ref)
    if (inventory["status"] != "PASS-complete-recorded-known-selection-inventory"
            or inventory["known_books"] != manifest["known_books"]
            or inventory["known_arenas"] != manifest["known_arenas"]):
        raise ValueError("exact reconciled known inventory, not a bare completeness flag")
    values, sources, counts = set(), {inventory_ref["path"]: inventory_ref["sha256"]}, {}
    for seed in SEEDS:
        lineage_ref = manifest["classical_parent_lineages"][str(seed)]
        lineage = read(lineage_ref)
        teacher_contract = read(known["teachers"][str(seed)]["contract"])
        provenance_ref = dict(path=teacher_contract["target_provenance_path"],
                              sha256=teacher_contract["target_provenance_sha256"])
        provenance = read(provenance_ref)
        if provenance["selection_receipt"]["original_lineage"] != lineage_ref:
            raise ValueError("CLASSIC ancestor must equal exact teacher-native data lineage")
        sources[provenance_ref["path"]] = provenance_ref["sha256"]
        sources[lineage_ref["path"]] = lineage_ref["sha256"]
        cfg_ref = dict(path=lineage["config_path"], sha256=lineage["config_sha256"])
        config = read(cfg_ref)
        journal_ref = dict(path=lineage["journal_path"], sha256=lineage["journal_sha256"])
        if sha(journal_ref["path"]) != journal_ref["sha256"]:
            raise ValueError("full original16384 journal SHA")
        envelope = json.loads(gzip.decompress(Path(journal_ref["path"]).read_bytes()))
        state = envelope["state"]
        canonical = json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if (hashlib.sha256(canonical.encode()).hexdigest() != envelope["state_sha256"]
                or state["schema"] != "classical-own-qsearch-selfplay-journal-v3"
                or state["config"] != config or config["seed"] != seed
                or state["actions"] != 16384):
            raise ValueError("whole raw original CLASSIC ancestor, including UNKNOWN/tails")
        games = [*state["games"], *([state["active"]] if state["active"] else [])]
        for game in games:
            root = config["roots"][game["root_index"]]
            history(root["root_fen"], [*root["prefix"], *[r["action"] for r in game["moves"]]],
                    values, guard)
        sources.update({journal_ref["path"]: journal_ref["sha256"],
                        cfg_ref["path"]: cfg_ref["sha256"]})
        info = known["children"][str(seed)]
        receipt = read(info["collection_receipt"])
        registration = read(info["collection_registration"])
        mc = known['target_variant'] == 'closed-terminal-mc-v1'
        expected_status = ('PASS-exact-closed-terminal-row-budget' if mc
                           else 'PASS-exact-row-budget')
        expected_schema = ('own-nnue-closed-terminal-collection-receipt-v1' if mc
                           else 'own-nnue-ownq-collection-receipt-v2')
        if (receipt['schema'] != expected_schema
                or receipt["registration_sha256"] != info["collection_registration"]["sha256"]
                or receipt["events_sha256"] != info["events"]["sha256"]
                or receipt["status"] != expected_status):
            raise ValueError("complete actual own parent-search trace")
        for ref in (info["collection_receipt"], info["collection_registration"], info["events"]):
            sources[ref["path"]] = ref["sha256"]
        if sha(info["events"]["path"]) != info["events"]["sha256"]:
            raise ValueError("events hash")
        for line in Path(info["events"]["path"]).read_bytes().splitlines():
            guard()
            event = json.loads(line)
            if event["type"] == "game_start":
                history(event["root_fen"], event["root_prefix_uci"], values, guard)
            elif event["type"] == "search_row":
                row = event["row"]
                board = history(row["root_fen"], row["history_uci"], values, guard)
                if row["selected_action_played"]:
                    history(row["root_fen"], [*row["history_uci"], row["selected_best_uci"]],
                            values, guard)
                values.add(alias(board))
            elif event["type"] == "game_end":
                values.add(event["final_state"]["alias"])
            else:
                raise ValueError("failed/unknown event cannot seal exposure")
        for chunk in receipt["alias_chunks"]:
            ref = dict(path=str(Path(info["events"]["path"]).parent / chunk["file"]),
                       sha256=chunk["sha256"])
            if Path(ref["path"]).stat().st_size != chunk["bytes"]:
                raise ValueError("all traced ownNN evaluator alias chunk bytes")
            values |= packed(ref)
            sources[ref["path"]] = ref["sha256"]
        values |= packed(registration["protected_aliases"])
        protected_ref = registration["protected_aliases"]
        sources[protected_ref["path"]] = protected_ref["sha256"]
        labels_ref = known["teachers"][str(seed)]["labels"]
        if sha(labels_ref["path"]) != labels_ref["sha256"]:
            raise ValueError("exact teacher source root labels")
        labels = json.loads(gzip.decompress(Path(labels_ref["path"]).read_bytes()))
        if len(labels["rows"]) != 4096 or labels["seed"] != seed:
            raise ValueError("all4096 teacher parent source roots")
        for row in labels["rows"]:
            history(row["root_fen"], row["prefix_uci"], values, guard)
        sources[labels_ref["path"]] = labels_ref["sha256"]
        counts[str(seed)] = dict(classical_actions=16384, classical_games_including_tail=len(games),
                                 teacher_source_roots=4096, own_rows=receipt["all_actor_rows"],
                                 own_alias_chunks=len(receipt["alias_chunks"]))
    # All historical known-selection books and played records must be explicitly inventoried.
    for ref in manifest["known_books"]:
        book = read(ref)
        sources[ref["path"]] = ref["sha256"]
        for row in book["splits"]["arena"]:
            history(chess.STARTING_FEN, row["opening"]["moves"], values, guard)
    for ref in manifest["known_arenas"]:
        packet = read(ref)
        sources[ref["path"]] = ref["sha256"]
        for game in packet["games"]:
            history(game.get("root_fen", chess.STARTING_FEN), game["moves"], values, guard)
    for ref in manifest.get("additional_recorded_alias_bins", []):
        values |= packed(ref)
        sources[ref["path"]] = ref["sha256"]
    for ref in manifest.get("additional_fullhistory_rows", []):
        packet = read(ref)
        sources[ref["path"]] = ref["sha256"]
        for row in packet["rows"]:
            history(row["root_fen"], row["prefix_uci"], values, guard)
    if any(sha(path) != digest for path, digest in sources.items()):
        raise ValueError("immutable recorded DAG changed")
    return values, sources, counts

"""New frozen-snapshot CPU actor; compact journals, not old online-native resumes."""

import gzip
import hashlib
import importlib.util
import json
import math
import os
import random
from pathlib import Path

import chess

SCHEMA = "fresh-qsearch-selfplay-journal-v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(path, expected_sha):
    if sha(path) != expected_sha:
        raise ValueError("helper SHA mismatch")
    import sys

    name = "qsearch_" + expected_sha
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tuple_tree(value):
    return tuple(tuple_tree(x) for x in value) if isinstance(value, list) else value


def board_for(root):
    board = chess.Board(root["root_fen"])
    if not board.is_valid():
        raise ValueError("invalid root")
    for text in root["prefix"]:
        move = chess.Move.from_uci(text)
        if move not in board.legal_moves:
            raise ValueError("illegal prefix")
        board.push(move)
    return board


def validate_config(config):
    if (
        config["nodes"],
        config["qdepth"],
        config["max_depth"],
        config["exploration"],
        config["total_ply_cap"],
    ) != (512, 2, 8, 0.05, 400):
        raise ValueError("prospective search/exploration settings differ")
    if not 1 <= config["max_actions"] <= 32768 or config["actors"] != 1:
        raise ValueError("one-actor action ceiling differs")
    if not math.isfinite(config["original_deadline_epoch"]) or not config["epoch_id"]:
        raise ValueError("registered epoch/deadline required")
    for key in (
        "model_sha256",
        "source_commit",
        "search_helper_sha256",
        "value_helper_sha256",
        "producer_sha256",
    ):
        value = config[key]
        if len(value) != (40 if key == "source_commit" else 64):
            raise ValueError("missing immutable identity")
        int(value, 16)
    if not config["roots"]:
        raise ValueError("empty roots")
    for root in config["roots"]:
        board = board_for(root)
        if not root["source_id"] or board.is_game_over(claim_draw=True) or board.ply() >= 400:
            raise ValueError("root already closed or over cap")


class Actor:
    """Evaluator must be frozen. Production callers verify model bytes each segment."""

    def __init__(self, config, search_factory, state=None):
        validate_config(config)
        self.config = config
        self.search_factory = search_factory
        self.rng = random.Random(config["seed"])
        self.state = state or {
            "schema": SCHEMA,
            "config": config,
            "config_sha256": digest(config),
            "actions": 0,
            "games": [],
            "active": None,
            "rng": self.rng.getstate(),
            "phase": "collecting",
        }
        if self.state["config"] != config or self.state["config_sha256"] != digest(config):
            raise ValueError("snapshot/config cannot change during epoch")
        if state is not None:
            replay(state, config)
        self.rng.setstate(tuple_tree(self.state["rng"]))

    def board(self):
        active = self.state["active"]
        board = board_for(self.config["roots"][active["root_index"]])
        for row in active["moves"]:
            move = chess.Move.from_uci(row["action"])
            if move not in board.legal_moves:
                raise ValueError("illegal active history")
            board.push(move)
        return board

    def advance(self, target_actions):
        if not self.state["actions"] <= target_actions <= self.config["max_actions"]:
            raise ValueError("action ceiling or backward cursor")
        while self.state["actions"] < target_actions:
            if self.state["active"] is None:
                self.state["active"] = {
                    "root_index": self.rng.randrange(len(self.config["roots"])),
                    "game_index": len(self.state["games"]),
                    "moves": [],
                }
            board = self.board()
            legal = sorted(board.legal_moves, key=lambda move: move.uci())
            result = self.search_factory().search(board.copy(stack=True))
            if result.move not in legal or result.root_actions != len(legal):
                raise ValueError("search root coverage/legal selected action differs")
            if not 0 <= result.evaluations <= result.nodes <= 512:
                raise ValueError("search budget differs")
            explored = self.rng.random() < 0.05
            action = legal[self.rng.randrange(len(legal))] if explored else result.move
            row = {
                "action": action.uci(),
                "selected": result.move.uci(),
                "explored": explored,
                "legal_count": len(legal),
                "mover": "white" if board.turn else "black",
                "pre_ply": board.ply(),
                "mu": 0.05 / len(legal) + (0.95 if action == result.move else 0),
                "nodes": result.nodes,
                "evaluations": result.evaluations,
                "depth": result.completed_depth,
                "root_actions": result.root_actions,
                "search_value": result.value,
            }
            self.state["active"]["moves"].append(row)
            board.push(action)
            self.state["actions"] += 1
            outcome = board.outcome(claim_draw=True)
            if outcome is not None or board.ply() >= 400:
                game = self.state["active"]
                game["result"] = board.result(claim_draw=True) if outcome else "UNKNOWN"
                game["termination"] = (
                    board.outcome(claim_draw=True).termination.name if outcome else "total-ply-cap"
                )
                self.state["games"].append(game)
                self.state["active"] = None
        self.state["rng"] = self.rng.getstate()
        self.state["phase"] = (
            "epoch-action-ceiling" if target_actions == self.config["max_actions"] else "paused"
        )
        return self.state


def save(path, state):
    """Publish-once deterministic gzip; checksum binds RNG and unfinished history."""
    path = Path(path)
    payload = {"state": state, "state_sha256": digest(state)}
    data = gzip.compress(canonical(payload), mtime=0)
    occupied = sum(p.stat().st_size for p in path.parent.iterdir() if p.is_file())
    if occupied + len(data) > 2 * 1024 * 1024:
        raise ValueError("journal exceeds 2MiB prospective artifact ceiling")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()
    return sha(path)


def read(path):
    payload = json.loads(gzip.decompress(Path(path).read_bytes()))
    if digest(payload["state"]) != payload["state_sha256"]:
        raise ValueError("journal checksum mismatch")
    return payload["state"]


def replay(state, expected_config):
    """Independent legal/outcome and actor-RNG replay. No neural recomputation claim."""
    validate_config(expected_config)
    if state["schema"] != SCHEMA or state["config"] != expected_config:
        raise ValueError("journal identity differs")
    if state["config_sha256"] != digest(expected_config):
        raise ValueError("config checksum differs")
    rng = random.Random(expected_config["seed"])
    count = 0
    packets = []
    games = state["games"] + ([state["active"]] if state["active"] is not None else [])
    for game_index, game in enumerate(games):
        root_index = rng.randrange(len(expected_config["roots"]))
        if game["root_index"] != root_index or game["game_index"] != game_index:
            raise ValueError("root/game RNG cursor differs")
        root = expected_config["roots"][root_index]
        board = board_for(root)
        positions = []
        for row in game["moves"]:
            if board.is_game_over(claim_draw=True) or board.ply() >= 400:
                raise ValueError("action after game closed")
            legal = sorted(board.legal_moves, key=lambda move: move.uci())
            selected = chess.Move.from_uci(row["selected"])
            if selected not in legal:
                raise ValueError("illegal search selection")
            explored = rng.random() < 0.05
            action = legal[rng.randrange(len(legal))] if explored else selected
            mover = "white" if board.turn else "black"
            mu = 0.05 / len(legal) + (0.95 if action == selected else 0)
            if (
                row["action"],
                row["explored"],
                row["mover"],
                row["pre_ply"],
                row["legal_count"],
                row["root_actions"],
                row["mu"],
            ) != (action.uci(), explored, mover, board.ply(), len(legal), len(legal), mu):
                raise ValueError("actor probability/mover/RNG differs")
            if (
                not 0 <= row["evaluations"] <= row["nodes"] <= 512
                or not 0 <= row["depth"] <= 8
                or not math.isfinite(row["search_value"])
            ):
                raise ValueError("invalid search counters")
            positions.append((board.copy(stack=True), mover))
            board.push(action)
            count += 1
        outcome = board.outcome(claim_draw=True)
        closed = game_index < len(state["games"])
        if closed:
            result = board.result(claim_draw=True) if outcome else "UNKNOWN"
            termination = outcome.termination.name if outcome else "total-ply-cap"
            if not outcome and board.ply() < 400:
                raise ValueError("premature UNKNOWN closure")
            if game["result"] != result or game["termination"] != termination:
                raise ValueError("terminal result differs")
            if result != "UNKNOWN":
                packets.append((root["source_id"], game_index, positions, result))
        elif outcome or board.ply() >= 400:
            raise ValueError("closed game left active")
    if count != state["actions"] or count > expected_config["max_actions"]:
        raise ValueError("action cursor differs")
    if canonical(rng.getstate()) != canonical(state["rng"]):
        raise ValueError("final RNG differs")
    return packets


def shrink840(state, expected_config, feature_function):
    """Game-uniform groups; row labels 0=win,1=draw,2=loss from mover POV."""
    import numpy as np

    rows, labels, groups, provenance = [], [], [], []
    for source, game_index, positions, result in replay(state, expected_config):
        start = len(rows)
        for board, mover in positions:
            features = feature_function(board)
            if features.shape != (840,) or not np.isfinite(features).all():
                raise ValueError("SHRINK feature shape/finite guard")
            rows.append(features)
            won = (result == "1-0" and mover == "white") or (result == "0-1" and mover == "black")
            labels.append(1 if result == "1/2-1/2" else 0 if won else 2)
        if positions:
            groups.append(list(range(start, len(rows))))
            provenance.append(
                {
                    "source_id": source,
                    "game_index": game_index,
                    "journal_state_sha256": digest(state),
                    "config_sha256": digest(expected_config),
                }
            )
    return (
        np.asarray(rows, dtype=np.float32).reshape(-1, 840),
        np.asarray(labels, dtype=np.int64),
        groups,
        provenance,
    )


def convert_verified(path, expected_journal_sha, config, feature_path, feature_sha):
    """Memory-only training handoff; immutable source identities stay attached."""
    if sha(path) != expected_journal_sha:
        raise ValueError("source journal SHA mismatch")
    features = load_module(feature_path, feature_sha)
    state = read(path)
    x, y, groups, provenance = shrink840(state, config, features.invariants)
    receipt = {
        "schema": "fresh-qsearch-to-shrink840-handoff-v1",
        "source_journal_sha256": expected_journal_sha,
        "source_config_sha256": digest(config),
        "feature_helper_sha256": feature_sha,
        "source_commit": config["source_commit"],
        "frozen_model_sha256": config["model_sha256"],
        "actions_replayed": state["actions"],
        "known_rows": len(y),
        "complete_game_groups": len(groups),
        "excluded_UNKNOWN_or_tail_rows": state["actions"] - len(y),
        "sampler": "uniform-complete-game-then-uniform-position",
        "sampling_bias": "conditional-on-completion;no-full-trajectory-correction",
        "old_online_native_resume_claimed": False,
    }
    return x, y, groups, provenance, receipt

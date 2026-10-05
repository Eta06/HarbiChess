"""Compact exact current-piece/metadata features from verified own complete games.

These are the existing sparse head's 832 piece/EP features plus 8 metadata values.
Every continuation is replayed legally, including discarded UNKNOWN games.
"""

import gzip
import hashlib
import json
from collections import defaultdict

import chess
import numpy as np
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def halfmove_table():
    # Match the NHWC float32 metadata reduction, including its strided sum order.
    inputs = torch.zeros((101, 8, 8, 104), dtype=torch.float32)
    inputs[:, :, :, 102] = torch.tensor([i / 100 for i in range(101)])[:, None, None]
    return inputs[:, :, :, 96:].mean((1, 2))[:, 6].numpy().copy()


HALFMOVE = halfmove_table()


def counts_metadata(board):
    turn = board.turn
    counts = [board.pieces_mask(p, c).bit_count() for c in (turn, not turn) for p in range(1, 7)]
    metadata = [
        float(turn == chess.WHITE),
        float(board.has_kingside_castling_rights(turn)),
        float(board.has_queenside_castling_rights(turn)),
        float(board.has_kingside_castling_rights(not turn)),
        float(board.has_queenside_castling_rights(not turn)),
        1 / 64 if board.ep_square is not None else 0,
        HALFMOVE[min(board.halfmove_clock, 100)],
        1.0 if board.is_repetition(3) else 0.5 if board.is_repetition(2) else 0.0,
    ]
    return np.array(counts + metadata, dtype=np.float32)


def invariants(board):
    """Exact existing sparse-value current NHWC piece/EP and metadata features."""
    result = np.zeros(840, dtype=np.float32)
    pieces = result[:768].reshape(64, 12)
    for color_index, color in enumerate((board.turn, not board.turn)):
        for piece in range(1, 7):
            for square in board.pieces(piece, color):
                oriented = square if board.turn == chess.WHITE else square ^ 56
                pieces[oriented, color_index * 6 + piece - 1] = 1
    if board.ep_square is not None:
        square = board.ep_square if board.turn == chess.WHITE else board.ep_square ^ 56
        result[768 + square] = 1
    result[832:] = counts_metadata(board)[12:]
    return result


def partition(source):
    # Original source game, not per-epoch game or oversampled alias, stays together.
    return int.from_bytes(hashlib.sha256(source.encode()).digest()[:8], "big") % 5


def prepare(paths, guard):
    features, labels, anchors, groups, sources, receipts = [], [], [], [], [], []
    for path in paths:
        guard()
        digest = sha(path)
        record = json.loads(gzip.decompress(path.read_bytes()))
        if record["schema"] != "torch-fresh-fullgame-ppo-v1":
            raise ValueError("own source learner schema differs")
        unhashed = {k: v for k, v in record.items() if k != "sample_chain_sha256"}
        canonical = json.dumps(
            unhashed, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        chain = hashlib.sha256(
            bytes.fromhex(record["previous_sample_chain_sha256"]) + canonical
        ).hexdigest()
        if chain != record["sample_chain_sha256"]:
            raise ValueError("old immutable journal chain checksum differs")
        collection = record["collection"]
        if collection["schema"] != "full-history-frozen-policy-epoch-v1":
            raise ValueError("this own-outcome dataset requires frozen raw-policy epoch v1")
        episodes = defaultdict(list)
        for row in collection["actions"]:
            t = row["transition"]
            episodes[t["source_id"], t["game_index"]].append(row)
        truncations = {
            (x["source_id"], x["game_index"])
            for x in collection["truncations"]
            if x["termination"] == "policy-epoch-truncation"
        }
        counts = dict(
            complete_games=0,
            normal_cap_games=0,
            epoch_truncated_games=0,
            excluded_actions=0,
            win_games=0,
            draw_games=0,
            loss_games=0,
        )
        known_rows = legal_plies = 0
        for identity, moves in episodes.items():
            guard()
            first_row = moves[0]
            first = first_row["transition"]
            pre = first["pre"]
            board = chess.Board(pre["root_fen"])
            if not board.is_valid():
                raise ValueError("invalid original root position")
            for uci in pre["moves"]:
                m = chess.Move.from_uci(uci)
                if not board.is_legal(m):
                    raise ValueError("invalid original opening history")
                board.push(m)
            start_turn = board.turn
            history = list(pre["moves"])
            rows, turns, base_probs = [], [], []
            for i, actor_row in enumerate(moves):
                t = actor_row["transition"]
                if i < len(moves) - 1 and (t["rollout_cutoff"] or t["terminal_result"] is not None):
                    raise ValueError("own episode continued after a closed boundary")
                if (
                    t["slot"] != first["slot"]
                    or t["pre"]["root_fen"] != pre["root_fen"]
                    or t["pre"]["moves"] != history
                    or board.outcome(claim_draw=True) is not None
                ):
                    raise ValueError("own game discontinuity or action after terminal")
                rows.append(invariants(board))
                turns.append(board.turn)
                probs = np.asarray(actor_row["base_wdl"], dtype=np.float32)
                if (
                    probs.shape != (3,)
                    or not np.isfinite(probs).all()
                    or np.any(probs < 0)
                    or not np.isclose(float(probs.sum()), 1.0, atol=2e-6)
                ):
                    raise ValueError("stored fixed-base WDL anchor is invalid")
                base_probs.append(probs)
                m = chess.Move.from_uci(t["action"])
                if not board.is_legal(m):
                    raise ValueError("illegal own continuation")
                board.push(m)
                history.append(t["action"])
                if t["post"] != {"root_fen": pre["root_fen"], "moves": history}:
                    raise ValueError("own game post-history mismatch")
                legal_plies += 1
            last = moves[-1]["transition"]
            outcome = board.outcome(claim_draw=True)
            if outcome is None:
                if last["terminal_result"] is not None or last["terminal_termination"] is not None:
                    raise ValueError("unknown episode invents a terminal outcome")
                if last["rollout_cutoff"]:
                    counts["normal_cap_games"] += 1
                elif identity in truncations:
                    counts["epoch_truncated_games"] += 1
                else:
                    raise ValueError("unfinished episode lacks an explicit UNKNOWN boundary")
                counts["excluded_actions"] += len(rows)
                continue
            if (
                last["terminal_result"] != outcome.result()
                or last["terminal_termination"] != outcome.termination.name.lower()
            ):
                raise ValueError("own final result differs from independent rules")
            counts["complete_games"] += 1
            counts[
                "draw_games"
                if outcome.winner is None
                else "win_games"
                if outcome.winner == start_turn
                else "loss_games"
            ] += 1
            offset = len(features)
            features.extend(rows)
            anchors.extend(base_probs)
            labels.extend(
                1 if outcome.winner is None else 0 if c == outcome.winner else 2 for c in turns
            )
            groups.append(tuple(range(offset, len(features))))
            sources.append(identity[0])
            known_rows += len(rows)
        registered = {k: record["target_counts"][k] for k in counts}
        if counts != registered or legal_plies != record["fresh_transitions"]:
            raise ValueError("independent own-outcome counts disagree with immutable journal")
        if sha(path) != digest:
            raise ValueError("immutable old replay changed during read")
        receipts.append(
            {
                "journal_sha256": digest,
                "epoch": record["epoch"],
                "legal_plies": legal_plies,
                "known_rows": known_rows,
                **counts,
            }
        )
        del record, collection, episodes, unhashed, canonical
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    base_wdl = np.asarray(anchors, dtype=np.float32)
    train = tuple(g for g, s in zip(groups, sources, strict=True) if partition(s) != 0)
    val = tuple(g for g, s in zip(groups, sources, strict=True) if partition(s) == 0)
    if not train or not val or x.shape != (len(y), 840):
        raise ValueError("source-disjoint internal own-data split must be nonempty")
    if base_wdl.shape != (len(y), 3):
        raise ValueError("base WDL anchors do not align to completed-game positions")
    dataset_hash = hashlib.sha256(x.tobytes() + y.tobytes() + base_wdl.tobytes()).hexdigest()
    return (
        torch.from_numpy(x),
        torch.from_numpy(y),
        torch.from_numpy(base_wdl),
        train,
        val,
        receipts,
        dataset_hash,
    )

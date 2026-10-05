"""Compact exact current-piece/metadata features from verified own complete games.

These are the existing network's 20 invariants, not a new chess encoding.
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


def invariants(board):
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


def partition(source):
    # Original source game, not per-epoch game or oversampled alias, stays together.
    return int.from_bytes(hashlib.sha256(source.encode()).digest()[:8], "big") % 5


def prepare(paths, guard):
    features, labels, groups, sources, receipts = [], [], [], [], []
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
            episodes[t["source_id"], t["game_index"]].append(t)
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
            first = moves[0]
            pre = first["pre"]
            board = chess.Board(pre["root_fen"])
            for uci in pre["moves"]:
                m = chess.Move.from_uci(uci)
                if not board.is_legal(m):
                    raise ValueError("invalid original opening history")
                board.push(m)
            start_turn = board.turn
            history = list(pre["moves"])
            rows, turns = [], []
            for i, t in enumerate(moves):
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
                m = chess.Move.from_uci(t["action"])
                if not board.is_legal(m):
                    raise ValueError("illegal own continuation")
                board.push(m)
                history.append(t["action"])
                if t["post"] != {"root_fen": pre["root_fen"], "moves": history}:
                    raise ValueError("own game post-history mismatch")
                legal_plies += 1
            last = moves[-1]
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
    train = tuple(g for g, s in zip(groups, sources, strict=True) if partition(s) != 0)
    val = tuple(g for g, s in zip(groups, sources, strict=True) if partition(s) == 0)
    if not train or not val or x.shape != (len(y), 20):
        raise ValueError("source-disjoint internal own-data split must be nonempty")
    dataset_hash = hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()
    return torch.from_numpy(x), torch.from_numpy(y), train, val, receipts, dataset_hash

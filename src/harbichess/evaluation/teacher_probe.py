"""Frozen full-history policy/search/value diagnostics against budgeted Stockfish."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import resource
import subprocess
import time
from pathlib import Path

import chess
import chess.engine
import torch

from harbichess.backends.torch_backend import TorchPolicyValueBackend
from harbichess.backends.torch_network import load_weights, sha256
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove, ChessState
from harbichess.search.batching import SharedBatchEvaluator
from harbichess.search.evaluator import NeuralPositionEvaluator, PositionEvaluation
from harbichess.search.full_gumbel import FullGumbelConfig, FullGumbelMCTS


def freeze_panel(source: Path, *, per_game: int = 6) -> list[dict]:
    if per_game <= 0:
        raise ValueError("per_game must be positive")
    games = json.loads(source.read_text())["games"]
    rows, seen = [], set()
    for game_index, game in enumerate(games):
        board = chess.Board()
        histories = {}
        for ply, uci in enumerate(game["moves"], 1):
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                raise ValueError(f"illegal source game {game_index} ply {ply}")
            board.push(move)
            histories[ply] = (board.fen(), board.outcome(claim_draw=True) is None)
        span = len(game["moves"]) - 8
        if span <= 0:
            continue
        ending = board.outcome(claim_draw=True)
        for i in range(1, per_game + 1):
            ply = 8 + i * span // (per_game + 1)
            fen, ongoing = histories[ply]
            moves = game["moves"][:ply]
            digest = hashlib.sha256(json.dumps(moves).encode()).hexdigest()
            if not ongoing or digest in seen:
                continue
            seen.add(digest)
            rows.append(
                {
                    "id": digest,
                    "root_fen": chess.STARTING_FEN,
                    "moves": moves,
                    "fen": fen,
                    "ply": ply,
                    "game": game_index,
                    "family": game["opening_pair"],
                    "source_candidate_color": game["candidate_color"],
                    "observed_candidate_score": game["score"],
                    "observed_value": (
                        None
                        if ending is None
                        else 0
                        if ending.winner is None
                        else 1
                        if ending.winner == (fen.split()[1] == "w")
                        else -1
                    ),
                }
            )
    return rows


def reference_info(info: dict, board: chess.Board) -> dict:
    """Native engine WDL, in the actual root STM, without invented terminal labels."""
    wdl = info["wdl"].pov(board.turn)
    counts = (wdl.wins, wdl.draws, wdl.losses)
    if min(counts) < 0 or sum(counts) <= 0:
        raise ValueError("invalid engine WDL")
    probabilities = [x / sum(counts) for x in counts]
    score = info["score"].pov(board.turn)
    return {
        "move": info["pv"][0].uci(),
        "cp_or_mate_score": score.score(mate_score=10000),
        "mate": score.mate(),
        "wdl": probabilities,
        "expected_score": probabilities[0] + probabilities[1] / 2,
        "nodes": info.get("nodes"),
        "depth": info.get("depth"),
        "lowerbound": info.get("lowerbound", False),
        "upperbound": info.get("upperbound", False),
    }


class StockfishReference:
    def __init__(self, path: Path, rules: PythonChessRules, deadline: float, log) -> None:
        self.rules, self.deadline, self.log = rules, deadline, log
        self.engine = chess.engine.SimpleEngine.popen_uci(str(path), timeout=15)
        self.engine.configure({"Threads": 1, "Hash": 16, "UCI_ShowWDL": True})
        self.engine_id = self.engine.id
        self.calls = 0
        self.nodes = 0

    def analyse(self, state: ChessState, nodes: int, forced: ChessMove | None = None) -> dict:
        if nodes <= 0:
            raise ValueError("nodes must be positive")
        if time.perf_counter() >= self.deadline:
            raise TimeoutError("probe wall budget exhausted")
        board = self.rules.board(state)
        root_moves = [chess.Move.from_uci(forced.uci)] if forced else None
        if root_moves and root_moves[0] not in board.legal_moves:
            raise ValueError("forced reference root move must be legal")
        started = time.perf_counter()
        # Changing game identity sends ucinewgame; no prior query's TT is retained.
        info = self.engine.analyse(
            board, chess.engine.Limit(nodes=nodes), root_moves=root_moves, game=object()
        )
        result = reference_info(info, board)
        if result["lowerbound"] or result["upperbound"]:
            raise ValueError("bounded engine score is not an exact candidate reference")
        if forced and result["move"] != forced.uci:
            raise ValueError("engine did not honor forced root")
        result["wall_seconds"] = time.perf_counter() - started
        result["budget_nodes"] = nodes
        result["forced"] = forced.uci if forced else None
        self.calls += 1
        self.nodes += result["nodes"] or 0
        self.log.write(
            json.dumps(
                {
                    "query": self.calls,
                    "root_fen": state.root_fen,
                    "moves": [m.uci for m in state.moves],
                    **result,
                }
            )
            + "\n"
        )
        return result

    def close(self) -> None:
        self.engine.quit()


class OracleValueEvaluator:
    def __init__(self, neural, reference: StockfishReference) -> None:
        self.neural, self.reference = neural, reference

    def evaluate(self, state: ChessState) -> PositionEvaluation:
        neural = self.neural.evaluate(state)
        oracle = self.reference.analyse(state, 1024)
        win, _, loss = oracle["wdl"]
        return PositionEvaluation(neural.priors, win - loss)


def root_wdl(evaluator: NeuralPositionEvaluator, state: ChessState) -> list[float]:
    board = evaluator.rules.inspect(state)
    encoded = evaluator.encoder.encode_state(state, board)
    logits = evaluator.evaluator.evaluate(encoded).wdl_logits
    if not all(math.isfinite(x) for x in logits):
        raise ValueError("nonfinite neural WDL")
    maximum = max(logits)
    exp = [math.exp(x - maximum) for x in logits]
    return [x / sum(exp) for x in exp]


def paired_effect(rows: list[dict], *, arm: str, control: str) -> dict:
    families = sorted({r["family"] for r in rows})
    differences = []
    for family in families:
        members = [r for r in rows if r["family"] == family]
        differences.append(
            sum(r["regret"][control] - r["regret"][arm] for r in members) / len(members)
        )
    if not differences:
        return {"families": 0, "mechanism_supported": False}
    mean = sum(differences) / len(differences)
    rng = random.Random(20261003)
    bootstrap = sorted(
        sum(rng.choices(differences, k=len(differences))) / len(differences) for _ in range(10000)
    )
    radius = math.sqrt(2 * math.log(40) / len(differences))
    return {
        "families": len(families),
        "family_regret_reductions": differences,
        "mean_regret_reduction": mean,
        "conditional_bootstrap_95": [bootstrap[249], bootstrap[9749]],
        "hoeffding_95": [max(-1, mean - radius), min(1, mean + radius)],
        "mechanism_supported": mean >= 0.05 and bootstrap[249] > 0,
        "scope": "fixed development families, finite-budget restricted reference",
    }


def run_probe(
    source: Path,
    initial: Path,
    latest: Path,
    stockfish: Path,
    output: Path,
    *,
    wall_seconds: float = 900,
) -> dict:
    if wall_seconds <= 0:
        raise ValueError("positive wall budget required")
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    torch.set_num_threads(1)
    panel = freeze_panel(source)
    if not panel:
        raise ValueError("source produced an empty panel")
    output.mkdir(parents=True, exist_ok=False)
    panel_bytes = (json.dumps(panel, indent=2) + "\n").encode()
    (output / "panel.json").write_bytes(panel_bytes)
    started = time.perf_counter()
    deadline = started + wall_seconds
    rules, bridges, rows = PythonChessRules(), [], []
    metadata = {
        "schema": 1,
        "source_commit": source_commit,
        "source_sha256": sha256(source),
        "panel_sha256": hashlib.sha256(panel_bytes).hexdigest(),
        "initial_sha256": sha256(initial),
        "latest_sha256": sha256(latest),
        "stockfish_sha256": sha256(stockfish),
        "seed": 20261003,
        "panel_size": len(panel),
        "wall_budget_seconds": wall_seconds,
        "reference_semantics": "budgeted engine STM WDL, separate from observed outcomes",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def evaluator(path):
        bridge = SharedBatchEvaluator(
            TorchPolicyValueBackend(load_weights(path)), max_batch_size=1, max_wait_seconds=0
        )
        bridges.append(bridge)
        return NeuralPositionEvaluator(bridge, rules=rules)

    neural, last = evaluator(initial), evaluator(latest)
    status, reason = "completed", "all frozen positions measured"
    with (
        (output / "queries.jsonl").open("x") as queries,
        (output / "positions.jsonl").open("x") as out,
    ):
        reference = StockfishReference(stockfish, rules, deadline, queries)
        searches = {
            "neural16": FullGumbelMCTS(
                neural, rules=rules, config=FullGumbelConfig(simulations=16)
            ),
            "neural128": FullGumbelMCTS(
                neural, rules=rules, config=FullGumbelConfig(simulations=128)
            ),
            "oracle16": FullGumbelMCTS(
                OracleValueEvaluator(neural, reference),
                rules=rules,
                config=FullGumbelConfig(simulations=16),
            ),
        }
        try:
            for item in panel:
                state = ChessState(item["root_fen"], tuple(ChessMove(m) for m in item["moves"]))
                root = neural.evaluate(state)
                last_root = last.evaluate(state)
                if not all(math.isfinite(x) for x in (root.value, last_root.value)):
                    raise ValueError("nonfinite neural value")
                choices = {
                    "raw": max(root.priors, key=lambda x: x[1])[0],
                    "latest_raw": max(last_root.priors, key=lambda x: x[1])[0],
                }
                timers = {}
                for arm, search in searches.items():
                    arm_started = time.perf_counter()
                    choices[arm] = search.search(state, rng=random.Random(20261003)).selected_action
                    timers[arm] = time.perf_counter() - arm_started
                low = reference.analyse(state, 4096)
                high = reference.analyse(state, 32768)
                choices.update(
                    reference4k=ChessMove(low["move"]), reference32k=ChessMove(high["move"])
                )
                scores = {
                    move.uci: reference.analyse(state, 32768, move)
                    for move in sorted(set(choices.values()), key=lambda m: m.uci)
                }
                best = max(x["expected_score"] for x in scores.values())
                ranked = sorted(root.priors, key=lambda x: -x[1])
                teacher_rank = next(
                    i + 1 for i, (m, _) in enumerate(ranked) if m.uci == high["move"]
                )
                row = {
                    **item,
                    "choices": {k: v.uci for k, v in choices.items()},
                    "reference_low": low,
                    "reference_high": high,
                    "candidate_scores": scores,
                    "neural_value": root.value,
                    "latest_value": last_root.value,
                    "neural_wdl": root_wdl(neural, state),
                    "latest_wdl": root_wdl(last, state),
                    "reference_value": 2 * high["expected_score"] - 1,
                    "teacher_policy_rank": teacher_rank,
                    "legal_moves": len(root.priors),
                    "regret": {
                        k: best - scores[v.uci]["expected_score"] for k, v in choices.items()
                    },
                    "arm_wall_seconds": timers,
                }
                rows.append(row)
                out.write(json.dumps(row) + "\n")
                out.flush()
                queries.flush()
                if len(rows) % 8 == 0:
                    print(
                        json.dumps(
                            {"positions": len(rows), "elapsed": time.perf_counter() - started}
                        ),
                        flush=True,
                    )
        except (TimeoutError, ValueError, chess.engine.EngineError) as error:
            status, reason = "incomplete", f"{type(error).__name__}: {error}"
        finally:
            reference.close()
            for bridge in bridges:
                bridge.close()
    arms = ("raw", "latest_raw", "neural16", "neural128", "oracle16")
    n = max(1, len(rows))
    known = [r for r in rows if r["observed_value"] is not None]
    outcome_metrics = {}
    for name in ("neural", "latest"):
        losses, briers = [], []
        for r in known:
            label = {1: 0, 0: 1, -1: 2}[r["observed_value"]]
            probs = r[f"{name}_wdl"]
            losses.append(-math.log(max(probs[label], 1e-12)))
            briers.append(sum((p - (i == label)) ** 2 for i, p in enumerate(probs)))
        outcome_metrics[name] = {
            "observed_wdl_ce": sum(losses) / max(1, len(losses)),
            "observed_wdl_brier": sum(briers) / max(1, len(briers)),
            "known_rows": len(known),
            "scope": "outcomes of weak fixed-budget source players, not optimal play",
        }
    report = {
        **metadata,
        "status": status,
        "reason": reason,
        "positions_completed": len(rows),
        "wall_seconds": time.perf_counter() - started,
        "mean_regret": {k: sum(r["regret"][k] for r in rows) / n for k in arms},
        "reference_best_agreement": sum(
            r["choices"]["reference4k"] == r["choices"]["reference32k"] for r in rows
        )
        / n,
        "teacher_top16_coverage": sum(r["teacher_policy_rank"] <= 16 for r in rows) / n,
        "neural_reference_value_mae": sum(
            abs(r["neural_value"] - r["reference_value"]) for r in rows
        )
        / n,
        "latest_reference_value_mae": sum(
            abs(r["latest_value"] - r["reference_value"]) for r in rows
        )
        / n,
        "oracle_value_effect": paired_effect(rows, arm="oracle16", control="neural16"),
        "search_budget_effect": paired_effect(rows, arm="neural128", control="neural16"),
        "reference_queries": reference.calls,
        "reference_actual_nodes": reference.nodes,
        "observed_outcome_metrics": outcome_metrics,
        "self_cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime,
        "self_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "promotion_ready": False,
    }
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "initial", "latest", "stockfish", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--wall-seconds", type=float, default=900)
    args = parser.parse_args()
    result = run_probe(
        args.source,
        args.initial,
        args.latest,
        args.stockfish,
        args.output,
        wall_seconds=args.wall_seconds,
    )
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

"""Bounded independent 160-game replay and six prescribed NN packet checks."""
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import chess

STUDY = Path("/workspace/HarbiChess/experiments/ufuk/cpu-residual-value-v1/arena")
CHECKOUT = Path("/workspace/work/harbichess/cpu-additive-source-6fcc8b4")
PROTOCOL = STUDY / "protocol.json"
ARENA = Path("/workspace/work/harbichess/cpu-residual-value-v1-actual/arena-ram/arena")
COHORT_PATH = ARENA / "cohort-result.json"
OUT = Path("/workspace/work/harbichess/cpu-residual-value-v1-actual/independent-final-review")
OUT.mkdir(exist_ok=True)
if any((OUT / name).exists() for name in ("full160-independent-audit.json", "six-fixed-neural-packets.json", "inventory.json")):
    raise FileExistsError("independent review outputs already exist")
FIRST = time.time()
DEADLINE = FIRST + 600.0
MONO_DEADLINE = time.monotonic() + 600.0
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def guard():
    import shutil

    if time.monotonic() >= MONO_DEADLINE:
        raise TimeoutError("independent review original600 exhausted")
    if shutil.disk_usage("/workspace").free < 256 * 1024**2:
        raise RuntimeError("workspace disk floor")


def audit_games():
    p = json.loads(PROTOCOL.read_text())
    cohort = json.loads(COHORT_PATH.read_text())
    contract = cohort["contract"]
    assert cohort["status"] == "completed-games-not-strength"
    assert contract["inputs"]["protocol.json"] == sha(PROTOCOL)
    assert contract["source_commit"] == p["source_commit"] == "6fcc8b476d25495d1c9c413e55b2c7ba4794013e"
    expected = {(s, a, o) for s in p["match_seeds"] for a, o in p["tasks"]}
    assert len(expected) == len(cohort["rows"]) == 10
    assert {(r["seed"], r["arm"], r["opponent"]) for r in cohort["rows"]} == expected
    book_path = Path(p["book_path"])
    assert sha(book_path) == p["book_sha256"] == "9c36b5972c9676d19c358ca7e598b9f7f5d110b1947bed838cd660f644d845da"
    book = json.loads(book_path.read_text())["splits"]["arena"]
    assert len(book) >= 8
    protected = {str(PROTOCOL): sha(PROTOCOL), str(COHORT_PATH): sha(COHORT_PATH), str(book_path): sha(book_path)}
    outputs = {}
    groups = []
    sf_total = nn_total = eval_total = plies_total = 0
    caps_total = sf_over = sf_max = 0
    median_candidate_times = []
    for owner in cohort["rows"]:
        guard()
        seed, arm, opponent = owner["seed"], owner["arm"], owner["opponent"]
        assert owner["status"] == "completed-games-awaiting-independent-audit"
        assert owner["finished_epoch"] <= owner["original_deadline_epoch"]
        result_path = ARENA / f"{seed}-{arm}-vs-{opponent}.json"
        assert sha(result_path) == owner["result_sha256"]
        protected[str(result_path)] = sha(result_path)
        x = json.loads(result_path.read_text())
        outputs[seed, arm, opponent] = x
        assert x["schema"] == "cpu-all-root-quiescent-alpha-beta-arena-v1"
        assert x["source_commit"] == p["source_commit"] and x["protocol_sha256"] == sha(PROTOCOL)
        assert x["seed"] == seed and not x["GPU_used"] and not x["promotion_ready"]
        assert (x["search_nodes_per_move"], x["quiescence_plies"], x["max_plies"]) == (512, 2, 400)
        assert x["opening_source_sha256"] == p["book_sha256"]
        assert x["finished_epoch"] <= x["original_deadline_epoch"] == owner["original_deadline_epoch"]
        for key, helper in (("helper_sha256", "tournament.py"), ("search_helper_sha256", "search.py"), ("value_helper_sha256", "value.py")):
            assert x[key] == contract["inputs"][helper] == sha(STUDY / helper)
        roles = [(arm, "candidate_sha256")]
        if opponent != "SF512":
            roles.append((opponent, "opponent_sha256"))
        for role, key in roles:
            info = p["models"][str(seed)][role]
            model_path = Path(info["path"])
            model_sha = sha(model_path)
            assert model_sha == info["sha256"] == x[key]
            protected[str(model_path)] = model_sha
        if opponent == "SF512":
            assert (x["stockfish_sha256"], x["stockfish_nodes"], x["stockfish_threads"], x["stockfish_hash_mib"]) == (p["stockfish_sha256"], 512, 1, 16)
        assert len(x["games"]) == 16
        seen = set()
        scores, sf_nodes, nn_nodes, nn_evals, game_plies = [], [], [], [], 0
        caps = 0
        for game in x["games"]:
            guard()
            key = (game["opening_pair"], game["candidate_color"])
            assert key not in seen and 0 <= key[0] < 8 and key[1] in ("white", "black")
            seen.add(key)
            candidate_white = key[1] == "white"
            opening = book[key[0]]["opening"]["moves"]
            assert game["opening"] == opening and game["moves"][: len(opening)] == opening
            neural = {r["ply"]: r for r in game["search_by_move"]}
            sf = {r["ply"]: r for r in game["stockfish_nodes_by_move"]}
            assert len(neural) == len(game["search_by_move"]) and len(sf) == len(game["stockfish_nodes_by_move"])
            board = chess.Board()
            want_nn, want_sf = [], []
            for i, uci in enumerate(game["moves"]):
                move = chess.Move.from_uci(uci)
                assert board.is_legal(move)
                if i >= len(opening):
                    assert board.outcome(claim_draw=True) is None
                    ply = i + 1
                    if board.turn == candidate_white or opponent != "SF512":
                        want_nn.append(ply)
                        row = neural[ply]
                        assert row["candidate"] == (board.turn == candidate_white)
                        assert row["selected_move"] == uci
                        assert row["root_actions"] == row["legal_root_actions"] == board.legal_moves.count()
                        assert row["root_actions"] + 1 <= row["nodes"] <= 512
                        assert 0 <= row["evaluations"] <= row["nodes"] and 1 <= row["completed_depth"] <= 8
                        assert math.isfinite(row["value"]) and -2 <= row["value"] <= 2
                        assert math.isfinite(row["wall_seconds"]) and row["wall_seconds"] >= 0
                        nn_nodes.append(row["nodes"])
                        nn_evals.append(row["evaluations"])
                        if row["candidate"]:
                            median_candidate_times.append(row["wall_seconds"])
                    else:
                        want_sf.append(ply)
                        row = sf[ply]
                        assert type(row["nodes"]) is int and row["nodes"] >= 0
                        assert math.isfinite(row["wall_seconds"]) and row["wall_seconds"] >= 0
                        sf_nodes.append(row["nodes"])
                board.push(move)
            assert sorted(neural) == want_nn and sorted(sf) == want_sf
            assert len(game["move_wall_seconds"]) == len(game["moves"]) - len(opening)
            assert all(math.isfinite(v) and v >= 0 for v in game["move_wall_seconds"])
            assert board.ply() == game["plies"]
            outcome = board.outcome(claim_draw=True)
            if outcome is None:
                assert game["termination"] == "max_plies" and game["plies"] == 400
                caps += 1
                score = 0.5
            else:
                assert game["termination"] == outcome.termination.name.lower()
                score = 0.5 if outcome.winner is None else float(outcome.winner == candidate_white)
            assert score == game["score"]
            scores.append(score)
            game_plies += len(game["moves"]) - len(opening)
        assert seen == {(i, color) for i in range(8) for color in ("white", "black")}
        wins = sum(v == 1.0 for v in scores)
        draws = sum(v == 0.5 for v in scores)
        losses = 16 - wins - draws
        mean = (wins + draws / 2) / 16
        assert (x["summary"]["wins"], x["summary"]["draws"], x["summary"]["losses"], x["summary"]["score"], x["summary"]["capped_games"]) == (wins, draws, losses, mean, caps)
        sf_total += sum(sf_nodes)
        nn_total += sum(nn_nodes)
        eval_total += sum(nn_evals)
        plies_total += game_plies
        caps_total += caps
        sf_over += sum(v > 512 for v in sf_nodes)
        sf_max = max(sf_max, max(sf_nodes, default=0))
        groups.append({"seed": seed, "arm": arm, "opponent": opponent, "wins": wins, "draws": draws, "losses": losses, "score": mean, "caps": caps, "continuation_plies": game_plies, "sf_nodes": sum(sf_nodes), "sf_moves_over512": sum(v > 512 for v in sf_nodes), "neural_nodes": sum(nn_nodes), "nn_evaluations": sum(nn_evals)})
    assert all(sha(path) == digest for path, digest in protected.items())
    assert time.time() < DEADLINE
    return p, outputs, protected, groups, {"games": 160, "continuation_plies": plies_total, "actual_SF_nodes": sf_total, "neural_nodes": nn_total, "NN_evaluations": eval_total, "SF_moves_over512": sf_over, "max_actual_SF_nodes_per_move": sf_max, "caps": caps_total, "median_candidate_move_seconds": sorted(median_candidate_times)[len(median_candidate_times)//2]},


def reproduce_six(p, outputs):
    # Import exact study adapter + pinned source inference; schema 2 must use
    # full masked_policy_value so E8 logits and residual are composed once.
    sys.path[:0] = [str(STUDY), str(CHECKOUT / "src")]
    import torch
    from search import BudgetSearch
    from value import NeuralValue

    torch.set_num_threads(1)
    assert not torch.cuda.is_available()

    def search_guard():
        guard()

    packets = []
    for role in ("e8", "rebased", "residual"):
        x = outputs[20262705, role, "SF512"]
        model_info = p["models"]["20262705"][role]
        model_path = Path(model_info["path"])
        model_hash_before = sha(model_path)
        evaluator = NeuralValue(model_path)
        if role == "residual":
            spec = evaluator.model.specification["value_sparse"]
            assert spec["schema"] == 2 and spec["composition"] == "additive-v1"
            assert evaluator.value_mode == "schema2-core-full-base-plus-residual-once"
        search = BudgetSearch(evaluator, nodes=512, quiescence_plies=2, max_depth=8, guard=search_guard)
        selected = []
        for game in x["games"]:
            selected.extend((game, r) for r in game["search_by_move"] if r["candidate"])
            if len(selected) >= 2:
                break
        selected = selected[:2]
        assert len(selected) == 2
        for game, packet in selected:
            assert game["opening_pair"] == 0 and game["candidate_color"] == "white"
            board = chess.Board()
            for uci in game["moves"][: packet["ply"] - 1]:
                board.push_uci(uci)
            before = (board.fen(), tuple(board.move_stack))
            start = time.perf_counter()
            actual = search.search(board)
            observed = {"selected_move": actual.move.uci(), "value": actual.value, "nodes": actual.nodes, "evaluations": actual.evaluations, "completed_depth": actual.completed_depth, "root_actions": actual.root_actions}
            expected = {key: packet[key] for key in observed}
            assert observed == expected
            assert (board.fen(), tuple(board.move_stack)) == before
            packets.append({"role": role, "seed": 20262705, "opening_pair": game["opening_pair"], "candidate_color": game["candidate_color"], "ply": packet["ply"], "fullhistory_plies": board.ply(), "mover": "white" if board.turn else "black", "recorded_equals_reproduced": True, "actual_packet": observed, "model_sha256": model_hash_before, "wall_seconds": time.perf_counter() - start})
        assert sha(model_path) == model_hash_before
    return packets


if __name__ == "__main__":
    protocol, outputs, protected, groups, totals = audit_games()
    packets = reproduce_six(protocol, outputs)
    gate_rows = []
    for seed in protocol["match_seeds"]:
        val = lambda arm, opp: next(g["score"] for g in groups if (g["seed"], g["arm"], g["opponent"]) == (seed, arm, opp))
        gates = {"direct_same_search_e8_gt_060": val("residual", "e8") > 0.60, "pairedSF_gain_same_search_e8_gt_010": val("residual", "SF512") - val("e8", "SF512") > 0.10, "finalSF_ge_025": val("residual", "SF512") >= 0.25, "trained_vs_same_search_rebased_gt_060": val("residual", "rebased") > 0.60, "pairedSF_gain_rebased_gt_0": val("residual", "SF512") - val("rebased", "SF512") > 0.0, "caps_le_005": all(g["caps"] / 16 <= 0.05 for g in groups if g["seed"] == seed)}
        gate_rows.append({"seed": seed, "residual_e8": val("residual", "e8"), "residual_rebased": val("residual", "rebased"), "residual_sf": val("residual", "SF512"), "e8_sf": val("e8", "SF512"), "rebased_sf": val("rebased", "SF512"), "gates": gates, "passed": all(gates.values())})
    packet_record = {"schema": "residual-six-fixed-neural-packet-reproduction-v1", "status": "PASS-six-exact-packets-NOT-new-games", "source_commit": protocol["source_commit"], "torch_version": None, "threads": 1, "rows": packets, "matched_packets": len(packets), "new_games": 0, "new_training_updates": 0, "GPU_used": False}
    try:
        import torch
        packet_record["torch_version"] = torch.__version__
    except ImportError:
        pass
    record = {"schema": "residual-independent-full160-review-v1", "status": "PASS-full160-integrity-FAIL-development-strength-screen", "protocol_sha256": sha(PROTOCOL), "source_commit": protocol["source_commit"], "cohort_sha256": sha(COHORT_PATH), "arena_original_first_epoch": json.loads(COHORT_PATH.read_text())["contract"]["original_first_epoch"], "arena_original_deadline_epoch": json.loads(COHORT_PATH.read_text())["contract"]["original_deadline_epoch"], "audit_original_first_epoch": FIRST, "audit_original_deadline_epoch": DEADLINE, "audit_finished_epoch": time.time(), "groups": groups, "totals": totals, "screens": gate_rows, "independent_SF_gate_result": "FAIL" if not all(row["passed"] for row in gate_rows) else "PASS", "fixed_packet_reproductions": 6, "packet_sha256": None, "protected_sha256": protected, "training_updates": 0, "new_games": 0, "GPU_used": False, "limitations": ["Six fixed neural move packets were independently rerun; other neural searches were not rerun.", "Known8 development opening families are not virgin confirmation.", "Stockfish actual nodes may exceed requested512; neural/SF node counts are not equal compute.", "The paired intervals and gates are development diagnostics, not population strength confirmation."]}
    for name, value in (("six-fixed-neural-packets.json", packet_record), ("full160-independent-audit.json", record)):
        with (OUT / name).open("x") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n")
    record_sha = sha(OUT / "full160-independent-audit.json")
    packet_sha = sha(OUT / "six-fixed-neural-packets.json")
    inventory = {"schema": "residual-independent-review-inventory-v1", "files": {"review.py": sha(Path(__file__),), "full160-independent-audit.json": record_sha, "six-fixed-neural-packets.json": packet_sha}}
    with (OUT / "inventory.json").open("x") as f:
        json.dump(inventory, f, indent=2, sort_keys=True)
        f.write("\n")
    if sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file()) > 128 * 1024:
        raise RuntimeError("review artifact cap")
    if time.time() >= DEADLINE:
        raise TimeoutError("independent review finished after original600")
    print(json.dumps({"games": totals["games"], "plies": totals["continuation_plies"], "SF_nodes": totals["actual_SF_nodes"], "NN_nodes": totals["neural_nodes"], "six_packets": len(packets), "screens": gate_rows, "status": record["status"], "finish": time.time()}))

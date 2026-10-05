"""Pre-action masked own-search acting; raw network pi and actual mu are distinct."""

from __future__ import annotations

import json
import math
import random
from dataclasses import replace
from types import SimpleNamespace

import chess

from harbichess.chess.actions import legal_action_indices, move_to_action
from harbichess.core.state import ChessMove
from harbichess.search.full_gumbel import FullGumbelConfig
from harbichess.search.mate_certificates import (
    immediate_mating_moves,
    shield_visited_losses,
    visited_mate_in_one_losses,
)
from harbichess.search.ownsearch_wavefront import (
    BatchedPositionEvaluator,
    WavefrontGumbel,
)
from harbichess.selfplay.online_actor import OnlineActors
from harbichess.selfplay.online_epoch import (
    ONLINE_POLICY_EPOCH_SCHEMA,
    EpochAction,
    PolicyEpoch,
    _tuplify,
)
from harbichess.training.ownsearch_targets import SearchSchedule, validate_search_ledger
from harbichess.training.search_acting_policy import (
    GreedyOwnSearchConfig,
    ledger_semantics,
    mixture_policy,
)

LEDGER_SCHEMA = "pre-action-masked-search-behavior-v4"
LOSS_SHIELD_EPSILON = 1e-12
COLLECTION_SCHEMA = "full-history-search-acting-epoch-v2"


def deserialize_search_acting_epoch(data):
    from harbichess.selfplay.online_epoch import deserialize_policy_epoch

    payload = json.loads(data)
    if payload.get("schema") != COLLECTION_SCHEMA:
        raise ValueError("search acting collection schema differs")
    payload["schema"] = ONLINE_POLICY_EPOCH_SCHEMA
    epoch = deserialize_policy_epoch(json.dumps(payload).encode())
    return replace(epoch, schema=COLLECTION_SCHEMA)


def normalized(row):
    if not row or any(not math.isfinite(p) or p < 0 for p in row):
        raise ValueError("invalid finite policy support")
    total = math.fsum(row)
    if not math.isclose(total, 1, abs_tol=1e-6):
        raise ValueError("policy must sum to one")
    return tuple(p / total for p in row)


def actor_mu(row):
    # Exactly OnlineActors' T=1 normalization path, including floating-point order.
    policy = normalized(row)
    logs = tuple(math.log(p) if p > 0 else -math.inf for p in policy)
    largest = max(logs)
    weights = tuple(math.exp(p - largest) for p in logs)
    total = math.fsum(weights)
    return tuple(p / total for p in weights)


class MaskedRootEvaluator:
    """Search root inference uses the COMPLETE actor batch; simulate selected subset."""

    def __init__(self, evaluator, states, mask):
        self.evaluator = evaluator
        evaluations = evaluator.evaluate_many(states)
        self.states = tuple(s for s, selected in zip(states, mask, strict=True) if selected)
        self.outputs = tuple(e for e, selected in zip(evaluations, mask, strict=True) if selected)
        self.first = True

    def evaluate_many(self, states):
        if self.first:
            self.first = False
            if tuple(states) != self.states:
                raise ValueError("masked root batch changed before search")
            return self.outputs
        return self.evaluator.evaluate_many(states)


def collect_search_acting_epoch(
    actors,
    *,
    steps,
    infer,
    model_digest,
    behavior,
    config,
    schedule_rng,
    search_rngs,
    device,
    guard,
):
    if actors.config.temperature != 1 or actors.config.max_additional_plies > steps:
        raise ValueError("search acting requires T1 and whole-epoch bounded games")
    if any(g.state != actors.openings[g.opening_index].state for g in actors.games):
        raise ValueError("search acting epochs start at fresh opening histories")
    before = model_digest()
    start_step = actors.steps
    schedule_before = schedule_rng.getstate()
    search_before = [r.getstate() for r in search_rngs]
    cursor_before, actor_before = actors.cursor(), actors.rng.getstate()
    schedule = SearchSchedule(rng=schedule_rng, block_plies=config.block_plies)
    # Private Torch feature traffic only; shared wavefront and MLX keep tuple encodings.
    from harbichess.training.torch_array_encoder import (
        TorchArrayBoardEncoder,
        TorchArrayPolicyValueBackend,
    )

    evaluator = BatchedPositionEvaluator(
        TorchArrayPolicyValueBackend(behavior, device=device), actors.rules
    )
    evaluator.encoder = TorchArrayBoardEncoder(actors.rules)
    actions, receipts, groups = [], [], []
    for _ in range(steps):
        guard()
        states = actors.states
        legal = tuple(tuple(legal_action_indices(actors.rules.inspect(s))) for s in states)
        previous_selected = len(schedule.selected)
        schedule.before_actor_step(actors)  # Before network, search and actor outcomes.
        selected = schedule.selected[previous_selected:]
        slots = [i - len(actions) for i in selected]
        mask = [slot in slots for slot in range(len(states))]
        output = infer(states, legal)
        if not (
            len(output.policy)
            == len(output.wdl)
            == len(output.base_policy)
            == len(output.base_wdl)
            == len(states)
        ):
            raise ValueError("full actor inference count differs")
        raw = tuple(normalized(p) for p in output.policy)
        actor_policies = list(raw)
        if slots:
            root_evaluator = MaskedRootEvaluator(evaluator, states, mask)
            search = WavefrontGumbel(
                root_evaluator,
                actors.rules,
                FullGumbelConfig(
                    config.simulations,
                    config.max_considered_actions,
                    config.gumbel_scale,
                    config.value_scale,
                    config.maxvisit_init,
                    actors.config.claim_draw,
                ),
            )
            results = search.search_many(
                [states[s] for s in slots], [search_rngs[s] for s in slots], guard=guard
            )
            groups.append(
                dict(
                    first_collection_index=len(actions),
                    actor_batch_size=len(states),
                    selected_slots=slots,
                    selected_indices=selected,
                )
            )
            for slot, index, result in zip(slots, selected, results, strict=True):
                board = actors.rules.inspect(states[slot])
                raw_by_action = {
                    move_to_action(board, chess.Move.from_uci(m.uci)): p
                    for m, p in result.action_weights
                }
                raw_policy = normalized(tuple(raw_by_action[a] for a in legal[slot]))
                visited_moves = tuple(row.move for row in result.moves if row.visits > 0)
                checked_moves = () if result.certified_mates else visited_moves
                losses = (
                    visited_mate_in_one_losses(
                        actors.rules,
                        states[slot],
                        checked_moves,
                        claim_draw=actors.config.claim_draw,
                    )
                    if checked_moves
                    else ()
                )
                root_moves = tuple(move for move, _ in result.action_weights)
                root_policy = tuple(probability for _, probability in result.action_weights)
                shielded, selected_move, shield_status = shield_visited_losses(
                    root_moves,
                    root_policy,
                    result.moves,
                    tuple(move for move, _ in losses),
                    selected_action=result.selected_action,
                    certified_wins=result.certified_mates,
                    epsilon=LOSS_SHIELD_EPSILON,
                )
                by_action = {
                    move_to_action(board, chess.Move.from_uci(move.uci)): p
                    for move, p in zip(root_moves, shielded, strict=True)
                }
                policy = normalized(tuple(by_action[a] for a in legal[slot]))
                actor_policies[slot] = (
                    mixture_policy(
                        policy,
                        legal[slot].index(
                            move_to_action(board, chess.Move.from_uci(selected_move.uci))
                        ),
                        config.greedy_fraction,
                    )
                    if isinstance(config, GreedyOwnSearchConfig)
                    else policy
                )
                loss_moves = {move for move, _ in losses}
                if selected_move == result.selected_action:
                    selected_reason = (
                        "original-certified-losing-action-no-safe-alternative"
                        if selected_move in loss_moves
                        else "raw-search-selection"
                    )
                elif selected_move in visited_moves:
                    selected_reason = "fallback-visited-nonloss"
                else:
                    selected_reason = "fallback-unvisited-unknown"
                receipts.append(
                    dict(
                        collection_index=index,
                        legal_actions=list(legal[slot]),
                        raw_search_policy=list(raw_policy),
                        search_policy=list(policy),
                        **(
                            {"acting_policy": list(actor_policies[slot])}
                            if isinstance(config, GreedyOwnSearchConfig)
                            else {}
                        ),
                        raw_selected_action=result.selected_action.uci,
                        selected_action=selected_move.uci,
                        selected_action_reason=selected_reason,
                        root_value=result.root_value,
                        moves=[
                            dict(
                                move=m.move.uci,
                                visits=m.visits,
                                prior=m.prior,
                                mean_value=m.mean_value,
                            )
                            for m in result.moves
                        ],
                        simulations=result.simulations,
                        certified_mates=[m.uci for m in result.certified_mates],
                        visited_loss_checked_actions=sorted(m.uci for m in checked_moves),
                        certified_losing_actions=[
                            {
                                "move": move.uci,
                                "opponent_mates": [mate.uci for mate in replies],
                            }
                            for move, replies in losses
                        ],
                        loss_shield_status=shield_status,
                        loss_shield_epsilon=LOSS_SHIELD_EPSILON,
                        root_ply=states[slot].ply,
                        mover=actors.rules.view(states[slot]).side_to_move.value,
                    )
                )
        if model_digest() != before:
            raise ValueError("frozen model changed before actor action")
        transitions = actors.step(tuple(actor_policies))
        for slot, tr in enumerate(transitions):
            board = actors.rules.inspect(tr.pre)
            action_index = legal[slot].index(move_to_action(board, board.parse_uci(tr.action.uci)))
            mu = actor_mu(actor_policies[slot])
            if not math.isclose(
                mu[action_index], tr.behavior_probability, abs_tol=1e-12, rel_tol=0
            ):
                raise ValueError("actual search acting probability differs")
            # Actor accepted search probabilities; archive raw network pi honestly.
            tr = replace(tr, policy_probability=raw[slot][action_index])
            actions.append(
                EpochAction(
                    tr,
                    legal[slot],
                    raw[slot],
                    mu,
                    normalized(output.base_policy[slot]),
                    normalized(output.wdl[slot]),
                    normalized(output.base_wdl[slot]),
                )
            )
    if model_digest() != before or actors.steps - start_step != steps:
        raise ValueError("frozen search acting collection changed")
    truncations = actors.close_policy_epoch()
    epoch = PolicyEpoch(
        COLLECTION_SCHEMA,
        steps,
        start_step,
        actors.steps,
        tuple(actions),
        truncations,
        actors.cursor(),
        actors.rng.getstate(),
        before,
    )
    movers = {"white": 0, "black": 0}
    plies = {}
    for r in receipts:
        movers[r["mover"]] += 1
        key = str(r["root_ply"])
        plies[key] = plies.get(key, 0) + 1
    ledger = dict(
        schema=ledger_semantics(config)[0],
        realized_target_rows=len(receipts),
        mover_counts=movers,
        root_ply_counts=plies,
        model_digest=before,
        selected_indices=schedule.selected,
        offsets=schedule.offsets,
        roots=receipts,
        neural_batch_sizes=evaluator.batch_sizes,
        neural_positions=sum(evaluator.batch_sizes),
        behavior=ledger_semantics(config)[1],
        targets=("all-legal-Gumbel-plus-exact-one-ply-mate-win-and-visited-child-mate-loss-shield"),
        groups=groups,
        schedule_rng_before=schedule_before,
        schedule_rng_after=schedule_rng.getstate(),
        search_rng_before=search_before,
        search_rng_after=[r.getstate() for r in search_rngs],
        actor_cursor_before=cursor_before,
        actor_rng_before=actor_before,
    )
    if isinstance(config, GreedyOwnSearchConfig):
        ledger["greedy_fraction"] = config.greedy_fraction
    validate_search_acting(epoch, ledger, config, actors=actors)
    return epoch, ledger


def validate_search_acting(epoch, ledger, config, *, actors):
    if (ledger["schema"], ledger["behavior"]) != ledger_semantics(config):
        raise ValueError("search acting ledger semantics differ")
    if isinstance(config, GreedyOwnSearchConfig):
        if ledger.get("greedy_fraction") != config.greedy_fraction:
            raise ValueError("greedy acting mixture config/ledger differs")
    elif "greedy_fraction" in ledger:
        raise ValueError("legacy ledger cannot imply greedy acting")
    legacy = dict(ledger, schema="ownsearch-random-block-targets-v2")
    validate_search_ledger(
        epoch,
        legacy,
        config,
        rules=actors.rules,
        claim_draw=actors.config.claim_draw,
    )
    selected = {r["collection_index"]: r for r in ledger["roots"]}
    for index, receipt in selected.items():
        row = epoch.actions[index]
        board = actors.rules.inspect(row.transition.pre)
        moves_by_action = {
            move_to_action(board, chess.Move.from_uci(item["move"])): item
            for item in receipt["moves"]
        }
        legal_moves = tuple(
            ChessMove(moves_by_action[action]["move"]) for action in row.legal_actions
        )
        raw_policy = tuple(receipt["raw_search_policy"])
        if (
            len(raw_policy) != len(row.legal_actions)
            or any(not math.isfinite(p) or p < 0 for p in raw_policy)
            or not math.isclose(math.fsum(raw_policy), 1.0, abs_tol=1e-9, rel_tol=0)
        ):
            raise ValueError("source8 raw Gumbel policy support/normalization differs")
        wins = tuple(ChessMove(m) for m in receipt["certified_mates"])
        expected_wins = immediate_mating_moves(
            actors.rules, row.transition.pre, claim_draw=actors.config.claim_draw
        )
        if wins != expected_wins:
            raise ValueError("source8 immediate-win certificate inventory differs")
        visited = tuple(ChessMove(item["move"]) for item in receipt["moves"] if item["visits"] > 0)
        checked = () if wins else visited
        if receipt["visited_loss_checked_actions"] != sorted(m.uci for m in checked):
            raise ValueError("source8 visited-only loss-check inventory differs")
        expected_losses = (
            ()
            if wins
            else visited_mate_in_one_losses(
                actors.rules,
                row.transition.pre,
                checked,
                claim_draw=actors.config.claim_draw,
            )
        )
        loss_receipt = [
            {"move": move.uci, "opponent_mates": [mate.uci for mate in replies]}
            for move, replies in expected_losses
        ]
        if receipt["certified_losing_actions"] != loss_receipt:
            raise ValueError("source8 exact visited-child loss certificate differs")
        stats = tuple(
            SimpleNamespace(
                move=ChessMove(item["move"]),
                visits=item["visits"],
                mean_value=item["mean_value"],
                prior=item["prior"],
            )
            for item in receipt["moves"]
        )
        expected_policy, expected_selected, expected_status = shield_visited_losses(
            legal_moves,
            raw_policy,
            stats,
            tuple(move for move, _ in expected_losses),
            selected_action=ChessMove(receipt["raw_selected_action"]),
            certified_wins=wins,
            epsilon=LOSS_SHIELD_EPSILON,
        )
        if (
            receipt["loss_shield_status"] != expected_status
            or receipt["loss_shield_epsilon"] != LOSS_SHIELD_EPSILON
            or receipt["selected_action"] != expected_selected.uci
            or len(receipt["search_policy"]) != len(expected_policy)
            or any(
                not math.isclose(actual, expected, abs_tol=1e-15, rel_tol=0)
                for actual, expected in zip(receipt["search_policy"], expected_policy, strict=True)
            )
        ):
            raise ValueError("source8 loss shield policy/action/status differs")
        if receipt["selected_action"] == receipt["raw_selected_action"]:
            expected_reason = (
                "original-certified-losing-action-no-safe-alternative"
                if ChessMove(receipt["selected_action"]) in {move for move, _ in expected_losses}
                else "raw-search-selection"
            )
        elif ChessMove(receipt["selected_action"]) in visited:
            expected_reason = "fallback-visited-nonloss"
        else:
            expected_reason = "fallback-unvisited-unknown"
        if isinstance(config, GreedyOwnSearchConfig):
            expected_acting = mixture_policy(
                tuple(receipt["search_policy"]),
                row.legal_actions.index(
                    move_to_action(board, chess.Move.from_uci(expected_selected.uci))
                ),
                config.greedy_fraction,
            )
            if tuple(receipt.get("acting_policy", ())) != expected_acting:
                raise ValueError("greedy actual acting policy differs from search target mixture")
        elif "acting_policy" in receipt:
            raise ValueError("legacy ledger cannot supply a separate acting mixture")
        if receipt["selected_action_reason"] != expected_reason:
            raise ValueError("source8 deterministic fallback reason differs")
    replay = OnlineActors(
        actors.openings,
        config=actors.config,
        rng=random.Random(0),
        cursor=ledger["actor_cursor_before"],
    )
    replay.rng.setstate(_tuplify(ledger["actor_rng_before"]))
    groups = []
    width = actors.config.games
    for start in range(0, len(epoch.actions), width):
        rows = epoch.actions[start : start + width]
        slots = [slot for slot in range(width) if start + slot in selected]
        if slots:
            groups.append(
                dict(
                    first_collection_index=start,
                    actor_batch_size=width,
                    selected_slots=slots,
                    selected_indices=[start + s for s in slots],
                )
            )
        policies = tuple(
            tuple(
                selected[start + s][
                    "acting_policy"
                    if isinstance(config, GreedyOwnSearchConfig)
                    else "search_policy"
                ]
            )
            if start + s in selected
            else row.policy
            for s, row in enumerate(rows)
        )
        transitions = replay.step(policies)
        for slot, (actual, row) in enumerate(zip(transitions, rows, strict=True)):
            mu = actor_mu(policies[slot])
            board = actors.rules.inspect(row.transition.pre)
            index = row.legal_actions.index(
                move_to_action(board, board.parse_uci(actual.action.uci))
            )
            expected = replace(actual, policy_probability=row.policy[index])
            if expected != row.transition or mu != row.behavior_policy:
                raise ValueError("raw pi/actual mu/action RNG replay differs")
    replay.close_policy_epoch()
    if (
        groups != ledger["groups"]
        or replay.cursor() != epoch.next_actor_cursor
        or replay.rng.getstate() != epoch.next_actor_rng_state
    ):
        raise ValueError("masked full actor batch/cursor/RNG differs")

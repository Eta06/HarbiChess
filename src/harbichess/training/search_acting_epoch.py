"""Pre-action masked own-search acting; raw network pi and actual mu are distinct."""

from __future__ import annotations

import json
import math
import random
from dataclasses import replace

import chess

from harbichess.backends.torch_backend import TorchPolicyValueBackend
from harbichess.chess.actions import legal_action_indices, move_to_action
from harbichess.search.full_gumbel import FullGumbelConfig
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

LEDGER_SCHEMA = "pre-action-masked-search-behavior-v2"
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
        self.states = tuple(
            s for s, selected in zip(states, mask, strict=True) if selected
        )
        self.outputs = tuple(
            e for e, selected in zip(evaluations, mask, strict=True) if selected
        )
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
    evaluator = BatchedPositionEvaluator(
        TorchPolicyValueBackend(behavior, device=device), actors.rules
    )
    actions, receipts, groups = [], [], []
    for _ in range(steps):
        guard()
        states = actors.states
        legal = tuple(
            tuple(legal_action_indices(actors.rules.inspect(s))) for s in states
        )
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
                by_action = {
                    move_to_action(board, chess.Move.from_uci(m.uci)): p
                    for m, p in result.action_weights
                }
                policy = normalized(tuple(by_action[a] for a in legal[slot]))
                actor_policies[slot] = policy
                receipts.append(
                    dict(
                        collection_index=index,
                        legal_actions=list(legal[slot]),
                        search_policy=list(policy),
                        selected_action=result.selected_action.uci,
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
                        root_ply=states[slot].ply,
                        mover=actors.rules.view(states[slot]).side_to_move.value,
                    )
                )
        if model_digest() != before:
            raise ValueError("frozen model changed before actor action")
        transitions = actors.step(tuple(actor_policies))
        for slot, tr in enumerate(transitions):
            board = actors.rules.inspect(tr.pre)
            action_index = legal[slot].index(
                move_to_action(board, board.parse_uci(tr.action.uci))
            )
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
        schema=LEDGER_SCHEMA,
        realized_target_rows=len(receipts),
        mover_counts=movers,
        root_ply_counts=plies,
        model_digest=before,
        selected_indices=schedule.selected,
        offsets=schedule.offsets,
        roots=receipts,
        neural_batch_sizes=evaluator.batch_sizes,
        neural_positions=sum(evaluator.batch_sizes),
        behavior="raw-T1-except-preselected-search-policy-T1;no-PPO",
        targets="all-legal-Gumbel-completed-Q-policy-detached-supervision",
        groups=groups,
        schedule_rng_before=schedule_before,
        schedule_rng_after=schedule_rng.getstate(),
        search_rng_before=search_before,
        search_rng_after=[r.getstate() for r in search_rngs],
        actor_cursor_before=cursor_before,
        actor_rng_before=actor_before,
    )
    validate_search_acting(epoch, ledger, config, actors=actors)
    return epoch, ledger


def validate_search_acting(epoch, ledger, config, *, actors):
    if (
        ledger["schema"] != LEDGER_SCHEMA
        or ledger["behavior"] != "raw-T1-except-preselected-search-policy-T1;no-PPO"
    ):
        raise ValueError("search acting ledger semantics differ")
    legacy = dict(ledger, schema="ownsearch-random-block-targets-v1")
    validate_search_ledger(epoch, legacy, config, rules=actors.rules)
    selected = {r["collection_index"]: r for r in ledger["roots"]}
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
            tuple(selected[start + s]["search_policy"])
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

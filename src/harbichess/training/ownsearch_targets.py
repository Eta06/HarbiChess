"""Own-model sparse Gumbel policy supervision; actual raw-policy behavior is unchanged."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import chess

from harbichess.backends.torch_backend import TorchPolicyValueBackend
from harbichess.chess.actions import move_to_action
from harbichess.search.full_gumbel import FullGumbelConfig
from harbichess.search.mate_certificates import immediate_mating_moves
from harbichess.search.ownsearch_wavefront import (
    BatchedPositionEvaluator,
    WavefrontGumbel,
)
from harbichess.training.torch_fullgame_ppo import torch_model_digest


@dataclass(frozen=True, slots=True)
class OwnSearchConfig:
    simulations: int
    max_considered_actions: int
    gumbel_scale: float
    value_scale: float
    maxvisit_init: float
    block_plies: int

    def __post_init__(self):
        FullGumbelConfig(
            self.simulations,
            self.max_considered_actions,
            self.gumbel_scale,
            self.value_scale,
            self.maxvisit_init,
        )
        if type(self.block_plies) is not int or self.block_plies <= 0:
            raise ValueError(
                "own search requires explicit positive per-game schedule block"
            )


class SearchSchedule:
    """Draw exogenous per-game block offsets before corresponding actor outcomes."""

    def __init__(self, *, rng, block_plies):
        self.rng, self.block_plies = rng, block_plies
        self.pending = {}
        self.offsets = []
        self.selected = []
        self.actions_seen = 0

    def before_actor_step(self, actors):
        active_keys = set()
        for slot, game in enumerate(actors.games):
            opening = actors.openings[game.opening_index]
            additional = game.state.ply - opening.state.ply
            block, offset = divmod(additional, self.block_plies)
            key = (opening.source_id, game.game_index, slot, block)
            active_keys.add(key)
            if key not in self.pending:
                chosen = self.rng.randrange(self.block_plies)
                self.pending[key] = chosen
                self.offsets.append(
                    dict(
                        source_id=key[0],
                        game_index=key[1],
                        slot=slot,
                        block=block,
                        chosen_offset=chosen,
                        first_collection_index=self.actions_seen + slot,
                    )
                )
            if offset == self.pending[key]:
                self.selected.append(self.actions_seen + slot)
        self.pending = {k: v for k, v in self.pending.items() if k in active_keys}
        self.actions_seen += len(actors.games)


def validate_search_ledger(epoch, ledger, config, *, rules, claim_draw=True):
    """Reproduce only exogenous schedule and legal receipt mapping; no optimizer/search rerun."""
    from harbichess.selfplay.online_epoch import _tuplify

    rng = random.Random()
    rng.setstate(_tuplify(ledger["schedule_rng_before"]))
    if (
        ledger["schema"] != "ownsearch-random-block-targets-v2"
        or ledger["model_digest"] != epoch.model_digest
    ):
        raise ValueError("own-search ledger schema/behavior mismatch")
    search_rngs = []
    for state in ledger["search_rng_before"]:
        own = random.Random()
        own.setstate(_tuplify(state))
        search_rngs.append(own)
    offsets = {}
    expected_selected = []
    seen_offsets = []
    games = {}
    for index, row in enumerate(epoch.actions):
        tr = row.transition
        identity = (tr.source_id, tr.game_index, tr.slot)
        first = games.setdefault(identity, tr.pre.ply)
        block, offset = divmod(tr.pre.ply - first, config.block_plies)
        key = (*identity, block)
        if key not in offsets:
            chosen = rng.randrange(config.block_plies)
            offsets[key] = chosen
            seen_offsets.append(
                dict(
                    source_id=key[0],
                    game_index=key[1],
                    slot=key[2],
                    block=block,
                    chosen_offset=chosen,
                    first_collection_index=index,
                )
            )
        if offset == offsets[key]:
            expected_selected.append(index)
    if (
        seen_offsets != ledger["offsets"]
        or expected_selected != ledger["selected_indices"]
        or rng.getstate() != _tuplify(ledger["schedule_rng_after"])
    ):
        raise ValueError("exogenous own-search schedule/RNG/selection mismatch")
    if ledger["realized_target_rows"] != len(expected_selected):
        raise ValueError("ownsearch realized policy target count differs")
    if len(ledger["roots"]) != len(expected_selected):
        raise ValueError("selected search root inventory differs")
    for index, receipt in zip(expected_selected, ledger["roots"], strict=True):
        row = epoch.actions[index]
        if receipt["collection_index"] != index or receipt["legal_actions"] != list(
            row.legal_actions
        ):
            raise ValueError("search root chronology/legal support differs")
        if (
            receipt["root_ply"] != row.transition.pre.ply
            or receipt["mover"] != rules.view(row.transition.pre).side_to_move.value
        ):
            raise ValueError("ownsearch root mover/history strata mismatch")
        moves = tuple(receipt["moves"])
        board = rules.inspect(row.transition.pre)
        mapped = {
            move_to_action(board, chess.Move.from_uci(m["move"])): m for m in moves
        }
        if set(mapped) != set(row.legal_actions) or len(mapped) != len(moves):
            raise ValueError("search receipt move support differs")
        for _ in row.legal_actions:
            search_rngs[row.transition.slot].random()
        visits = [mapped[a]["visits"] for a in row.legal_actions]
        probabilities = receipt["search_policy"]
        certified = [
            m.uci
            for m in immediate_mating_moves(
                rules, row.transition.pre, claim_draw=claim_draw
            )
        ]
        if receipt.get("certified_mates") != certified:
            raise ValueError("exact one-ply mate certificate ledger differs")
        if (
            any(type(v) is not int or v < 0 for v in visits)
            or sum(visits) != config.simulations
        ):
            raise ValueError("own-search simulation/visit budget mismatch")
        if (
            len(probabilities) != len(row.legal_actions)
            or any(not math.isfinite(p) or p < 0 for p in probabilities)
            or abs(math.fsum(probabilities) - 1) > 1e-9
        ):
            raise ValueError("search target finite legal normalization mismatch")
        if receipt["selected_action"] not in {m["move"] for m in moves}:
            raise ValueError("search selected action illegal")
        if certified:
            if receipt["selected_action"] not in certified:
                raise ValueError("exact-mate root did not select a certified winning move")
            if abs(receipt["root_value"] - 1.0) > 1e-12:
                raise ValueError("terminal mate backup has wrong mover perspective")
            selected_index = move_to_action(
                board, chess.Move.from_uci(receipt["selected_action"])
            )
            if mapped[selected_index]["visits"] != config.simulations:
                raise ValueError("exact-mate solved route changed simulation budget")
            nonmate_mass = math.fsum(
                probability
                for action, probability in zip(
                    row.legal_actions, probabilities, strict=True
                )
                if mapped[action]["move"] not in certified
            )
            if nonmate_mass > 1e-9 or any(probability <= 0 for probability in probabilities):
                raise ValueError("certified policy lost exactness or full legal support")
    actual_movers = {"white": 0, "black": 0}
    actual_plies = {}
    for receipt in ledger["roots"]:
        actual_movers[receipt["mover"]] += 1
        key = str(receipt["root_ply"])
        actual_plies[key] = actual_plies.get(key, 0) + 1
    if (
        ledger["mover_counts"] != actual_movers
        or ledger["root_ply_counts"] != actual_plies
    ):
        raise ValueError("ownsearch root mover/ply summary differs")
    if [rng.getstate() for rng in search_rngs] != [
        _tuplify(state) for state in ledger["search_rng_after"]
    ]:
        raise ValueError("actor-local search RNG receipt mismatch")


def make_search_ledger(
    epoch, schedule, behavior, rules, config, search_rngs, *, device, guard, claim_draw
):
    evaluator = BatchedPositionEvaluator(
        TorchPolicyValueBackend(behavior, device=device), rules
    )
    search = WavefrontGumbel(
        evaluator,
        rules,
        FullGumbelConfig(
            config.simulations,
            config.max_considered_actions,
            config.gumbel_scale,
            config.value_scale,
            config.maxvisit_init,
            claim_draw,
        ),
    )
    receipts = []
    # Actor-local search RNGs are reused sequentially. One batch is at most one root/slot.
    queues = {slot: [] for slot in range(len(search_rngs))}
    for index in schedule.selected:
        queues[epoch.actions[index].transition.slot].append(index)
    groups = []
    while any(queues.values()):
        groups.append([rows.pop(0) for rows in queues.values() if rows])
    for indices in groups:
        guard()
        rows = [epoch.actions[i] for i in indices]
        results = search.search_many(
            [r.transition.pre for r in rows],
            [search_rngs[r.transition.slot] for r in rows],
            guard=guard,
        )
        for index, row, result in zip(indices, rows, results, strict=True):
            board = rules.inspect(row.transition.pre)
            policy = {
                move_to_action(board, chess.Move.from_uci(move.uci)): p
                for move, p in result.action_weights
            }
            receipts.append(
                dict(
                    collection_index=index,
                    legal_actions=list(row.legal_actions),
                    search_policy=[policy[a] for a in row.legal_actions],
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
                    certified_mates=[m.uci for m in result.certified_mates],
                )
            )
    receipts.sort(key=lambda row: row["collection_index"])
    mover_counts = {"white": 0, "black": 0}
    ply_counts = {}
    for receipt in receipts:
        state = epoch.actions[receipt["collection_index"]].transition.pre
        side = rules.view(state).side_to_move.value
        receipt["root_ply"] = state.ply
        receipt["mover"] = side
        mover_counts[side] += 1
        key = str(state.ply)
        ply_counts[key] = ply_counts.get(key, 0) + 1
    return dict(
        schema="ownsearch-random-block-targets-v2",
        realized_target_rows=len(receipts),
        mover_counts=mover_counts,
        root_ply_counts=ply_counts,
        model_digest=torch_model_digest(behavior),
        selected_indices=schedule.selected,
        offsets=schedule.offsets,
        roots=receipts,
        neural_batch_sizes=evaluator.batch_sizes,
        neural_positions=sum(evaluator.batch_sizes),
        behavior="raw-current-T1-not-search-action-policy",
        targets="all-legal-Gumbel-plus-exact-one-ply-mate-solver-policy",
    )

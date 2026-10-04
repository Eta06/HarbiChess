"""Teacher-free, complete-game WDL targets for frozen-policy collection epochs."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise

from harbichess.chess.actions import move_to_action
from harbichess.chess.rules import PythonChessRules
from harbichess.selfplay.online_actor import ActorTransition
from harbichess.selfplay.online_epoch import EpochAction, PolicyEpoch
from harbichess.training.online_targets import WDL, expected_score

FULLGAME_TARGET_SCHEMA = "completed-own-game-terminal-wdl-v1"


@dataclass(frozen=True, slots=True)
class FullGameTarget:
    game_index: int
    source_id: str
    slot: int
    transition: ActorTransition
    legal_actions: tuple[int, ...]
    policy: tuple[float, ...]
    behavior_policy: tuple[float, ...]
    base_policy: tuple[float, ...]
    base_value_wdl: WDL
    target_wdl: WDL
    advantage: float
    action_index: int
    action_ratio_old: float
    termination: str
    game_length: int


@dataclass(frozen=True, slots=True)
class FullGameTargetSet:
    schema: str
    targets: tuple[FullGameTarget, ...]
    complete_games: int
    normal_cap_games: int
    epoch_truncated_games: int
    excluded_actions: int
    win_games: int
    draw_games: int
    loss_games: int


class GameBalancedSampler:
    """Sample completed games uniformly, then one position within each game."""

    def __init__(self, targets: tuple[FullGameTarget, ...], *, seed: int):
        if not targets or type(seed) is not int or seed < 0:
            raise ValueError("game-balanced sampler needs targets and an explicit seed")
        by_game: dict[tuple[str, int], list[int]] = defaultdict(list)
        for index, target in enumerate(targets):
            by_game[(target.source_id, target.game_index)].append(index)
        self.targets = targets
        self.indices = {key: tuple(value) for key, value in sorted(by_game.items())}
        self.game_ids = tuple(self.indices)
        self.rng = random.Random(seed)

    def sample(self, size: int) -> tuple[FullGameTarget, ...]:
        if type(size) is not int or size <= 0:
            raise ValueError("game-balanced batch size must be positive")
        selected = []
        for game in self.rng.choices(self.game_ids, k=size):
            selected.append(self.targets[self.rng.choice(self.indices[game])])
        return tuple(selected)


def build_fullgame_targets(
    rules: PythonChessRules,
    epoch: PolicyEpoch,
    *,
    claim_draw: bool,
) -> FullGameTargetSet:
    """Label only fully observed games; caps and epoch tails stay unknown.

    Episodes use full histories and are checked for legal continuity before
    their final outcome is propagated to every mover. Behavior ratios are
    per-action pi/mu; this record alone does not claim full-trajectory
    off-policy correction.
    """
    if type(claim_draw) is not bool:
        raise ValueError("full-game target construction needs explicit claim_draw")
    truncation_ids = {
        (row["source_id"], row["game_index"])
        for row in epoch.truncations
        if row.get("termination") == "policy-epoch-truncation"
    }
    groups: dict[tuple[str, int], list[EpochAction]] = defaultdict(list)
    for sample in epoch.actions:
        transition = sample.transition
        groups[(transition.source_id, transition.game_index)].append(sample)
    if not epoch.actions:
        raise ValueError("full-game targets require at least one collected transition")
    targets = []
    complete = caps = truncated = excluded = wins = draws = losses = 0
    for identity, rows in groups.items():
        first, last = rows[0].transition, rows[-1].transition
        if any(
            row.transition.slot != first.slot
            or row.transition.source_id != first.source_id
            or row.transition.game_index != first.game_index
            for row in rows
        ):
            raise ValueError("episode group mixes actor slots, IDs, or sources")
        for previous, current in pairwise(rows):
            if previous.transition.post != current.transition.pre:
                raise ValueError("complete-game replay history is discontinuous")
        for sample in rows:
            transition = sample.transition
            if rules.outcome(transition.pre, claim_draw=claim_draw) is not None:
                raise ValueError("episode contains an action after terminal state")
            if rules.apply(transition.pre, transition.action) != transition.post:
                raise ValueError("episode action does not reproduce full-history post-state")
            if len(sample.legal_actions) != len(sample.policy) or len(sample.policy) != len(
                sample.behavior_policy
            ) or len(sample.policy) != len(sample.base_policy):
                raise ValueError("episode stored policies have inconsistent legal support")
        outcome = rules.outcome(last.post, claim_draw=claim_draw)
        if outcome is None:
            if last.rollout_cutoff:
                caps += 1
            elif identity in truncation_ids:
                truncated += 1
            else:
                # Every still-live game at the epoch boundary must have a
                # matching explicit truncation receipt.
                raise ValueError("unfinished game lacks cap or policy-epoch truncation")
            excluded += len(rows)
            continue
        if last.terminal_result != outcome.result.value or (
            last.terminal_termination != outcome.termination
        ):
            raise ValueError("stored terminal metadata differs from full-history chess rules")
        complete += 1
        root_result = outcome.value_for(rules.view(first.pre).side_to_move)
        wins += int(root_result == 1)
        draws += int(root_result == 0)
        losses += int(root_result == -1)
        for sample in rows:
            transition = sample.transition
            value = outcome.value_for(rules.view(transition.pre).side_to_move)
            target = (float(value == 1), float(value == 0), float(value == -1))
            action_index = sample.legal_actions.index(move_to_action(
                rules.inspect(transition.pre),
                rules.inspect(transition.pre).parse_uci(transition.action.uci),
            ))
            old_mu = sample.behavior_policy[action_index]
            if old_mu <= 0:
                raise ValueError("played action has zero behavior probability")
            targets.append(FullGameTarget(
                game_index=transition.game_index,
                source_id=transition.source_id,
                slot=transition.slot,
                transition=transition,
                legal_actions=sample.legal_actions,
                policy=sample.policy,
                behavior_policy=sample.behavior_policy,
                base_policy=sample.base_policy,
                base_value_wdl=sample.base_wdl,
                target_wdl=target,
                advantage=expected_score(target) - expected_score(sample.online_pre_wdl),
                action_index=action_index,
                action_ratio_old=sample.policy[action_index] / old_mu,
                termination=outcome.termination,
                game_length=len(rows),
            ))
    return FullGameTargetSet(
        schema=FULLGAME_TARGET_SCHEMA,
        targets=tuple(targets),
        complete_games=complete,
        normal_cap_games=caps,
        epoch_truncated_games=truncated,
        excluded_actions=excluded,
        win_games=wins,
        draw_games=draws,
        loss_games=losses,
    )

"""Frozen-policy full-history collection epochs for terminal-return learning."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass

from harbichess.chess.actions import legal_action_indices, move_to_action
from harbichess.core.state import ChessMove, ChessState
from harbichess.selfplay.online_actor import ActorTransition, OnlineActors


@dataclass(frozen=True, slots=True)
class EpochPolicyOutput:
    """Canonical legal-action rows from current and immutable base networks."""

    policy: tuple[tuple[float, ...], ...]
    wdl: tuple[tuple[float, float, float], ...]
    base_policy: tuple[tuple[float, ...], ...]
    base_wdl: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True, slots=True)
class EpochAction:
    transition: ActorTransition
    legal_actions: tuple[int, ...]
    policy: tuple[float, ...]
    behavior_policy: tuple[float, ...]
    base_policy: tuple[float, ...]
    online_pre_wdl: tuple[float, float, float]
    base_wdl: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class PolicyEpoch:
    schema: str
    requested_steps: int
    start_actor_step: int
    end_actor_step: int
    actions: tuple[EpochAction, ...]
    truncations: tuple[dict, ...]
    next_actor_cursor: dict
    next_actor_rng_state: object
    model_digest: str


ONLINE_POLICY_EPOCH_SCHEMA = "full-history-frozen-policy-epoch-v1"


def _state_payload(state: ChessState) -> dict:
    return {"root_fen": state.root_fen, "moves": [move.uci for move in state.moves]}


def _transition_payload(row: ActorTransition) -> dict:
    return {
        "slot": row.slot,
        "game_index": row.game_index,
        "source_id": row.source_id,
        "pre": _state_payload(row.pre),
        "action": row.action.uci,
        "post": _state_payload(row.post),
        "policy_probability": row.policy_probability,
        "behavior_probability": row.behavior_probability,
        "rollout_cutoff": row.rollout_cutoff,
        "terminal_result": row.terminal_result,
        "terminal_termination": row.terminal_termination,
    }


def _state_from(payload: dict) -> ChessState:
    return ChessState(payload["root_fen"], tuple(ChessMove(move) for move in payload["moves"]))


def _transition_from(payload: dict) -> ActorTransition:
    return ActorTransition(
        payload["slot"],
        payload["game_index"],
        payload["source_id"],
        _state_from(payload["pre"]),
        ChessMove(payload["action"]),
        _state_from(payload["post"]),
        payload["policy_probability"],
        payload["behavior_probability"],
        payload["rollout_cutoff"],
        payload["terminal_result"],
        payload["terminal_termination"],
    )


def _tuplify(value):
    return tuple(_tuplify(row) for row in value) if isinstance(value, list) else value


def serialize_policy_epoch(epoch: PolicyEpoch) -> bytes:
    """Canonical closed-epoch receipt suitable for a hashed replay artifact."""
    payload = {
        "schema": epoch.schema,
        "requested_steps": epoch.requested_steps,
        "start_actor_step": epoch.start_actor_step,
        "end_actor_step": epoch.end_actor_step,
        "actions": [
            {
                "transition": _transition_payload(row.transition),
                "legal_actions": list(row.legal_actions),
                "policy": list(row.policy),
                "behavior_policy": list(row.behavior_policy),
                "base_policy": list(row.base_policy),
                "online_pre_wdl": list(row.online_pre_wdl),
                "base_wdl": list(row.base_wdl),
            }
            for row in epoch.actions
        ],
        "truncations": list(epoch.truncations),
        "next_actor_cursor": epoch.next_actor_cursor,
        "next_actor_rng_state": epoch.next_actor_rng_state,
        "model_digest": epoch.model_digest,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def deserialize_policy_epoch(data: bytes) -> PolicyEpoch:
    """Read a serialized closed epoch without discarding actor or RNG cursors."""
    payload = json.loads(data)
    if payload.get("schema") != ONLINE_POLICY_EPOCH_SCHEMA:
        raise ValueError("policy epoch artifact schema mismatch")
    actions = tuple(
        EpochAction(
            _transition_from(row["transition"]),
            tuple(row["legal_actions"]),
            tuple(row["policy"]),
            tuple(row["behavior_policy"]),
            tuple(row["base_policy"]),
            tuple(row["online_pre_wdl"]),
            tuple(row["base_wdl"]),
        )
        for row in payload["actions"]
    )
    if not actions or type(payload["requested_steps"]) is not int:
        raise ValueError("serialized policy epoch must contain requested steps and samples")
    return PolicyEpoch(
        schema=payload["schema"],
        requested_steps=payload["requested_steps"],
        start_actor_step=payload["start_actor_step"],
        end_actor_step=payload["end_actor_step"],
        actions=actions,
        truncations=tuple(payload["truncations"]),
        next_actor_cursor=payload["next_actor_cursor"],
        next_actor_rng_state=_tuplify(payload["next_actor_rng_state"]),
        model_digest=payload["model_digest"],
    )


def collect_policy_epoch(
    actors: OnlineActors,
    *,
    steps: int,
    infer: Callable[[tuple[ChessState, ...], tuple[tuple[int, ...], ...]], EpochPolicyOutput],
    model_digest: Callable[[], str],
) -> PolicyEpoch:
    """Collect exact actor transitions under one unchanged network snapshot.

    The caller owns Torch/CUDA inference and checkpointing. It must close over
    an immutable evaluation copy of the behavior model; digest checks cannot
    detect a mutate-then-restore bug in a mutable optimizer model. The callback
    returns probabilities in each supplied action-index order. Completed games
    and rollout caps are preserved by the actor; every remaining game is
    explicitly closed as an unknown policy-epoch truncation at the boundary.
    """
    if type(steps) is not int or steps <= 0:
        raise ValueError("policy epoch requires a positive integer step count")
    before = model_digest()
    if not isinstance(before, str) or not before:
        raise ValueError("policy snapshot digest must be a nonempty string")
    start_step = actors.steps
    if actors.config.temperature != 1.0:
        raise ValueError("full-game collection requires on-policy temperature exactly one")
    if actors.config.max_additional_plies > steps:
        raise ValueError("every epoch-opening game must terminate or cap within this epoch")
    for game in actors.games:
        if game.state != actors.openings[game.opening_index].state:
            raise ValueError("each policy epoch must start from fresh opening roots")
    actions: list[EpochAction] = []
    for _ in range(steps):
        states = actors.states
        legal = tuple(
            tuple(legal_action_indices(actors.rules.inspect(state))) for state in states
        )
        output = infer(states, legal)
        if not isinstance(output, EpochPolicyOutput) or not (
            len(output.policy) == len(output.wdl) == len(output.base_policy)
            == len(output.base_wdl) == len(states)
        ):
            raise ValueError("epoch inference must return one policy/value/base row per actor")
        policy_rows, base_policy_rows, wdl_rows, base_wdl_rows = [], [], [], []
        for row, value, base, base_value, support in zip(
            output.policy, output.wdl, output.base_policy, output.base_wdl, legal, strict=True
        ):
            if (
                len(row) != len(support)
                or len(base) != len(support)
                or len(value) != 3
                or len(base_value) != 3
            ):
                raise ValueError("epoch inference rows do not match legal support/WDL schema")
            for probabilities in (row, base):
                if (
                    any(not math.isfinite(x) or x < 0 for x in probabilities)
                    or not math.isclose(math.fsum(probabilities), 1.0, abs_tol=1e-6)
                ):
                    raise ValueError("epoch policy rows must be finite normalized probabilities")
            policy_total, base_total = math.fsum(row), math.fsum(base)
            policy_rows.append(tuple(x / policy_total for x in row))
            base_policy_rows.append(tuple(x / base_total for x in base))
            for wdl in (value, base_value):
                if (
                    any(not math.isfinite(x) or x < 0 for x in wdl)
                    or not math.isclose(math.fsum(wdl), 1.0, abs_tol=1e-6)
                ):
                    raise ValueError("epoch WDL rows must be finite normalized probabilities")
            value_total, base_value_total = math.fsum(value), math.fsum(base_value)
            wdl_rows.append(tuple(x / value_total for x in value))
            base_wdl_rows.append(tuple(x / base_value_total for x in base_value))
        if model_digest() != before:
            raise ValueError("online model changed during frozen-policy collection epoch")
        transitions = actors.step(tuple(policy_rows))
        for row, support, policy, value, base, base_value in zip(
            transitions, legal, policy_rows, wdl_rows, base_policy_rows,
            base_wdl_rows, strict=True
        ):
            board = actors.rules.inspect(row.pre)
            action = move_to_action(board, board.parse_uci(row.action.uci))
            selected = support.index(action)
            temperature = actors.config.temperature
            finite_logs = tuple(math.log(p) if p > 0 else -math.inf for p in policy)
            largest = max(finite_logs)
            scaled = tuple(math.exp((p - largest) / temperature) for p in finite_logs)
            normalizer = math.fsum(scaled)
            behavior = tuple(p / normalizer for p in scaled)
            if not math.isclose(
                behavior[selected], row.behavior_probability, rel_tol=0, abs_tol=1e-12
            ):
                raise ValueError("stored behavior distribution differs from actor sampling")
            actions.append(EpochAction(
                transition=row,
                legal_actions=support,
                policy=tuple(policy),
                behavior_policy=behavior,
                base_policy=tuple(base),
                online_pre_wdl=tuple(value),
                base_wdl=tuple(base_value),
            ))
    after = model_digest()
    if after != before:
        raise ValueError("online model changed during frozen-policy collection epoch")
    truncations = actors.close_policy_epoch()
    if actors.steps - start_step != steps:
        raise ValueError("actor collection step count differs from requested epoch")
    return PolicyEpoch(
        schema=ONLINE_POLICY_EPOCH_SCHEMA,
        requested_steps=steps,
        start_actor_step=start_step,
        end_actor_step=actors.steps,
        actions=tuple(actions),
        truncations=truncations,
        next_actor_cursor=actors.cursor(),
        next_actor_rng_state=actors.rng.getstate(),
        model_digest=before,
    )

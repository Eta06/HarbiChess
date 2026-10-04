"""Backend-neutral one-ply actors with full-history resumable cursors.

Sampling uses an explicit fixed temperature and a caller-owned Python RNG.
Cutoffs close a rollout without inventing a terminal result. An actor cursor is
not a training checkpoint: current/base/EMA/optimizer/RNG must accompany it.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass

from harbichess.chess.actions import legal_action_indices, move_to_action
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove, ChessState

ONLINE_ACTOR_SCHEMA = "online-actors-full-history-v1"


@dataclass(frozen=True, slots=True)
class OnlineActorConfig:
    games: int
    max_additional_plies: int
    claim_draw: bool
    temperature: float

    def __post_init__(self):
        if (
            type(self.games) is not int
            or self.games <= 0
            or type(self.max_additional_plies) is not int
            or self.max_additional_plies <= 0
            or type(self.claim_draw) is not bool
            or not math.isfinite(self.temperature)
            or self.temperature <= 0
        ):
            raise ValueError("invalid explicit online actor configuration")


@dataclass(frozen=True, slots=True)
class ActorOpening:
    source_id: str
    state: ChessState


@dataclass(frozen=True, slots=True)
class ActorGame:
    game_index: int
    opening_index: int
    state: ChessState


@dataclass(frozen=True, slots=True)
class ActorTransition:
    slot: int
    game_index: int
    source_id: str
    pre: ChessState
    action: ChessMove
    post: ChessState
    policy_probability: float
    behavior_probability: float
    rollout_cutoff: bool
    terminal_result: str | None
    terminal_termination: str | None


def _state_json(state):
    return {"root_fen": state.root_fen, "moves": [move.uci for move in state.moves]}


class OnlineActors:
    def __init__(
        self,
        openings: tuple[ActorOpening, ...],
        *,
        config: OnlineActorConfig,
        rng: random.Random,
        cursor: dict | None = None,
    ):
        self.config, self.rng = config, rng
        self.rules = PythonChessRules()
        if not openings or any(
            not isinstance(row.source_id, str) or not row.source_id.strip() for row in openings
        ):
            raise ValueError("online actor book needs nonempty source-identified roots")
        for row in openings:
            if (
                not self.rules.inspect(ChessState(row.state.root_fen)).is_valid()
                or not self.rules.inspect(row.state).is_valid()
                or self.rules.outcome(row.state, claim_draw=config.claim_draw) is not None
            ):
                raise ValueError("online actor opening must be valid and nonterminal")
        self.openings = openings
        serialized = [{"source_id": row.source_id, **_state_json(row.state)} for row in openings]
        self.book_sha256 = hashlib.sha256(
            json.dumps(serialized, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.steps = self.next_game = 0
        self.terminations = Counter()
        if cursor is None:
            self.games = tuple(self._new_game() for _ in range(config.games))
        else:
            self._restore(cursor)

    def _new_game(self):
        index = self.rng.randrange(len(self.openings))
        game = ActorGame(self.next_game, index, self.openings[index].state)
        self.next_game += 1
        return game

    @property
    def states(self) -> tuple[ChessState, ...]:
        return tuple(game.state for game in self.games)

    def step(self, legal_policies: tuple[tuple[float, ...], ...]) -> tuple[ActorTransition, ...]:
        """Advance every game once using sorted canonical legal-action probabilities.

        Incoming float probabilities are normalized in float64 for sampling.
        Recorded pi and mu describe these actual normalized distributions. All
        rows are checked before any RNG/cursor change. Fixed-temperature mu has
        the same nonzero support as pi; numeric underflow that removes support
        is rejected before changing the cursor. No entropy calibration
        or adaptive-temperature mechanism is implied by this version.
        """
        if len(legal_policies) != len(self.games):
            raise ValueError("online policy count must equal active game count")
        plans = []
        for game, probabilities in zip(self.games, legal_policies, strict=True):
            board = self.rules.inspect(game.state)
            indices = legal_action_indices(board)
            if len(probabilities) != len(indices) or any(
                not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities
            ):
                raise ValueError("invalid sorted legal policy probabilities")
            total = math.fsum(probabilities)
            if not math.isclose(total, 1, rel_tol=0, abs_tol=1e-6):
                raise ValueError("legal policy probabilities must sum to one")
            policy = tuple(float(value) / total for value in probabilities)
            logs = tuple(math.log(value) if value > 0 else -math.inf for value in policy)
            largest = max(logs)
            weights = tuple(math.exp((value - largest) / self.config.temperature) for value in logs)
            normalizer = math.fsum(weights)
            behavior = tuple(weight / normalizer for weight in weights)
            if any(p > 0 and mu == 0 for p, mu in zip(policy, behavior, strict=True)):
                raise ValueError("temperature sampling underflow removed policy support")
            # Ordered UCI association must follow action indices, not move-generator order.
            by_action = {
                move_to_action(board, move): ChessMove(move.uci()) for move in board.legal_moves
            }
            plans.append((policy, behavior, tuple(by_action[index] for index in indices)))
        transitions, games = [], []
        for slot, (game, plan) in enumerate(zip(self.games, plans, strict=True)):
            policy, behavior, moves = plan
            selected = self.rng.choices(range(len(moves)), weights=behavior, k=1)[0]
            post = self.rules.apply(game.state, moves[selected])
            outcome = self.rules.outcome(post, claim_draw=self.config.claim_draw)
            additional = post.ply - self.openings[game.opening_index].state.ply
            cutoff = additional >= self.config.max_additional_plies
            transitions.append(
                ActorTransition(
                    slot,
                    game.game_index,
                    self.openings[game.opening_index].source_id,
                    game.state,
                    moves[selected],
                    post,
                    policy[selected],
                    behavior[selected],
                    cutoff,
                    outcome.result.value if outcome else None,
                    outcome.termination if outcome else None,
                )
            )
            if outcome is not None or cutoff:
                self.terminations[outcome.termination if outcome else "unknown-rollout-cutoff"] += 1
                games.append(self._new_game())
            else:
                games.append(ActorGame(game.game_index, game.opening_index, post))
        self.games = tuple(games)
        self.steps += 1
        return tuple(transitions)

    def cursor(self) -> dict:
        return {
            "schema": ONLINE_ACTOR_SCHEMA,
            "config": asdict(self.config),
            "book_sha256": self.book_sha256,
            "steps": self.steps,
            "next_game": self.next_game,
            "terminations": dict(sorted(self.terminations.items())),
            "games": [
                {
                    "game_index": game.game_index,
                    "opening_index": game.opening_index,
                    **_state_json(game.state),
                }
                for game in self.games
            ],
        }

    def close_policy_epoch(self) -> tuple[dict, ...]:
        """Record every still-live game as unknown and start fresh games.

        Call only after a fixed-policy collection epoch. Active histories are
        deliberately not continued under the next policy snapshot; each game
        with at least one collected move is returned as an explicit unknown
        policy-epoch truncation. Fresh, zero-action games stay at their opening
        root for the next epoch.
        """
        closed = []
        games = list(self.games)
        for slot, game in enumerate(self.games):
            opening = self.openings[game.opening_index]
            if game.state == opening.state:
                # This game began on the final collection step and has no
                # behavior trajectory to discard; keep it fresh for next epoch.
                continue
            closed.append({
                "slot": slot,
                "game_index": game.game_index,
                "source_id": opening.source_id,
                "opening_index": game.opening_index,
                "root_fen": game.state.root_fen,
                "moves": [move.uci for move in game.state.moves],
                "termination": "policy-epoch-truncation",
            })
            self.terminations["policy-epoch-truncation"] += 1
            games[slot] = self._new_game()
        self.games = tuple(games)
        return tuple(closed)

    def _restore(self, cursor):
        if (
            cursor["schema"] != ONLINE_ACTOR_SCHEMA
            or cursor["config"] != asdict(self.config)
            or cursor["book_sha256"] != self.book_sha256
            or len(cursor["games"]) != self.config.games
        ):
            raise ValueError("online actor schema/config/book/slot count mismatch")
        self.steps, self.next_game = cursor["steps"], cursor["next_game"]
        if (
            type(self.steps) is not int
            or self.steps < 0
            or type(self.next_game) is not int
            or self.next_game < self.config.games
        ):
            raise ValueError("invalid online actor counters")
        self.terminations = Counter(cursor["terminations"])
        if (
            any(type(count) is not int or count < 0 for count in self.terminations.values())
            or sum(self.terminations.values()) != self.next_game - self.config.games
            or sum(self.terminations.values()) > self.steps * self.config.games
        ):
            raise ValueError("online closed-game counters differ")
        games = []
        for row in cursor["games"]:
            index, game_index = row["opening_index"], row["game_index"]
            if (
                type(index) is not int
                or not 0 <= index < len(self.openings)
                or type(game_index) is not int
                or not 0 <= game_index < self.next_game
            ):
                raise ValueError("invalid online opening/game index")
            state = ChessState(row["root_fen"], tuple(ChessMove(move) for move in row["moves"]))
            opening = self.openings[index].state
            if (
                state.root_fen != opening.root_fen
                or state.moves[: opening.ply] != opening.moves
                or not 0 <= state.ply - opening.ply < self.config.max_additional_plies
                or state.ply - opening.ply > self.steps
                or self.rules.outcome(state, claim_draw=self.config.claim_draw) is not None
            ):
                raise ValueError("online active game history or terminal/cap cursor differs")
            games.append(ActorGame(game_index, index, state))
        if len({game.game_index for game in games}) != len(games):
            raise ValueError("online active game IDs must be unique")
        self.games = tuple(games)

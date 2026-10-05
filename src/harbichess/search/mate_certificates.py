"""Exact, history-aware one-ply checkmate certificates for root search."""

from __future__ import annotations

import math

from harbichess.core.state import ChessMove, ChessState, TerminalResult


def immediate_mating_moves(rules, state: ChessState, *, claim_draw: bool = True):
    """Return every legal move that ends this full-history root in checkmate.

    The rules cache is read-only: work on a full-stack copy so repetition and
    claim-draw semantics are preserved. A candidate is accepted only after the
    project rules API confirms its terminal result and winner from the mover's
    perspective. Stalemates, claimable draws, and nonterminal checks are excluded.
    """
    if rules.outcome(state, claim_draw=claim_draw) is not None:
        return ()
    board = rules.inspect(state).copy(stack=True)
    mover_is_white = board.turn
    certified = []
    for move in sorted(board.legal_moves, key=lambda item: item.uci()):
        if not board.gives_check(move):
            continue
        board.push(move)
        is_mate = board.is_checkmate()
        board.pop()
        if not is_mate:
            continue
        post = rules.apply(state, ChessMove(move.uci()))
        outcome = rules.outcome(post, claim_draw=claim_draw)
        if outcome is None or outcome.result == TerminalResult.DRAW:
            raise RuntimeError("checkmate certificate lacks exact decisive rule outcome")
        if (outcome.result == TerminalResult.WHITE_WIN) != mover_is_white:
            raise RuntimeError("one-ply mate certificate has wrong mover perspective")
        certified.append(ChessMove(move.uci()))
    return tuple(certified)


def certified_policy(legal_moves, certified_moves, scores, *, epsilon=1e-12):
    """Return full-legal-support probabilities concentrated on exact mates."""
    legal = tuple(legal_moves)
    certified = tuple(m for m in certified_moves if m in legal)
    if not legal or not certified or not 0 < epsilon < 1:
        raise ValueError("certified policy requires legal moves and exact mate support")
    if len(set(certified)) != len(certified):
        raise ValueError("duplicate certified mating move")
    best = max(scores[m] for m in certified)
    winners = tuple(m for m in certified if scores[m] == best)
    weight = (1.0 - epsilon * (len(legal) - len(winners))) / len(winners)
    winner_set = set(winners)
    probabilities = tuple(weight if move in winner_set else epsilon for move in legal)
    return probabilities, min(winners, key=lambda move: move.uci)


def visited_mate_in_one_losses(rules, state: ChessState, visited_moves, *, claim_draw=True):
    """Certify losses only among searched root actions with visited children.

    A root move is certified losing only when its complete-history child is
    nonterminal and the opponent has an exact legal mate-in-one. Terminal
    children (including claimable/automatic draws) are deliberately excluded.
    """
    if rules.outcome(state, claim_draw=claim_draw) is not None:
        return ()
    root_board = rules.inspect(state)
    legal = {ChessMove(move.uci()) for move in root_board.legal_moves}
    visited = tuple(sorted(set(visited_moves), key=lambda move: move.uci))
    if any(move not in legal for move in visited):
        raise ValueError("visited losing-action candidates must be legal at the root")
    certified = []
    for move in visited:
        child = rules.apply(state, move)
        if rules.outcome(child, claim_draw=claim_draw) is not None:
            continue
        replies = immediate_mating_moves(rules, child, claim_draw=claim_draw)
        if replies:
            certified.append((move, replies))
    return tuple(certified)


def shield_visited_losses(
    legal_moves,
    policy,
    move_stats,
    certified_losses,
    *,
    selected_action,
    certified_wins=(),
    epsilon=1e-12,
):
    """Set exact visited one-ply losses to epsilon without removing legal support.

    Exact current-side wins take precedence. If every legal action has a loss
    certificate, the original policy/action is retained and no action is
    described as safe. Otherwise the searched action is replaced, if needed,
    by deterministic max-visits, then root-Q, prior, UCI among all noncertified
    actions. Nonvisited alternatives remain UNKNOWN and eligible for fallback.
    """
    legal = tuple(legal_moves)
    weights = tuple(float(value) for value in policy)
    losses = set(certified_losses)
    wins = set(certified_wins)
    if (
        not legal
        or len(weights) != len(legal)
        or any(not math.isfinite(p) or p < 0 for p in weights)
        or not math.isclose(math.fsum(weights), 1.0, abs_tol=1e-9, rel_tol=0)
        or any(move not in legal for move in losses | wins)
        or not 0 < epsilon < 1 / len(legal)
    ):
        raise ValueError("invalid visited-loss policy/support input")
    if wins:
        return weights, selected_action, "winning-certificate-precedence"
    if not losses:
        return weights, selected_action, "no-visited-loss-certificate"
    if len(losses) == len(legal):
        return weights, selected_action, "all-legal-actions-certified-loss-no-safe-action"

    safe = tuple(move for move in legal if move not in losses)
    safe_raw = [max(weights[i], epsilon) for i, move in enumerate(legal) if move in safe]
    safe_total = math.fsum(safe_raw)
    remaining = 1.0 - epsilon * len(losses)
    safe_probabilities = [remaining * value / safe_total for value in safe_raw]
    output = []
    cursor = 0
    for move in legal:
        if move in losses:
            output.append(epsilon)
        else:
            output.append(safe_probabilities[cursor])
            cursor += 1
    stats = {row.move: row for row in move_stats}
    if selected_action in losses:
        selected_action = min(
            safe,
            key=lambda move: (
                -stats[move].visits if move in stats else 0,
                -stats[move].mean_value if move in stats else 0.0,
                -stats[move].prior if move in stats else 0.0,
                move.uci,
            ),
        )
    return tuple(output), selected_action, "visited-losses-epsilon-shielded"

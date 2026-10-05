"""Exact, history-aware one-ply checkmate certificates for root search."""

from __future__ import annotations

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

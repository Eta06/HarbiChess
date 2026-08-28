"""Fixed 8x8x73 policy action encoding for chess moves."""

from __future__ import annotations

import chess

POLICY_PLANES = 73
POLICY_SIZE = 64 * POLICY_PLANES
ACTION_SCHEMA_VERSION = 1

_RAY_DIRECTIONS = (
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
)
_KNIGHT_DELTAS = (
    (-2, -1),
    (-2, 1),
    (-1, -2),
    (-1, 2),
    (1, -2),
    (1, 2),
    (2, -1),
    (2, 1),
)
_UNDERPROMOTIONS = (chess.KNIGHT, chess.BISHOP, chess.ROOK)


def _canonical_square(square: chess.Square, turn: chess.Color) -> chess.Square:
    return square if turn is chess.WHITE else chess.square_mirror(square)


def move_to_action(board: chess.Board, move: chess.Move) -> int:
    """Map a move to a side-to-move canonical policy index."""

    origin = _canonical_square(move.from_square, board.turn)
    target = _canonical_square(move.to_square, board.turn)
    delta_file = chess.square_file(target) - chess.square_file(origin)
    delta_rank = chess.square_rank(target) - chess.square_rank(origin)

    if move.promotion in _UNDERPROMOTIONS:
        if delta_rank != 1 or delta_file not in (-1, 0, 1):
            raise ValueError(f"invalid underpromotion geometry: {move.uci()}")
        piece_offset = _UNDERPROMOTIONS.index(move.promotion)
        plane = 64 + piece_offset * 3 + delta_file + 1
    elif move.promotion not in (None, chess.QUEEN):
        raise ValueError(f"unsupported promotion piece: {move.uci()}")
    elif (delta_file, delta_rank) in _KNIGHT_DELTAS:
        plane = 56 + _KNIGHT_DELTAS.index((delta_file, delta_rank))
    else:
        distance = max(abs(delta_file), abs(delta_rank))
        if distance == 0 or distance > 7:
            raise ValueError(f"invalid ray move geometry: {move.uci()}")
        unit_file = delta_file // distance
        unit_rank = delta_rank // distance
        if (
            (unit_file, unit_rank) not in _RAY_DIRECTIONS
            or unit_file * distance != delta_file
            or unit_rank * distance != delta_rank
        ):
            raise ValueError(f"invalid ray move geometry: {move.uci()}")
        plane = _RAY_DIRECTIONS.index((unit_file, unit_rank)) * 7 + distance - 1

    return origin * POLICY_PLANES + plane


def legal_action_indices(board: chess.Board) -> tuple[int, ...]:
    """Return sorted, collision-free action indices for all legal moves."""

    indices = tuple(sorted(move_to_action(board, move) for move in board.legal_moves))
    if len(indices) != len(set(indices)):
        raise RuntimeError("legal moves collided in the policy action encoding")
    return indices


def legal_action_mask(board: chess.Board) -> tuple[bool, ...]:
    """Return a dense legal-action mask aligned to the policy output."""

    mask = [False] * POLICY_SIZE
    for index in legal_action_indices(board):
        mask[index] = True
    return tuple(mask)


def action_to_legal_move(board: chess.Board, action: int) -> chess.Move:
    """Resolve an action through the current legal move set."""

    if not 0 <= action < POLICY_SIZE:
        raise ValueError(f"policy action is out of range: {action}")
    matches = [move for move in board.legal_moves if move_to_action(board, move) == action]
    if len(matches) != 1:
        raise ValueError(f"action {action} does not identify exactly one legal move")
    return matches[0]


def action_destination_square(action: int) -> chess.Square | None:
    """Return the canonical destination square encoded by an action index."""

    if not 0 <= action < POLICY_SIZE:
        raise ValueError(f"policy action is out of range: {action}")
    origin, plane = divmod(action, POLICY_PLANES)
    origin_file = chess.square_file(origin)
    origin_rank = chess.square_rank(origin)
    if plane < 56:
        file_step, rank_step = _RAY_DIRECTIONS[plane // 7]
        distance = plane % 7 + 1
        target_file = origin_file + file_step * distance
        target_rank = origin_rank + rank_step * distance
    elif plane < 64:
        file_step, rank_step = _KNIGHT_DELTAS[plane - 56]
        target_file = origin_file + file_step
        target_rank = origin_rank + rank_step
    else:
        target_file = origin_file + (plane - 64) % 3 - 1
        target_rank = origin_rank + 1
    if not 0 <= target_file < 8 or not 0 <= target_rank < 8:
        return None
    return chess.square(target_file, target_rank)

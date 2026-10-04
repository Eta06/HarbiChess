"""Versioned lossless storage of the existing canonical104 encoder tensors."""

import chess
import numpy as np

PACKED_SCHEMA = 1


def pack_board(board: chess.Board) -> tuple[np.ndarray, np.ndarray, int]:
    perspective = board.turn
    pieces = np.zeros((8, 12), dtype=np.uint64)
    historical = board.copy(stack=7)
    for step in range(8):
        if step:
            if not historical.move_stack:
                break
            historical.pop()
        for color, side in ((perspective, 0), (not perspective, 6)):
            for kind in range(1, 7):
                bits = historical.pieces_mask(kind, color)
                pieces[step, side + kind - 1] = bits if perspective else chess.flip_vertical(bits)
    repetition = 1.0 if board.is_repetition(3) else 0.5 if board.is_repetition(2) else 0.0
    metadata = np.array(
        [
            float(perspective),
            board.has_kingside_castling_rights(perspective),
            board.has_queenside_castling_rights(perspective),
            board.has_kingside_castling_rights(not perspective),
            board.has_queenside_castling_rights(not perspective),
            0,
            min(board.halfmove_clock, 100) / 100,
            repetition,
        ],
        dtype=np.float32,
    )
    ep = board.ep_square
    if ep is not None and not perspective:
        ep = chess.square_mirror(ep)
    return pieces, metadata, 64 if ep is None else ep


def decode_positions(pieces: np.ndarray, metadata: np.ndarray, ep: np.ndarray) -> np.ndarray:
    """Decode a batch into exact contiguous NHWC FP32 without framework imports."""
    count = len(pieces)
    if (
        count == 0
        or pieces.shape != (count, 8, 12)
        or pieces.dtype.kind != "u"
        or pieces.dtype.itemsize != 8
        or metadata.shape != (count, 8)
        or metadata.dtype != np.float32
        or ep.shape != (count,)
        or ep.dtype != np.uint8
        or not np.isfinite(metadata).all()
        or np.any(ep > 64)
        or not np.isin(metadata[:, :5], (0, 1)).all()
        or np.any(metadata[:, 5] != 0)
        or np.any(metadata[:, 6:] < 0)
        or np.any(metadata[:, 6:] > 1)
        or not np.isin(metadata[:, 7], (0, 0.5, 1)).all()
    ):
        raise ValueError("invalid packed canonical104 batch")
    octets = pieces.astype("<u8", copy=False).view(np.uint8).reshape(count, 8, 12, 8)
    bits = np.unpackbits(octets, axis=-1, bitorder="little")
    values = np.empty((count, 64, 104), dtype=np.float32)
    values[:, :, :96] = bits.transpose(0, 3, 1, 2).reshape(count, 64, 96)
    values[:, :, 96:] = metadata[:, None, :]
    selected = np.flatnonzero(ep != 64)
    values[selected, ep[selected], 101] = 1
    return values.reshape(count, 8, 8, 104)

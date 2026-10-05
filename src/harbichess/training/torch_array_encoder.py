"""Experimental Torch-only array encoding; standard tuple encoder stays unchanged."""
from dataclasses import dataclass

import chess
import numpy as np
import torch
from harbichess.backends.torch_backend import TorchPolicyValueBackend
from harbichess.chess.encoding import ENCODER_CHANNELS, HISTORY_STEPS, BoardEncoder


@dataclass(frozen=True, slots=True)
class _TorchArrayPosition:
    values: np.ndarray
    shape: tuple[int, ...]
    schema_version: int


class TorchArrayBoardEncoder(BoardEncoder):
    """Return immutable byte-backed float32 features for private Torch consumers.

    This experimental values representation is not a portable tuple API claim.
    No global encoder replacement, native layout change, or MLX integration.
    """

    def encode_board(self, board):
        perspective = board.turn
        historical = board.copy(stack=HISTORY_STEPS - 1)
        masks = []
        for step in range(HISTORY_STEPS):
            if step > 0:
                if not historical.move_stack:
                    break
                historical.pop()
            pieces = (historical.pawns, historical.knights, historical.bishops,
                      historical.rooks, historical.queens, historical.kings)
            for color in (perspective, not perspective):
                occupied = historical.occupied_co[color]
                masks.extend(mask & occupied for mask in pieces)
        masks.extend([0] * (96 - len(masks)))
        packed = np.asarray(masks, dtype='<u8').view(np.uint8).reshape(96, 8)
        planes = np.unpackbits(packed, axis=1, bitorder='little')
        values = np.zeros((64, ENCODER_CHANNELS), dtype=np.float32)
        order = np.arange(64) if perspective == chess.WHITE else np.arange(64) ^ 56
        values[:, :96] = planes[:, order].T
        values[:, 96] = float(perspective == chess.WHITE)
        values[:, 97] = float(board.has_kingside_castling_rights(perspective))
        values[:, 98] = float(board.has_queenside_castling_rights(perspective))
        values[:, 99] = float(board.has_kingside_castling_rights(not perspective))
        values[:, 100] = float(board.has_queenside_castling_rights(not perspective))
        if board.ep_square is not None:
            square = board.ep_square if perspective == chess.WHITE else board.ep_square ^ 56
            values[square, 101] = 1.0
        values[:, 102] = min(board.halfmove_clock, 100) / 100
        values[:, 103] = (1.0 if board.is_repetition(3)
                          else 0.5 if board.is_repetition(2) else 0.0)
        immutable = np.frombuffer(values.tobytes(), dtype=np.float32)
        return _TorchArrayPosition(immutable, (8, 8, ENCODER_CHANNELS), 1)


class TorchArrayPolicyValueBackend(TorchPolicyValueBackend):
    """Pack private array features before Torch conversion; public backend unchanged."""

    def _inputs(self, positions):
        shape = (8, 8, self.network.config.input_channels)
        if any(p.shape != shape or p.schema_version != 1 for p in positions):
            raise ValueError("encoded position shape/schema incompatible with backend")
        packed = np.asarray([p.values for p in positions], dtype=np.float32)
        return torch.tensor(packed, dtype=torch.float32, device=self.device).reshape(
            len(positions), *shape
        )

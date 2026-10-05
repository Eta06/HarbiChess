"""Read-only value-only CPU execution of the versioned sparse value head."""

import json
from pathlib import Path

import chess
import numpy as np
from safetensors import safe_open

from harbichess.core.network_config import validate_sparse_value


class NumpySparseValue:
    """Equivalent active-column sum; no policy, optimizer or terminal override."""

    def __init__(self, path: Path):
        with safe_open(path, framework="numpy") as handle:
            metadata = json.loads((handle.metadata() or {}).get("harbichess", "{}"))
            specification = metadata.get("specification", {})
            if (
                metadata.get("schema") != 1
                or metadata.get("layout") not in ("torch-oihw", "mlx-ohwi")
                or specification.get("architecture") != "pairwise"
                or specification.get("config", {}).get("input_channels") != 104
            ):
                raise ValueError("versioned canonical pairwise sparse weights required")
            spec = validate_sparse_value(specification.get("value_sparse"), 104)
            if spec is None:
                raise ValueError("sparse value specification required")
            if spec["schema"] != 1:
                raise ValueError(
                    "additive sparse value requires full inherited logits; use Torch or MLX"
                )
            channels, hidden = spec["channels"], spec["hidden"]
            shapes = {
                "feature.weight": (channels, 832),
                "feature.bias": (channels,),
                "metadata.weight": (channels, 8),
                "hidden.weight": (hidden, channels),
                "hidden.bias": (hidden,),
                "output.weight": (3, hidden),
                "output.bias": (3,),
            }
            prefix = "value_sparse_head."
            tensor_names = handle.keys()
            if {key for key in tensor_names if key.startswith(prefix)} != {
                prefix + key for key in shapes
            }:
                raise ValueError("unexpected sparse value tensors")
            weights = {}
            for name, shape in shapes.items():
                value = np.array(handle.get_tensor(prefix + name), copy=True)
                if (
                    value.dtype != np.float32 or value.shape != shape
                    or not np.isfinite(value).all()
                ):
                    raise ValueError("invalid sparse value tensor: " + name)
                value.flags.writeable = False
                weights[name] = value
        self._weights = weights

    @staticmethod
    def features(board: chess.Board) -> tuple[list[int], np.ndarray]:
        perspective = board.turn
        columns = []
        for square, piece in board.piece_map().items():
            canonical = square if perspective else chess.square_mirror(square)
            side = 0 if piece.color == perspective else 6
            columns.append(canonical * 12 + side + piece.piece_type - 1)
        if board.ep_square is not None:
            ep = board.ep_square if perspective else chess.square_mirror(board.ep_square)
            columns.append(768 + ep)
        repetition = 1.0 if board.is_repetition(3) else 0.5 if board.is_repetition(2) else 0.0
        metadata = np.array(
            [
                float(perspective),
                board.has_kingside_castling_rights(perspective),
                board.has_queenside_castling_rights(perspective),
                board.has_kingside_castling_rights(not perspective),
                board.has_queenside_castling_rights(not perspective),
                1 / 64 if board.ep_square is not None else 0,
                min(board.halfmove_clock, 100) / 100,
                repetition,
            ],
            dtype=np.float32,
        )
        return sorted(columns), metadata

    def logits(self, board: chess.Board) -> np.ndarray:
        columns, metadata = self.features(board)
        weights = self._weights
        projected = weights["feature.weight"][:, columns].sum(axis=1)
        projected += weights["feature.bias"] + weights["metadata.weight"] @ metadata
        active = np.clip(projected, 0, 1)
        hidden = np.clip(weights["hidden.weight"] @ active + weights["hidden.bias"], 0, 1)
        return weights["output.weight"] @ hidden + weights["output.bias"]

    def wdl(self, board: chess.Board) -> np.ndarray:
        logits = self.logits(board)
        masses = np.exp(logits - logits.max())
        return masses / masses.sum()

    def __call__(self, board: chess.Board) -> float:
        wdl = self.wdl(board)
        return float(wdl[0] - wdl[2])

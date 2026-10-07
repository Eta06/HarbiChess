"""Float64 sparse king-bucket residual; fixed classical prior outside model."""

import math

import torch

BUCKETS, PIECES, SQUARES, HIDDEN = 16, 12, 64, 16
VOCAB = BUCKETS * PIECES * SQUARES
MODEL_SCHEMA = "own-antisymmetric-kingbucket-nnue16-model-v1"
FEATURE_SCHEMA = "same-board-paired-mover-opposite-king2x2-piece12-square64-v1"


def feature_indices(bitboards, own_king, mover):
    """12 absolute white/black piece bitboards; no FEN/history reconstruction."""
    if (
        len(bitboards) != 12
        or type(mover) is not bool
        or type(own_king) is not int
        or not 0 <= own_king < 64
    ):
        raise ValueError("exact board piece/kings packet required")
    used = 0
    for bits in bitboards:
        if type(bits) is not int or not 0 <= bits < 2**64 or used & bits:
            raise ValueError("disjoint unsigned piece bitboards")
        used |= bits
    if (
        bitboards[5].bit_count() != 1
        or bitboards[11].bit_count() != 1
        or not bitboards[5 if mover else 11] & (1 << own_king)
    ):
        raise ValueError("both king bitboards and mover king must agree")
    oriented_king = own_king if mover else own_king ^ 56
    bucket = (oriented_king // 8 // 2) * 4 + (oriented_king % 8 // 2)
    ids = []
    # Fixed mover-relative piece order, then ascending oriented-square order.
    for relative in range(12):
        color_offset = (0 if mover else 6) if relative < 6 else (6 if mover else 0)
        bits = bitboards[color_offset + relative % 6]
        squares = []
        while bits:
            low = bits & -bits
            square = low.bit_length() - 1
            squares.append(square if mover else square ^ 56)
            bits ^= low
        ids.extend(bucket * 768 + relative * 64 + square for square in sorted(squares))
    if not 1 <= len(ids) <= 32:
        raise ValueError("legal chess piece count1..32 required")
    return ids


def board_indices(board):
    import chess

    king = board.king(board.turn)
    if king is None:
        raise ValueError("mover king missing")
    return feature_indices(
        [
            board.pieces_mask(piece, color)
            for color in [chess.WHITE, chess.BLACK]
            for piece in range(1, 7)
        ],
        king,
        board.turn,
    )


class NNUE16(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.EmbeddingBag(VOCAB, HIDDEN, mode="sum", dtype=torch.float64)
        self.head = torch.nn.Linear(HIDDEN, 1, dtype=torch.float64)
        torch.nn.init.normal_(self.embedding.weight, std=0.01)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)
        self.head.bias.register_hook(lambda gradient: torch.zeros_like(gradient))

    def residual(self, ids, offsets):
        actual_ids, opposite_ids = ids
        actual_offsets, opposite_offsets = offsets
        # Algebraically the shared head bias cancels. Exclude it from both
        # affine branches and retain an exact zero-gradient term for Adam.
        a = torch.nn.functional.linear(
            torch.tanh(self.embedding(actual_ids, actual_offsets)), self.head.weight
        ).squeeze(-1)
        o = torch.nn.functional.linear(
            torch.tanh(self.embedding(opposite_ids, opposite_offsets)), self.head.weight
        ).squeeze(-1)
        return 0.5 * (a - o) + self.head.bias.reshape(()) * 0.0

    def zero_head(self):
        return not bool(
            torch.count_nonzero(self.head.weight) or torch.count_nonzero(self.head.bias)
        )


def single_tensor_batch(rows):
    ids, offsets = [], []
    for row in rows:
        raw = row["indices"]
        if (
            not 1 <= len(raw) <= 32
            or len(set(raw)) != len(raw)
            or any(type(i) is not int or not 0 <= i < VOCAB for i in raw)
            or raw != sorted(raw)
        ):
            raise ValueError("ordered sparse feature IDs required")
        if any(not math.isfinite(row[k]) for k in ["prior_logit", "target"]):
            raise ValueError("finite own/teacher target and authoritative prior")
        if not -1 <= row["target"] <= 1:
            raise ValueError("bounded mover target")
        offsets.append(len(ids))
        ids.extend(raw)
    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(offsets, dtype=torch.long),
        torch.tensor([r["prior_logit"] for r in rows], dtype=torch.float64),
        torch.tensor([r["target"] for r in rows], dtype=torch.float64),
    )


def paired_board_indices(board):
    """Two orientations of ONE actual board; turn and rule state never mutate."""
    import chess

    bitboards = [
        board.pieces_mask(piece, color)
        for color in [chess.WHITE, chess.BLACK]
        for piece in range(1, 7)
    ]
    return (
        feature_indices(bitboards, board.king(board.turn), board.turn),
        feature_indices(bitboards, board.king(not board.turn), not board.turn),
    )


def tensor_batch(rows):
    actual = [
        dict(indices=r["indices"], prior_logit=r["prior_logit"], target=r["target"]) for r in rows
    ]
    opposite = [
        dict(indices=r["opposite_indices"], prior_logit=r["prior_logit"], target=r["target"])
        for r in rows
    ]
    a, ao, prior, target = single_tensor_batch(actual)
    o, oo, _, _ = single_tensor_batch(opposite)
    return (a, o), (ao, oo), prior, target


class TorchEvaluator:
    """Portable reference; actual parity/latency is an explicit ROOT gate."""

    def __init__(self, weights, prior):
        self.model = NNUE16()
        self.model.load_state_dict(weights, strict=True)
        self.model.eval()
        self.prior_module = prior
        self.prior = prior.ClassicalValue()
        self.zero = self.model.zero_head()
        if torch.count_nonzero(self.model.head.bias.detach().reshape(-1).view(torch.uint8)):
            raise ValueError("antisymmetric positive-zero bias invariant")

    def nonterminal(self, board):
        if self.zero:
            return self.prior.nonterminal(board)
        a, o = paired_board_indices(board)
        logit = (
            sum(
                w * x
                for w, x in zip(self.prior.weights, self.prior_module.features(board), strict=True)
            )
            / self.prior_module.SCALE
        )
        ids, offsets, _, _ = tensor_batch(
            [dict(indices=a, opposite_indices=o, prior_logit=logit, target=0.0)]
        )
        with torch.no_grad():
            residual = float(self.model.residual(ids, offsets).item())
        return math.tanh(logit + residual)

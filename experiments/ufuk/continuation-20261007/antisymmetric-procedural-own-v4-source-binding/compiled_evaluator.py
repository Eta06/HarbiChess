"""Versioned paired same-board C inference; no training/native schema relabel."""

import hashlib
import importlib.util
import math
import sys
from pathlib import Path

import torch
from native import validate_weights

SCHEMA = "antisymmetric-paired-compiled-inference-v2"
C_SOURCE_SHA = "454f8c8f2b5d177f0ff31fd520b9554636a278b10ddb2f1eedcd92a0f9ad44e0"
C_BINARY_SHA = "495505cdebd6dbc029ee314846dbac79d5cca6d445de26c804a43821808d644b"
C_BUILD_SHA = "d998140e9505cb667f91b8295f8aae0cd68091b03ddc387633e79fc3fb326e80"
PRIOR_SHA = "a99cddfc397bd9221b99a70691427fdd663488b033fb239d9f48240786b2f277"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned(ref, expected):
    path = Path(ref["path"])
    if (
        not path.is_file()
        or path.is_symlink()
        or ref["sha256"] != expected
        or sha(path) != expected
    ):
        raise ValueError("immutable exact original compiled/prior source binding")
    return path


def load_compiled(refs):
    pinned(refs["source"], C_SOURCE_SHA)
    pinned(refs["build"], C_BUILD_SHA)
    path = pinned(refs["binary"], C_BINARY_SHA)
    previous = sys.modules.get("_kingbucket16")
    if previous is not None:
        if Path(previous.__file__).resolve() != path.resolve():
            raise ValueError("compiled module namespace cannot alias unpinned binary")
        return previous
    spec = importlib.util.spec_from_file_location("_kingbucket16", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["_kingbucket16"] = module
    return module


def load_prior(ref):
    path = pinned(ref, PRIOR_SHA)
    spec = importlib.util.spec_from_file_location("antisymmetric_original_prior18", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Evaluator:
    def __init__(self, state, *, compiled_refs, prior_ref):
        validate_weights(state)
        if torch.count_nonzero(state["head.bias"].reshape(-1).view(torch.uint8)):
            raise ValueError("positive-zero canceled head-bias storage")
        self.compiled = load_compiled(compiled_refs)
        self.module = load_prior(prior_ref)
        self.prior = self.module.ClassicalValue()
        self.zero = not bool(torch.count_nonzero(state["head.weight"]))
        self.payload = (
            torch.cat(
                [
                    state["embedding.weight"].reshape(-1),
                    state["head.weight"].reshape(-1),
                    state["head.bias"],
                ]
            )
            .contiguous()
            .numpy()
            .tobytes()
        )
        if len(self.payload) != 196625 * 8 or sys.byteorder != "little":
            raise ValueError("original196625 float64 little-endian C payload geometry")
        self.binding = dict(
            schema=SCHEMA,
            c_source_sha256=C_SOURCE_SHA,
            c_binary_sha256=C_BINARY_SHA,
            c_build_sha256=C_BUILD_SHA,
            prior_sha256=PRIOR_SHA,
            payload_sha256=hashlib.sha256(self.payload).hexdigest(),
            same_board_two_orientations=True,
            orientation_is_transition=False,
            no_native_or_optimizer_conversion=True,
        )

    def paired_residual(self, board):
        bitboards = [
            board.pieces_mask(piece, color) for color in [True, False] for piece in range(1, 7)
        ]
        # Same exact bitboards for both calls; rights/history/turn never mutate.
        actual = self.compiled.forward(bitboards, board.king(board.turn), board.turn, self.payload)
        opposite = self.compiled.forward(
            bitboards, board.king(not board.turn), not board.turn, self.payload
        )
        result = 0.5 * (actual - opposite)
        if not math.isfinite(result):
            raise ValueError("finite paired compiled residual")
        return result

    def nonterminal(self, board):
        if self.zero:
            return self.prior.nonterminal(board)
        residual = self.paired_residual(board)
        logit = (
            sum(w * x for w, x in zip(self.module.PRIOR, self.module.features(board), strict=True))
            / self.module.SCALE
        )
        return math.tanh(logit + residual)

    def __call__(self, board):
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            return (
                0.0 if outcome.winner is None else (1.0 if outcome.winner == board.turn else -1.0)
            )
        return self.nonterminal(board)

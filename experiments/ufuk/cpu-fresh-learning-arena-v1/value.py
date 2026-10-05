"""Exact trained W-L value adapter; fast sparse head avoids unused policy work.

No external labels or handcrafted static scores. Network source/architecture
must be explicit; generic fallback uses the existing full production network.
"""

import torch

from harbichess.backends.torch_network import load_weights
from harbichess.search.evaluator import _softmax
from harbichess.training.torch_array_encoder import TorchArrayBoardEncoder


class NeuralValue:
    def __init__(self, weights):
        self.model = load_weights(weights).eval().requires_grad_(False)
        self.encoder = TorchArrayBoardEncoder()
        self.device = "cpu"
        sparse = self.model.specification.get("value_sparse")
        if sparse is not None:
            if sparse.get("schema") == 1:
                self.value_mode = "legacy-v1-sparse-only"
            elif (
                sparse.get("schema") == 2 and sparse.get("composition") == "additive-v1"
            ):
                self.value_mode = "schema2-core-full-base-plus-residual-once"
            else:
                raise ValueError("unsupported sparse value schema/composition")
        else:
            self.value_mode = "legacy-full-or-zero-linear"
        self.linear_only = self.model.architecture == "pairwise" and all(
            not torch.count_nonzero(p)
            for name in ("value_output", "value_tower_output", "global_value_output")
            for p in getattr(self.model, name).parameters()
        )

    def logits(self, board):
        # Keep the existing full-history encoder, including repetition metadata.
        encoded = self.encoder.encode_board(board)
        inputs = torch.from_numpy(encoded.values.copy()).reshape(1, 8, 8, 104)
        if self.value_mode == "schema2-core-full-base-plus-residual-once":
            actions = torch.zeros((1, 1), dtype=torch.long)
            # Core6fcc already adds full inherited WDL + sparse residual exactly once.
            return self.model.masked_policy_value(inputs, actions)[1]
        if self.value_mode == "legacy-v1-sparse-only":
            return self.model.value_sparse_head(inputs)
        if self.linear_only:
            value = self.model.invariant_value_linear(self.model._invariants(inputs))
            zero = torch.zeros_like(value)
            return ((zero + value) + zero) + zero
        actions = torch.zeros((1, 1), dtype=torch.long)
        return self.model.masked_policy_value(inputs, actions)[1]

    def __call__(self, board):
        with torch.inference_mode():
            win, _, loss = _softmax(tuple(self.logits(board)[0].tolist()))
            return win - loss

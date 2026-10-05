"""Full-history W/D/L anchor wrapper; all identities are CLI-bound separately."""


class FrozenWDLAnchor:
    def __init__(self, weights, value_helper):
        self.network = value_helper.NeuralValue(weights)
        from harbichess.search.evaluator import _softmax

        self.softmax = _softmax

    def __call__(self, board):
        import torch

        with torch.inference_mode():
            logits = tuple(float(value) for value in self.network.logits(board)[0].tolist())
            return self.softmax(logits)

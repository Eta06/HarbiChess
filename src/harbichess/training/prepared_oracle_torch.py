"""Explicit Torch adapter for framework-neutral immutable prepared panels."""

import torch

from harbichess.training.prepared_oracle import PreparedOracle
from harbichess.training.torch_learner import TorchTrainingBatch


class TorchPreparedPanel:
    def __init__(self, panel):
        self.panel, self.size = panel, panel.size

    def select(self, indices):
        batch = self.panel.select(indices)
        return TorchTrainingBatch(
            *(
                torch.from_numpy(getattr(batch, name))
                for name in ("inputs", "policies", "legal_masks", "wdl", "value_weights")
            )
        )


def load_prepared_panels(directory, source, **kwargs):
    cache = PreparedOracle(directory, source=source)
    training, validation, info = cache.panels(**kwargs)
    return TorchPreparedPanel(training), TorchPreparedPanel(validation), info

"""Effort heads never alter the unchanged classical scalar value."""

import json
from pathlib import Path

from support import load, sha


class MixedValue:
    def __init__(self, path, protocol):
        path = Path(path)
        if any(path.resolve() == Path(r[role]["path"]).resolve()
               for r in protocol["models"].values() for role in ["learned", "newzero"]):
            packet = json.loads(path.read_bytes())
            dep = protocol["selective_helpers"]["model.py"]
            module = load(dep["path"], dep["sha256"], "selective_arena_model")
            self.selective_weights = module.load_model(packet)
            prior = protocol["classical_value_helper"]
            value = load(prior["path"], prior["sha256"], "selective_prior_features")
            prior_sha = protocol["models"][str(protocol["match_seeds"][0])]["prior"]["sha256"]
            if sha(protocol["original_prior_path"]) != prior_sha:
                raise ValueError("unchanged scalar prior file SHA differs")
            self.value = value.load_classical(protocol["original_prior_path"]).nonterminal
            self.kind = "selective-Q-effort-unchanged-humanprior18"
        else:
            dep = protocol["original_mixed_value"]
            module = load(dep["path"], dep["sha256"], "selective_original_mixed")
            self.value = module.MixedValue(path, protocol)
            self.kind = self.value.kind
        expected = {r[role]["sha256"] for r in protocol["models"].values() for role in r}
        if sha(path) not in expected:
            raise ValueError("unregistered endpoint")

    def __call__(self, board):
        return self.value(board)

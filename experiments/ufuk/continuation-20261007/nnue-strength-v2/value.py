"""Own child and exact teacher parent share SAME compiled NNUE/scalar/search path."""

from pathlib import Path

from admission import import_nnue
from teacher_admission import module, pin


class MixedValue:
    def __init__(self, path, protocol):
        path = Path(path).resolve()
        matches = []
        for seed, roles in protocol["models"].items():
            for role, ref in roles.items():
                if path == Path(ref["path"]).resolve():
                    matches.append((int(seed), role, ref))
        if not matches:
            raise ValueError("unregistered endpoint")
        seed, role, ref = matches[0]
        pin(ref)
        if any(other_role != role for _, other_role, _ in matches):
            raise ValueError("role alias")
        if role in ["learned", "parent"]:
            _, native, evaluator, compiled = import_nnue(protocol)
            import torch

            packet = torch.load(path, map_location="cpu", weights_only=False)
            phase, steps = ("own-learning", 64) if role == "learned" else ("teacher-bootstrap", 256)
            if (
                packet.keys() != {"schema", "contract", "model"}
                or packet["schema"] != "own-kingbucket-nnue16-model-v1"
                or packet["contract"]["phase"] != phase
                or packet["contract"]["updates"] != steps
                or packet["contract"]["seed"] != seed
            ):
                raise ValueError("exact own64/teacher256 candidate types")
            native.validate_weights(packet["model"])
            prior = module(protocol["prior_helper"], "nnue_arena_original_humanprior")
            authoritative = evaluator.AuthoritativePrior(prior, prior.ClassicalValue())
            self.value = evaluator.Evaluator(
                packet["model"], prior=authoritative, compiled=compiled
            ).nonterminal
            self.kind = (
                "own64-NNUE" if role == "learned" else "teacher256-parent-NNUE-NOT-selflearning"
            )
        elif role == "e8":
            original = module(protocol["original_mixed_value"], "nnue_arena_E8")
            self.value = original.MixedValue(path, protocol)
            self.kind = self.value.kind
        else:
            raise ValueError("no humanprior or zero substituted for teacher ancestor")

    def __call__(self, board):
        return self.value(board)

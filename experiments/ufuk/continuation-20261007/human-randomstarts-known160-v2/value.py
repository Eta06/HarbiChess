"""Own child and exact current human/own parent share SAME compiled NNUE/scalar/search path."""

from pathlib import Path

from admission import child, import_nnue
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

            admitted_child, admitted_parent, _ = child(protocol, seed, native)
            packet = torch.load(path, map_location="cpu", weights_only=False)
            admitted = admitted_child if role == 'learned' else admitted_parent
            if not native.bits_equal(packet['model'], admitted):
                raise ValueError('strict admitted endpoint bytes differ')
            if protocol['target_variant'] != 'human-randomstarts-own-q-v2':
                raise ValueError('separate teacher-free human own pipeline only')
            c = packet['contract']
            expected = protocol['children'][str(seed)] if role == 'learned' else None
            if expected is not None:
                from teacher_admission import read

                contract = read(expected['contract'])
            else:
                from teacher_admission import read

                parent_seal = read(protocol['parents'][str(seed)]['admission_seal'])
                contract = read(parent_seal['parent_contract'])
            if (set(packet) != {'schema', 'contract', 'model'}
                    or packet['schema'] != 'own-kingbucket-nnue16-model-v1'
                    or c != contract or c['seed'] != seed):
                raise ValueError('actual full typed human-zero/current-own/child contract')
            native.validate_weights(packet["model"])
            prior = module(protocol["prior_helper"], "nnue_arena_original_humanprior")
            authoritative = evaluator.AuthoritativePrior(prior, prior.ClassicalValue())
            self.value = evaluator.Evaluator(
                packet["model"], prior=authoritative, compiled=compiled
            ).nonterminal
            self.kind = ('human-prior-own64-generation-' + str(c['generation'])
                         if role == 'learned' else
                         ('human-prior-zero0-NOT-selflearning' if c['generation'] == 0 else
                          'exact-own64-parent-generation-' + str(c['generation'])))
        elif role == "e8":
            original = module(protocol["original_mixed_value"], "nnue_arena_E8")
            self.value = original.MixedValue(path, protocol)
            self.kind = self.value.kind
        else:
            raise ValueError("no humanprior or zero substituted for current own parent")

    def __call__(self, board):
        return self.value(board)

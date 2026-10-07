"""Strict actual forensic own64 and literalzero H0; original E8 path untouched."""

from pathlib import Path

from admission import child
from runtime import import_runtime
from teacher_admission import module, pin, read


def classify(packet, protocol, seed, role):
    if protocol.get("runtime_revision") != "forensic-h0-known160-runtime-v6":
        raise ValueError("new v6 qualification required")
    if (
        packet.keys() != {"schema", "contract", "model"}
        or packet["schema"] != "own-kingbucket-nnue16-model-v1"
    ):
        raise ValueError("exact existing model container, no model family relabel")
    if role == "learned":
        c = read(protocol["children"][str(seed)]["contract"])
        if (
            c["contract_schema"] != "human-prior-own-forensic-training-contract-v4"
            or c["native_schema"] != "human-prior-own-nnue16-native-cpu-forensic-v4"
            or c["phase"] != "own-learning"
            or c["updates"] != 64
            or c["teacher_labels_used_in_own_phase"] is not False
        ):
            raise ValueError("actual forensic own64 typed contract")
        kind = "forensic-v4-own64-NNUE-with-strict-native-baseline"
    elif role == "parent":
        c = read(protocol["parents"][str(seed)]["contract"])
        if (
            c["phase"] != "human-prior-zero-residual-init-v1"
            or c["updates"] != 0
            or c["generation"] != 0
            or c["lineage_origin"]["teacher_labels_used"] is not False
        ):
            raise ValueError("actual literalzero H0 parent, never teacher256")
        kind = "literalzero-H0-native0-NOT-teacher256"
    else:
        raise ValueError("exact native learned/parent endpoint role")
    if packet["contract"] != c or c["seed"] != seed:
        raise ValueError("ALL admitted candidate contract/source/data fields must match")
    return kind


class MixedValue:
    def __init__(self, path, protocol):
        path = Path(path).resolve()
        matches = [
            (int(seed), role, ref)
            for seed, roles in protocol["models"].items()
            for role, ref in roles.items()
            if path == Path(ref["path"]).resolve()
        ]
        if not matches or len({(role, ref["sha256"]) for _, role, ref in matches}) != 1:
            raise ValueError("exact source-pinned endpoint, no role alias")
        if len(matches) > 1 and matches[0][1] != "e8":
            raise ValueError("own endpoint cannot impersonate both seeds")
        seed, role, endpoint = matches[0]
        pin(endpoint)
        if role in ("learned", "parent"):
            _, native, evaluator, compiled = import_runtime(protocol, seed)
            # This reopens all6+2 through the typed forensic reader and exact H0
            # bridge, replays data and checks ALL baseline/model/Adam/RNG bits.
            admitted_child, admitted_parent, _ = child(protocol, seed, native)
            import torch

            packet = torch.load(path, map_location="cpu", weights_only=False)
            admitted = admitted_child if role == "learned" else admitted_parent
            if not native.bits_equal(packet["model"], admitted):
                raise ValueError("exact fullnative admitted candidate storage")
            self.kind = classify(packet, protocol, seed, role)
            native.validate_weights(packet["model"])
            prior = module(protocol["prior_helper"], "forensic_v6_original_humanprior")
            authoritative = evaluator.AuthoritativePrior(prior, prior.ClassicalValue())
            self.value = evaluator.Evaluator(
                packet["model"], prior=authoritative, compiled=compiled
            ).nonterminal
        elif role == "e8":
            if (
                endpoint["sha256"]
                != "e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03"
            ):
                raise ValueError("unchanged E8 bytes")
            original = module(protocol["original_mixed_value"], "forensic_v6_E8")
            self.value = original.MixedValue(path, protocol)
            self.kind = self.value.kind
        else:
            raise ValueError("exact own64/H0/E8 roles only")

    def __call__(self, board):
        return self.value(board)

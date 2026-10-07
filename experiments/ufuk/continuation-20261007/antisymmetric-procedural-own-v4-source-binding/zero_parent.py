"""Explicit original native-v3 zero admission; model-only init into NEW native-v4."""

from parent_bridge import original_admit


def admit(binding, seed):
    return original_admit(binding, seed)


def admitted_weights(binding, seed):
    state, _ = admit(binding, seed)
    return state["model"]

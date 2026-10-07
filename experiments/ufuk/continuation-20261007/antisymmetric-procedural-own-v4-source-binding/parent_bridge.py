"""Read ONLY original v3 zero/native0/profile closure; no old state relabel."""

from pathlib import Path

from support import module, namespace, ref

ORIGINAL = Path(__file__).resolve().parent.parent / "antisymmetric-procedural-own-v3-production"
INVENTORY = "dbf7e13d5b406ff30d8d4c13c6ea33d9ebfdaf6abc14d18d46a69f98d8a3f8a7"


def original_admit(binding, seed):
    inventory = ref(ORIGINAL / "source-inventory.json")
    if inventory["sha256"] != INVENTORY:
        raise ValueError("exact unchanged original v3 paired literalzero source closure")
    with namespace(ORIGINAL, [inventory]):
        original = module(ref(ORIGINAL / "zero_parent.py"), "antisym_v4_original_v3_zero")
        return original.admit(binding, seed)


def profile_helper():
    value = ref(ORIGINAL / "qualify_profile.py")
    if value["sha256"] != "cd2a9fc46a9230f9e3ad0e8579040a65451b484e0c444bea187720e7e406ef7c":
        raise ValueError("original actual zero24 helper unchanged")
    return value

"""Typed forensic-v4 native is the runtime reader; original C/evaluator math is unchanged."""

from admission import SEEDS, td_modules
from teacher_admission import module, read


def import_runtime(q, seed):
    if q.get("runtime_revision") != "forensic-h0-known160-runtime-v6":
        raise ValueError("new explicit v6 inference binding, not relabeled v5 qualification")
    if q["target_variant"] != "forensic-own-v5-h0" or seed not in SEEDS:
        raise ValueError("actual forensic-v4 own64 / literalzero0 endpoints only")
    info = q["children"][str(seed)]
    c = read(info["contract"])
    if (
        c["contract_schema"] != "human-prior-own-forensic-training-contract-v4"
        or c["native_schema"] != "human-prior-own-nnue16-native-cpu-forensic-v4"
        or c["phase"] != "own-learning"
        or c["updates"] != 64
        or c["seed"] != seed
    ):
        raise ValueError("typed actual forensic source/native/phase/seed")
    for other in SEEDS:
        if q["children"][str(other)]["variant_helpers"] != info["variant_helpers"]:
            raise ValueError("both actual native/model runtime sources must be identical")
    with td_modules(info, c) as loaded:
        # Evaluator captures validate_weights from THIS strict forensic native,
        # not old teacher/native.py. All module globals remain original objects
        # after the source-scoped sys.modules context is restored.
        evaluator = module(q["nnue_helpers"]["evaluator.py"], "forensic_v6_typed_evaluator")
        compiled = module(q["binary"], "_kingbucket16")
        return loaded["model"], loaded["native"], evaluator, compiled

"""Prospectively frozen same-seed9-proof to16384 collection transfer only."""

ALLOWED_FIELDS = frozenset(
    {"max_actions", "epoch_id", "original_first_epoch", "original_deadline_epoch"}
)


def validate_transfer(proof, production, transfer, qualification, actual_proof_file_sha):
    if (
        transfer["proof_config_sha256"] != actual_proof_file_sha
        or qualification["config_sha256"] != actual_proof_file_sha
    ):
        raise ValueError("actual proof config input SHA differs")
    if set(proof) != set(production):
        raise ValueError("extra/missing config field differs")
    differing = {k for k in proof if proof[k] != production[k]}
    declared = transfer["allowed_difference_values"]
    if not differing <= ALLOWED_FIELDS or set(declared) != differing:
        raise ValueError("undeclared/unauthorized config difference")
    for field in differing:
        if declared[field] != {"proof": proof[field], "production": production[field]}:
            raise ValueError("literal declared config difference differs")
    if (
        proof["seed"] != production["seed"]
        or proof["max_actions"] != 9
        or production["max_actions"] != 16384
    ):
        raise ValueError("same seed and exact9-to16384 required")
    if (
        proof["original_first_epoch"] != qualification["first"]
        or proof["original_deadline_epoch"] != qualification["deadline"]
    ):
        raise ValueError("qualification observedclock differs")
    if (
        qualification["deadline"] != qualification["first"] + 600
        or not qualification["first"]
        <= qualification["finished_epoch"]
        <= qualification["deadline"]
        <= 1791273600.0
    ):
        raise ValueError("original actual600/hard08 qualification clock differs")
    return True

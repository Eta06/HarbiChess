"""ROOT-only actual trained C80 parity on fixed first24 TRAIN histories perseed."""

import gzip
import json
import math
import sys
from pathlib import Path

from value import MixedValue, load, sha


def verify(protocol, guard):
    directory = Path(protocol["coarse_directory"])
    sys.path.insert(0, str(directory))
    from coarse80 import features_from_board, residual_python
    from dataset80 import replay_legal_position

    classical = protocol["classical_value_helper"]
    module = load(classical["path"], classical["sha256"])
    fast_dep = protocol["fast_reader"]
    fast = load(fast_dep["path"], fast_dep["sha256"])
    records = []
    for seed in protocol["match_seeds"]:
        guard()
        binding = protocol["native_phase_bindings"]["fit"]["registration"]
        reg = json.loads(Path(binding["path"]).read_bytes())
        labels = Path(reg["labels"][str(seed)])
        if sha(labels) != reg["input_pins"][str(labels)]:
            raise ValueError("immutable TRAIN labels changed")
        payload = json.loads(gzip.decompress(labels.read_bytes()))
        prior = module.ClassicalValue()
        zero = fast.FastEvaluator80(
            prior,
            [0.0] * 80,
            classical_features=module.features,
            prior_scale=module.SCALE,
            binary=protocol["compiled_binary"]["path"],
        )
        info = protocol["models"][str(seed)]["learned"]
        candidate = json.loads(Path(info["path"]).read_bytes())
        evaluator = MixedValue(info["path"], protocol)
        seen = set()
        count = 0
        maximum = 0.0
        for ordinal, row in enumerate(payload["roots"]):
            guard()
            if ordinal != row["ordinal"]:
                raise ValueError("fixed chronological TRAIN order changed")
            if row["history_sha256"] in seen:
                continue
            seen.add(row["history_sha256"])
            h = row["history"]
            import hashlib

            digest = hashlib.sha256(
                (h["root_fen"] + "\n" + " ".join(h["prefix_uci"])).encode()
            ).hexdigest()
            if digest != row["history_sha256"]:
                raise ValueError("full history binding differs")
            board = replay_legal_position(h["root_fen"], h["prefix_uci"])
            if board.outcome(claim_draw=True) is not None:
                raise ValueError("actual known nonterminal TRAIN roots required")
            if zero.nonterminal(board).hex() != prior.nonterminal(board).hex():
                raise ValueError("zero does not reproduce exact frozen human prior")
            phi, _ = features_from_board(board)
            raw = module.features(board)
            prior_logit = sum(w * x for w, x in zip(prior.weights, raw, strict=True)) / module.SCALE
            expected = math.tanh(prior_logit + residual_python(candidate["parameters"], phi))
            actual = evaluator(board)
            error = abs(expected - actual)
            if not math.isfinite(actual) or error > 1e-12:
                raise ValueError("actual trained Python/C value parity exceeds1e-12")
            maximum = max(maximum, error)
            count += 1
            if count == 24:
                break
        if count != 24:
            raise ValueError("fixed TRAIN24 roots eachseed unavailable")
        records.append(
            dict(
                seed=seed,
                zero_hex_roots=count,
                trained_parity_roots=count,
                maximum_absolute_error=maximum,
                candidate_sha256=info["sha256"],
                labels_sha256=sha(labels),
            )
        )
    return dict(
        status="PASS-TRAIN48-zeroHEX-and-trained-C80-Python-parity",
        rows=records,
        tolerance=1e-12,
        selection="FIRST24-distinct-fullhistory-TRAINroots-perseed",
    )

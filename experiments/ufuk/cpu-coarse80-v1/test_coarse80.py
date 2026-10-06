from __future__ import annotations

import copy
import math
import subprocess
from pathlib import Path

import pytest
from coarse80 import PARAMS, combine_value, features_from_masks, project_zero_mean, residual_python
from compiled80 import Compiled80
from learner80 import Learner, decode_native, encode_native
from proof_synthetic import fixture, run


def test_feature_mapping_color_flip_and_piece_type_partition():
    # A white pawn at a2 and black pawn at a7 occupy the same forward-oriented bin.
    black_pawn = 1 << 48
    white_pawn = 1 << 8
    masks = [black_pawn, 0, 0, 0, 0, white_pawn, 0, 0, 0, 0]
    white_to_move = features_from_masks(masks, True)
    black_to_move = features_from_masks(masks, False)
    assert sum(white_to_move[:16]) == 0.0
    assert sum(black_to_move[:16]) == 0.0
    assert white_to_move[:16] == tuple(-v for v in black_to_move[:16])
    assert sum(white_to_move[16:]) == 0.0
    assert all(math.isfinite(v) for v in white_to_move)


def test_zero_mean_and_exact_zero_passthrough():
    raw = [float((i % 9) - 4) for i in range(PARAMS)]
    theta = project_zero_mean(raw)
    for piece in range(5):
        group = theta[piece * 16 : (piece + 1) * 16]
        assert sum(group[:15]) + group[15] == 0.0
    prior = float.fromhex("0x1.23456789abcdep-4")
    assert combine_value(prior, 123.5, 0.0).hex() == prior.hex()
    assert combine_value(prior, 0.25, 0.5) == math.tanh(0.75)
    assert residual_python([0.0] * 80, [0.0] * 80) == 0.0


def test_native_rejects_changed_source_and_candidate_contract():
    groups, contract = fixture()
    learner = Learner(contract["seed"], contract)
    learner.advance(groups, 1)
    state = learner.native()
    blob = encode_native(state)
    assert encode_native(decode_native(blob, contract)) == blob
    bad = copy.deepcopy(state)
    bad["contract"]["source_commit"] = "9" * 40
    with pytest.raises(ValueError):
        Learner(contract["seed"], contract, state=bad)
    bad = copy.deepcopy(state)
    bad["candidate"]["parameters"][0] = 0.125
    with pytest.raises(ValueError):
        Learner(contract["seed"], contract, state=bad)


def test_whole_pause_fresh_process_resume_and_six_loads():
    receipt = run()
    assert receipt["synthetic_only"] is True
    assert receipt["whole_resume_native_byte_equal"] is True
    assert receipt["six_strict_fresh_loads"] is True
    assert receipt["production_labels_loaded"] is False


def test_bad_feature_shapes_and_nan_rejected():
    with pytest.raises(ValueError):
        features_from_masks([0] * 9, True)
    with pytest.raises(ValueError):
        residual_python([0.0] * 80, [0.0] * 79)
    with pytest.raises(ValueError):
        residual_python([float("nan")] + [0.0] * 79, [0.0] * 80)


def test_compiled_kernel_matches_synthetic_feature_sum(tmp_path):
    root = Path(__file__).resolve().parent
    output = tmp_path / "libcoarse80.so"
    subprocess.run(
        [
            "cc",
            "-O3",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-fPIC",
            "-shared",
            "-std=c11",
            str(root / "forward80.c"),
            "-o",
            str(output),
        ],
        check=True,
        timeout=20,
    )
    compiled = Compiled80(output)
    theta = project_zero_mean([((i * 13) % 23 - 11) * 0.001 for i in range(80)])
    masks = [
        1 << 8,
        1 << 18,
        1 << 27,
        1 << 35,
        1 << 44,
        1 << 48,
        1 << 42,
        1 << 37,
        1 << 26,
        1 << 19,
    ]
    for mover in (False, True):
        x = features_from_masks(masks, mover)
        assert abs(compiled(theta, masks, mover) - residual_python(theta, x)) < 1e-15

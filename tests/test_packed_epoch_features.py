import random
from types import SimpleNamespace

import numpy as np
import torch
from test_fullgame_epoch_features import make_row

from harbichess.chess.encoding import BoardEncoder
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove
from harbichess.training.packed_epoch_features import PackedEpochFeatures
from harbichess.training.torch_fullgame_ppo import _tensor_batch, compile_epoch_features


def test_arbitrary_float32_bits_signed_zero_nan_and_fraction_roundtrip():
    bits = np.array(
        [0, 0x80000000, 0x3F800000, 0x3EAAAAAB, 0x7FC00019, 0x7F800000, 0xFF800000, 1, 0xFFFFFFFF],
        dtype=np.uint32,
    )
    features = PackedEpochFeatures.encode(bits.view(np.float32))
    assert features.decode().view(np.uint32).tobytes() == bits.tobytes()
    assert not features.binary.flags.writeable and not features.exception_bits.flags.writeable
    assert not features.decode().flags.writeable


def test_real_history_clock_fraction_and_every_batch_tensor_identical():
    rules = PythonChessRules()
    state = rules.initial_state()
    states = [state]
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6"):
        state = rules.apply(state, ChessMove(uci))
        states.append(state)
    rows = tuple(make_row(rules, s) for s in states + states[::-1])
    encoder = BoardEncoder(rules)
    before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    dense = compile_epoch_features(rows, encoder)
    packed = compile_epoch_features(rows, encoder, compact=True)
    assert sum(x[0].nbytes for x in packed.values()) < sum(x[0].nbytes for x in dense.values()) / 8
    assert len(dense) == len(packed) == len(states)
    fake = SimpleNamespace(config=SimpleNamespace(input_channels=104))
    for expected, actual in zip(
        _tensor_batch(fake, encoder, rules, rows, "cpu", dense),
        _tensor_batch(fake, encoder, rules, rows, "cpu", packed),
        strict=True,
    ):
        assert expected.dtype == actual.dtype and expected.shape == actual.shape
        assert (
            expected.contiguous().view(torch.uint8).numpy().tobytes()
            == actual.contiguous().view(torch.uint8).numpy().tobytes()
        )
    assert random.getstate() == before[0]
    assert np.random.get_state()[1].tobytes() == before[1][1].tobytes()
    assert torch.equal(torch.get_rng_state(), before[2])


def test_dense_nonbinary_fallback_remains_exact_and_readonly():
    values = np.arange(104 * 64, dtype=np.float32) / 123.0
    encoder = SimpleNamespace(encode=lambda _: SimpleNamespace(values=values))
    row = SimpleNamespace(transition=SimpleNamespace(pre="state"), legal_actions=(7,))
    compiled = compile_epoch_features((row,), encoder, compact=True)["state"][0]
    assert isinstance(compiled, np.ndarray)
    assert compiled.tobytes() == values.tobytes() and not compiled.flags.writeable

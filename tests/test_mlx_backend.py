from concurrent.futures import ThreadPoolExecutor

import pytest

from harbichess.chess.encoding import BoardEncoder
from harbichess.chess.rules import PythonChessRules
from harbichess.core.backend import EncodedPosition, PolicyValueBackend

mx = pytest.importorskip("mlx.core")
backend_module = pytest.importorskip("harbichess.backends.mlx_backend")
network_module = pytest.importorskip("harbichess.backends.mlx_network")
MLXPolicyValueBackend = backend_module.MLXPolicyValueBackend
HarbiChessNetwork = network_module.HarbiChessNetwork
NetworkConfig = network_module.NetworkConfig


def test_mlx_backend_implements_batch_contract() -> None:
    rules = PythonChessRules()
    position = BoardEncoder(rules).encode(rules.initial_state())
    network = HarbiChessNetwork(NetworkConfig(trunk_channels=8, residual_blocks=1))
    backend = MLXPolicyValueBackend(network, compiled=False, dtype=mx.float32)

    outputs = backend.evaluate([position, position])

    assert isinstance(backend, PolicyValueBackend)
    assert backend.capabilities.name == "mlx"
    assert len(outputs) == 2
    assert len(outputs[0].policy_logits) == 4_672
    assert len(outputs[0].wdl_logits) == 3
    assert backend.evaluate([]) == []


def test_mlx_backend_gathers_only_requested_policy_logits() -> None:
    rules = PythonChessRules()
    position = BoardEncoder(rules).encode(rules.initial_state())
    backend = MLXPolicyValueBackend(
        HarbiChessNetwork(NetworkConfig(trunk_channels=8, residual_blocks=1)),
        compiled=False,
        dtype=mx.float32,
    )
    actions = ((0, 17, 4_671),)

    full = backend.evaluate([position])[0]
    masked = backend.evaluate_masked([position], actions)[0]

    assert masked.policy_logits == pytest.approx(
        tuple(full.policy_logits[index] for index in actions[0])
    )
    assert masked.wdl_logits == pytest.approx(full.wdl_logits)
    with pytest.raises(ValueError, match="one legal action"):
        backend.evaluate_masked([position], ())


def test_mlx_backend_rejects_mixed_position_shapes() -> None:
    backend = MLXPolicyValueBackend(
        HarbiChessNetwork(NetworkConfig(trunk_channels=8, residual_blocks=1)),
        compiled=False,
    )
    first = EncodedPosition((0.0,) * 64, (8, 8, 1), 1)
    second = EncodedPosition((0.0,) * 128, (8, 8, 2), 1)

    with pytest.raises(ValueError, match="share shape"):
        backend.evaluate([first, second])


def test_compiled_mlx_backend_runs_on_inference_worker_thread() -> None:
    rules = PythonChessRules()
    position = BoardEncoder(rules).encode(rules.initial_state())
    backend = MLXPolicyValueBackend(
        HarbiChessNetwork(NetworkConfig(trunk_channels=8, residual_blocks=1)),
        compiled=True,
    )

    with ThreadPoolExecutor(max_workers=1) as worker:
        first = worker.submit(backend.evaluate, [position]).result(timeout=5)
        second = worker.submit(backend.evaluate, [position]).result(timeout=5)

    assert len(first[0].policy_logits) == 4_672
    assert len(second[0].wdl_logits) == 3


def test_fixed_mlx_batch_padding_preserves_outputs_across_request_sizes() -> None:
    rules = PythonChessRules()
    encoder = BoardEncoder(rules)
    first = encoder.encode(rules.initial_state())
    second_state = rules.apply(rules.initial_state(), rules.legal_moves(rules.initial_state())[0])
    second = encoder.encode(second_state)
    backend = MLXPolicyValueBackend(
        HarbiChessNetwork(NetworkConfig(trunk_channels=8, residual_blocks=1)),
        compiled=False,
        dtype=mx.float32,
        fixed_batch_size=4,
    )

    alone = backend.evaluate([first])[0]
    together = backend.evaluate([first, second])[0]

    assert alone == together
    with pytest.raises(ValueError, match="exceeds"):
        backend.evaluate([first] * 5)
    with pytest.raises(ValueError, match="positive"):
        MLXPolicyValueBackend(fixed_batch_size=0)

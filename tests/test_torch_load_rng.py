import pytest
import torch

from harbichess.backends.torch_network import TorchChessNetwork, load_weights, save_weights
from harbichess.core.network_config import NetworkConfig


@pytest.mark.parametrize("architecture", ["base", "invariant", "decoupled", "plastic", "pairwise"])
def test_loading_complete_weights_preserves_sampling_rng(tmp_path, architecture):
    torch.set_num_threads(1)
    network = TorchChessNetwork(
        NetworkConfig(trunk_channels=2, residual_blocks=1, value_hidden=4),
        architecture=architecture,
    )
    path = tmp_path / "weights.safetensors"
    save_weights(path, network)
    torch.manual_seed(371)
    before = torch.get_rng_state().clone()
    restored = load_weights(path)
    assert torch.equal(torch.get_rng_state(), before)
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in network.state_dict().items())

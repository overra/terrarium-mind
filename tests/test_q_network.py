import torch

from terrarium.organism.q_network import QNetwork


def test_q_network_output_shape() -> None:
    net = QNetwork(input_dim=10, hidden_dim=8, num_actions=5)
    x = torch.randn(2, 10)
    out = net(x)
    assert out.shape == (2, 5)

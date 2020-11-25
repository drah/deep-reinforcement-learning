import torch
from torch import Tensor
from torch import nn


class SimpleFC(nn.Module):
    def __init__(self, in_size: int, out_size: int, **kwargs):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.fc1_size = 32
        self.fc2_size = 32
        self.fc1 = nn.Linear(self.in_size, self.fc1_size)
        self.fc2 = nn.Linear(self.fc1_size, self.fc2_size)
        self.fc_out = nn.Linear(self.fc2_size, self.out_size)

    def forward(self, x) -> Tensor:
        net = self.fc1(x)
        net = nn.functional.relu(net)
        net = self.fc2(net)
        net = nn.functional.relu(net)
        net = self.fc_out(net)
        return net

    def act(self, states) -> Tensor:
        net = self.forward(states)
        net = nn.functional.tanh(net)
        return net

    def clone(self) -> 'SimpleFC':
        net = SimpleFC(self.in_size, self.out_size)
        net.load_state_dict(self.state_dict())
        return net

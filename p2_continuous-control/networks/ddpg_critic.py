import torch
from torch import Tensor
from torch import nn
import numpy as np

from .utils import make_tensor
from .network import Critic


class DDPGCritic(Critic):
    def __init__(self, input_shapes: [[int]], output_shapes: [[int]], **kwargs):
        super().__init__(input_shapes, output_shapes, **kwargs)
        assert len(input_shapes) == 2 and len(
            output_shapes) == 1, 'Only 2 input and 1 output is supported.'
        assert len(input_shapes[0]) == 2 and len(
            output_shapes[0]) == 2, 'Rank of input and output should be 2.'
        in_size_1 = input_shapes[0][1]
        in_size_2 = input_shapes[1][1]
        out_size = output_shapes[0][1]
        self.fc1_size = 400
        self.fc2_size = 300
        self.bn_in = nn.BatchNorm1d(in_size_1, momentum=0.001)
        self.fc1 = nn.Linear(in_size_1, self.fc1_size)
        self.bn1 = nn.BatchNorm1d(self.fc1_size, momentum=0.001)
        self.fc2 = nn.Linear(self.fc1_size + in_size_2, self.fc2_size)
        self.bn2 = nn.BatchNorm1d(self.fc2_size, momentum=0.001)
        self.fc_out = nn.Linear(self.fc2_size, out_size)
        nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc_out.weight, a=1, mode='fan_in')
        # nn.init.uniform_(self.fc_out.weight, a=-3e-3, b=3e-3)

    def forward(self, states, actions) -> Tensor:
        net = self.bn_in(states)
        net = self.fc1(net)
        net = self.bn1(net)
        net = nn.functional.relu(net)
        net = torch.cat((net, actions), 1)
        net = self.fc2(net)
        net = self.bn2(net)
        net = nn.functional.relu(net)
        net = self.fc_out(net)
        return net

    def score(self, states, actions) -> Tensor:
        states = make_tensor(states)
        actions = make_tensor(actions)
        net = self.forward(states, actions)
        return net

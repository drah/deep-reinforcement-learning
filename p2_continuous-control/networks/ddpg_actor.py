import torch
from torch import Tensor
from torch import nn
import numpy as np

from .network import Actor
from .utils import make_tensor


class DDPGActor(Actor):
    def __init__(self, input_shapes: [[int]], output_shapes: [[int]], **kwargs):
        super().__init__(input_shapes, output_shapes, **kwargs)
        assert len(input_shapes) == 1 and len(
            output_shapes) == 1, 'Only 1 input and 1 output is supported.'
        assert len(input_shapes[0]) == 2 and len(
            output_shapes[0]) == 2, 'Rank of input and output should be 2.'
        in_size = input_shapes[0][1]
        out_size = output_shapes[0][1]
        self.fc1_size = 32
        self.fc2_size = 32
        self.fc1 = nn.Linear(in_size, self.fc1_size)
        self.fc2 = nn.Linear(self.fc1_size, self.fc2_size)
        self.fc_out = nn.Linear(self.fc2_size, out_size)
        nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.uniform_(self.fc_out.weight, a=-3e-3, b=3e-3)

    def forward(self, x) -> Tensor:
        net = self.fc1(x)
        net = nn.functional.relu(net)
        net = self.fc2(net)
        net = nn.functional.relu(net)
        net = self.fc_out(net)
        return net

    def act(self, states) -> Tensor:
        states = make_tensor(states)
        net = self.forward(states)
        net = nn.functional.tanh(net)
        return net

import torch
from torch import Tensor
from torch import nn
import numpy as np

class SimpleFC(nn.Module):
    def __init__(self, input_shapes: [[int]], output_shapes: [[int]], **kwargs):
        super().__init__()
        assert len(input_shapes) == 1 and len(output_shapes) == 1, 'Only 1 input and 1 output is supported.'
        assert len(input_shapes[0]) == 2 and len(output_shapes[0]) == 2, 'Rank of input and output should be 2.'
        self.in_size = input_shapes[0][1]
        self.out_size = output_shapes[0][1]
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
        if isinstance(states, np.ndarray) and states.dtype is np.float32:
            states = torch.from_numpy(states)
        elif not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=torch.float32)
        net = self.forward(states)
        net = nn.functional.tanh(net)
        return net

    def clone(self) -> 'SimpleFC':
        net = SimpleFC(self.in_size, self.out_size)
        net.load_state_dict(self.state_dict())
        return net

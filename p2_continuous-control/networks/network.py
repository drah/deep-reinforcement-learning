from torch import nn
from torch import save
from torch import load

class Network(nn.Module):
    def __init__(self, input_shapes: [[int]], output_shapes: [[int]]):
        raise NotImplementedError("Abstract Class")

    def forward(self, x):
        raise NotImplementedError("Abstract Method")

    def act(self, x):
        raise NotImplementedError("Abstract Method")

    def clone(self) -> 'Network':
        raise NotImplementedError("Abstract Method")

def save_net(net: nn.Module, save_path: str):
    save(net.state_dict(), save_path)

def load_net(net: nn.Module, load_path: str):
    state_dict = load(load_path)
    net.load_state_dict(state_dict)

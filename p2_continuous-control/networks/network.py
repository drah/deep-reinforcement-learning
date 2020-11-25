import torch
from torch import nn
from torch import save
from torch import load

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Network(nn.Module):
    def __init__(self):
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

def get_net(net_name, **kwargs) -> nn.Module:
    net = eval(net_name)(**kwargs)
    net.to(device)
    return net

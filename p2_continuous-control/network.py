from torch import nn
from torch import save
from torch import load

class SimpleFC(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        pass

    def act(self, states):
        pass

def save_net(net: nn.Module, save_path: str):
    save(net.state_dict(), save_path)

def load_net(net: nn.Module, load_path: str):
    state_dict = load(load_path)
    net.load_state_dict(state_dict)

def get_net(net_name, **kwargs):
    return eval(net_name)(**kwargs)

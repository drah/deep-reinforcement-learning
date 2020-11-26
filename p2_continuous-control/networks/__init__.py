import torch
from .network import save_net
from .network import load_net

from .network import Network
from .simple_fc import SimpleFC

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_net(net_name, **kwargs) -> torch.nn.Module:
    net = eval(net_name)(**kwargs)
    net.to(device)
    return net

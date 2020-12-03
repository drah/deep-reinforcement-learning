import torch

from global_constant import k_device

from .network import Network
from .network import Actor
from .network import Critic
from .ddpg_actor import DDPGActor
from .ddpg_critic import DDPGCritic


def get_net(net_name, **kwargs) -> Network:
    net = eval(net_name)(**kwargs)
    net.to(k_device)
    if 'load_path' in kwargs and kwargs['load_path']:
        net.load(kwargs['load_path'])
    return net


def get_actor(net_name, **kwargs) -> Actor:
    return get_net(net_name, **kwargs)


def get_critic(net_name, **kwargs) -> Critic:
    return get_net(net_name, **kwargs)

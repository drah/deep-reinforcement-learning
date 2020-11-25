import torch
from torch import nn
from ..networks import Network
from ..memory import Memory

class DDPG:
    def __init__(self, actor: Network, critic: Network, replay_buffer: Memory, **kwargs):
        self.actor = actor
        self.critic = critic
        self.replay_buffer = replay_buffer

        self.target_actor = actor.clone()
        self.target_critic = critic.clone()

    def run(self):
        pass
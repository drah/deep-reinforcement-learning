import torch
from torch import nn
from environment import Environment
from networks import Network
from networks import save_net
from networks import load_net
from memory import ReplayBuffer

class DDPG:
    def __init__(self, actor: Network, critic: Network, buffer_size: int, env: Environment, **kwargs):
        self.actor = actor
        self.critic = critic
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.env = env

        self.target_actor = actor.clone()
        self.target_critic = critic.clone()

    def train(self, n_episode, **kwargs):
        for _ in range(n_episode):

            states = self.env.reset()

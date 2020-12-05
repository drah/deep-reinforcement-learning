import logging

import numpy as np
import torch
from torch import nn
from networks import get_critic
from environment import Environment
from networks import Actor
from networks.utils import make_tensor
from memory import ReplayBufferNumpy
from oup import OrnsteinUhlenbeckProcess
from .algorithm import Algorithm
from .utils import soft_update


_log = logging.getLogger('main')


class DDPG(Algorithm):
    def __init__(self, env: Environment, actor: Actor, **kwargs):
        self.actor = actor
        self.env = env

    def run(self, n_episode, **kwargs):
        # TODO: move this config to a json file
        critic = get_critic(kwargs.get('critic', 'DDPGCritic'),
                            input_shapes=[[None, self.env.state_size],
                                          [None, self.env.action_size]],
                            output_shapes=[[None, 1]],
                            load_path=kwargs.get('ckpt_critic', None))
        target_actor = self.actor.clone()
        target_critic = critic.clone()

        actor_optim = torch.optim.Adam(self.actor.parameters(), 1e-4)
        critic_optim = torch.optim.Adam(
            critic.parameters(), 1e-3, (0.9, 0.99), weight_decay=1e-2)

        replay_buffer = ReplayBufferNumpy(int(1e6))
        warm_start_size = int(1e3)

        tao = 1e-3
        gamma = 0.99  # check
        update_target_cycle = 5  # check

        batch_size = 64
        noise = OrnsteinUhlenbeckProcess(self.env.action_size, 0.2, 0.15, 0.01)

        assert warm_start_size >= batch_size

        _log.info('Training start')

        for i in range(1, n_episode + 1):

            states = self.env.reset()

            # import pdb; pdb.set_trace()
            while True:
                with torch.no_grad():
                    actions = self.actor.act(states)
                noised_actions = actions.numpy() + noise.sample()
                next_states, rewards, dones, _ = self.env.step(noised_actions)

                for state, action, reward, next_state in zip(states, actions, rewards, next_states):
                    replay_buffer.push(state, action, reward,
                                       next_state)  # check dones

                if len(replay_buffer) >= warm_start_size:
                    b_states, b_actions, b_rewards, b_next_states = replay_buffer.sample(
                        batch_size)
                    with torch.no_grad():
                        b_rewards = make_tensor(b_rewards).unsqueeze_(-1)
                        y = b_rewards + gamma * \
                            target_critic.score(
                                b_next_states, target_actor.act(b_next_states))
                    y_pred = critic.score(b_states, b_actions)
                    loss_critic = nn.functional.mse_loss(y_pred, y)

                    critic_optim.zero_grad()
                    loss_critic.backward()
                    critic_optim.step()

                    cur_b_actions = self.actor.act(b_states)
                    loss_actor = -critic.score(b_states, cur_b_actions).mean()

                    actor_optim.zero_grad()
                    loss_actor.backward()
                    actor_optim.step()

                    soft_update(target_actor, self.actor, tao)
                    soft_update(target_critic, critic, tao)

                if np.any(dones):
                    break

        _log.info('Done training for {} episodes.'.format(n_episode))

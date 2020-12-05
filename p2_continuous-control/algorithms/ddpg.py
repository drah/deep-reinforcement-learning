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
from .utils import make_datetime_path


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
        critic_optim = torch.optim.Adam(critic.parameters(), 1e-3)

        replay_buffer = ReplayBufferNumpy(int(1e6))
        warm_start_size = int(1e4)

        tao = 5e-3
        gamma = 0.99

        batch_size = 256
        assert warm_start_size >= batch_size

        save_dir = kwargs.get('save_dir', 'DDPG_logs')

        fh = logging.FileHandler(make_datetime_path(save_dir, 'logs'))
        fh.setLevel(logging.INFO)
        _log.addHandler(fh)

        from collections import deque
        scores = deque([], maxlen=100)

        _log.debug('Training start')

        self.actor.train()
        target_actor.train()
        critic.train()
        target_critic.train()

        for i in range(1, n_episode + 1):

            noise = OrnsteinUhlenbeckProcess(self.env.action_size, 0.2, 0.15, 0.01)

            states = self.env.reset()

            score = 0.0

            while True:
                with torch.no_grad():
                    actions = self.actor.act(states)
                noised_actions = actions.numpy() + noise.sample()
                next_states, rewards, dones, _ = self.env.step(noised_actions)

                score += np.mean(rewards)

                for state, action, reward, next_state in zip(states, actions, rewards, next_states):
                    replay_buffer.push(state, action, reward, next_state)

                if len(replay_buffer) >= warm_start_size:
                    b_states, b_actions, b_rewards, b_next_states = replay_buffer.sample(
                        batch_size)
                    with torch.no_grad():
                        b_rewards = (b_rewards - np.mean(b_rewards)) / (np.std(b_rewards) + 1e-7)
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

            scores.append(score)
            mean = np.mean(scores)
            _log.info(
                '[{}/{}] score: {}, moving average: {}'.format(i, n_episode, score, mean))

            self.actor.save(make_datetime_path(save_dir, 'DDPGAlgo_actor_{}'.format(i)))
            critic.save(make_datetime_path(save_dir, 'DDPGAlgo_critic_{}'.format(i)))

        _log.debug('Done training for {} episodes.'.format(n_episode))

        self.actor.eval()
        target_actor.eval()
        critic.eval()
        target_critic.eval()

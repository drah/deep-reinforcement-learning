from itertools import count
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

        actor_optim = torch.optim.Adam(self.actor.parameters(), kwargs.get('actor_lr', 1e-4))
        critic_optim = torch.optim.Adam(critic.parameters(), kwargs.get('critic_lr', 1e-3),
                                        weight_decay=kwargs.get('critic_weight_decay', 1e-3))

        replay_buffer = ReplayBufferNumpy(int(kwargs.get('buffer_size', 1e5)))
        warm_start_size = int(128)

        tao = kwargs.get('tao', 1e-3)
        gamma = 0.99

        batch_size = 128
        assert warm_start_size >= batch_size

        save_dir = kwargs.get('save_dir', 'DDPG_logs')

        fh = logging.FileHandler(make_datetime_path(save_dir, 'logs'))
        fh.setLevel(logging.INFO)
        _log.addHandler(fh)

        from collections import deque
        scores = deque([], maxlen=100)
        solved = 30.

        _log.debug('Training start')

        self.actor.train()
        target_actor.train()
        critic.train()
        target_critic.train()

        noise_coef = 1.
        noise_coef_decay = kwargs.get('noise_coef_decay', 0.99)
        noise_coef_min = 0.01

        # r_base = 0.05

        update_cycle = 5
        update_times = 5

        t_max = 1000
        t_max_limit = 1000

        state_mean = kwargs.get('state_mean', 0.)
        state_std = kwargs.get('state_std', 1.)
        _log.info('state_mean: {}, state_std: {}'.format(state_mean, state_std))
        
        for i in range(1, n_episode + 1):

            noise = OrnsteinUhlenbeckProcess(self.env.action_size, 0.2, 0.15, 1)

            states = self.env.reset()
            states = np.clip(np.subtract(states, state_mean) / state_std, -1, 1)

            score = 0.0

            for step in range(t_max):
                self.actor.eval()
                with torch.no_grad():
                    actions = self.actor.act(states)
                self.actor.train()
                actions = actions.numpy()
                if len(replay_buffer) >= warm_start_size:
                    sampled_noise = noise.sample() * noise_coef
                    if step == 0:
                        _log.info('sampled_noise: {}, actions: {}'.format(sampled_noise, actions[0]))
                    actions += sampled_noise
                next_states, rewards, dones, _ = self.env.step(np.clip(actions, -1, 1))
                next_states = np.subtract(next_states, state_mean) / state_std

                if np.any(dones):
                    break

                score += np.mean(rewards)

                # mean = np.mean(rewards)
                # std = np.std(rewards)
                # r_mean += (mean - r_mean) * r_momentum
                # r_std += (std - r_std) * r_momentum

                # rewards = np.clip((rewards - r_mean) / (r_std + 1e-7), -10., 10.)
                for state, action, reward, next_state in zip(states, actions, rewards, next_states):
                    replay_buffer.push(state, action, reward, next_state)

                states = next_states
                
                if step % update_cycle == 0 and len(replay_buffer) >= warm_start_size:
                    for _ in range(update_times):
                        b_states, b_actions, b_rewards, b_next_states = replay_buffer.sample(
                            batch_size)

                        with torch.no_grad():
                            b_rewards = make_tensor(b_rewards).unsqueeze_(-1)
                            y = b_rewards + gamma * \
                                target_critic.score(
                                    b_next_states, target_actor.act(b_next_states))
                        y_pred = critic.score(b_states, b_actions)
                        # squared_y_pred = torch.pow(y_pred, 2.)
                        # pan = torch.where(squared_y_pred > torch.Tensor([25.]), squared_y_pred, torch.zeros(1)).sum(-1).mean()
                        
                        loss_critic = 0.5 * torch.mean(torch.sum(torch.pow(y - y_pred, 2.), -1))# + pan

                        critic_optim.zero_grad()
                        loss_critic.backward()
                        nn.utils.clip_grad_norm_(critic.parameters(), max_norm=5.)
                        critic_optim.step()
                        
                    for _ in range(update_times):
                        b_states, b_actions, b_rewards, b_next_states = replay_buffer.sample(
                            batch_size)
                        
                        cur_b_actions = self.actor.act(b_states)
                        loss_actor = -critic.score(b_states, cur_b_actions).mean()

                        actor_optim.zero_grad()
                        critic_optim.zero_grad()
                        loss_actor.backward()
                        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5.)
                        actor_optim.step()

                    for _ in range(update_times):
                        soft_update(target_actor, self.actor, tao)
                        soft_update(target_critic, critic, tao)

                    _log.info('loss: critic: {}, actor: {}'.format(loss_critic, loss_actor))

            scores.append(score)
            mean = np.mean(scores)
            _log.info(
                '[{}/{}] score: {:.4f}, moving average: {:.4f}, steps: {}'.format(
                    i, n_episode, score, mean, step))

            if mean > solved:
                __log.info('Solved!')
                break

            if i % 100 == 0:
                self.actor.save(make_datetime_path(save_dir, 'DDPGAlgo_actor_{}'.format(i)))
                critic.save(make_datetime_path(save_dir, 'DDPGAlgo_critic_{}'.format(i)))

            noise_coef = max(noise_coef * noise_coef_decay, noise_coef_min)
            t_max = min(t_max + 1, t_max_limit)

        _log.debug('Done training for {} episodes.'.format(n_episode))

        self.actor.eval()
        target_actor.eval()
        critic.eval()
        target_critic.eval()

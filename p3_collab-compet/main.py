import argparse
import gym
import random
import sys
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import ddpg_agent
from Tennis import Tennis


def reset_all_agents(agents):
    ''' Call to each agent's reset method '''
    [agent.reset() for agent in agents]


def collect_data(reward_accum_steps, agents, state, env, score):
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    for _ in range(reward_accum_steps):
        action = [agent.act(s) for agent, s in zip(agents, state)]
        next_state, reward, done, _ = env.step(action)
        score += reward

        states.append(state)  # [accum_steps, num_agents, state_size]
        actions.append(action)  # [accum_steps, num_agents, action_size]
        rewards.append(reward)  # [accum_steps, num_agents]
        next_states.append(next_state)  # [accum_steps, num_agents, state_size]
        dones.append(done)  # [accum_steps, num_agents]

        if any(done):
            break

        state = next_state

    return rewards, states, actions, next_states, dones, score


def accum_rewards(rewards, discounts):
    ''' Accumulate and mix up the rewards of the two agent '''
    rewards = np.array(rewards, dtype=np.float32)
    length = len(rewards)
    for accum_step_i in range(length):
        rewards[accum_step_i, :] = np.sum(
            rewards[accum_step_i:, :] * discounts[:length-accum_step_i, :], 0)
    rewards = rewards + 0.1 * rewards[:, ::-1]
    return rewards


def agents_step(rewards, env, agents, states, actions, next_states, dones):
    ''' Call to each agent's step method '''
    for accum_step_i in range(len(rewards)):
        for agent_i in range(env.num_agents):
            agents[agent_i].step(
                states[accum_step_i][agent_i],
                actions[accum_step_i][agent_i],
                rewards[accum_step_i][agent_i],
                next_states[accum_step_i][agent_i],
                dones[accum_step_i][agent_i],
                states[accum_step_i][(agent_i + 1) % 2],
                next_states[accum_step_i][(agent_i + 1) % 2])


def maddpg(agents, env, **kwargs):
    '''
    Multi-Agent DDPG Algorithm.
    This method trains two agents in the environment Tennis.
    '''
    n_episode = kwargs.get('n_episodes', 10000)
    max_t = kwargs.get('max_t', 1000)
    window_size = kwargs.get('window_size', 100)
    ckpt_prefix = kwargs.get('ckpt_prefix', 'checkpoint')
    reward_accum_steps = kwargs.get('reward_accum_steps', 1000)
    reset_cycle = kwargs.get('reset_cycle', 20000)

    scores_deque = deque(maxlen=window_size)
    scores = []
    max_t = max_t // reward_accum_steps * reward_accum_steps

    discounts = np.expand_dims(0.99 ** np.arange(reward_accum_steps), 1)
    sample_count = 0
    for i_episode in range(1, n_episode + 1):
        state = env.reset()

        if sample_count >= reset_cycle:
            reset_all_agents(agents)
            sample_count = 0

        score = np.zeros([env.num_agents])

        t = 0
        while t < max_t:
            rewards, states, actions, next_states, dones, score = collect_data(
                reward_accum_steps, agents, state, env, score)
            rewards = accum_rewards(rewards, discounts)
            agents_step(rewards, env, agents, states,
                        actions, next_states, dones)
            t += len(rewards)
            sample_count += len(rewards)

        cur_max = np.max(score)
        scores_deque.append(cur_max)
        scores.append(cur_max)
        moving_mean = np.mean(scores_deque)
        print('[{}] Average Score: {:.4f} Cur: {:.4f}'.format(
            i_episode, moving_mean, cur_max), end='\n')
        for agent_i, agent in enumerate(agents):
            torch.save(agent.actor_local.state_dict(),
                       ckpt_prefix + '_actor_{}.pth'.format(agent_i))
            torch.save(agent.critic_local.state_dict(),
                       ckpt_prefix + '_critic_{}.pth'.format(agent_i))

        if len(scores_deque) == window_size and moving_mean >= 1.0:
            print("Solved at episode {}!".format(i_episode - window_size + 1))
            break

    return scores


def plot(scores, save_path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(save_path)


def show(agents, env):
    state = env.reset(train_mode=False)
    for _ in range(1000):
        action = [agent.act(s, add_noise=False)
                  for agent, s in zip(agents, state)]
        state, reward, done, _ = env.step(action)


def main(args):
    env = Tennis(args.env_path)

    config = {
        'state_size': env.state_size,
        'action_size': env.action_size,
        'reward_accum_steps': 1000,
        'random_seed': 1,
        'gamma': 0.99,
        'update_cycle': 400,
        'update_times': 10,
        'buffer_size': int(1e6),
        'batch_size': 1024,
        'warm_start_size': 1024,
        'n_episode': 1000000,
        'max_t': 1000,
        'window_size': 100,
        'ckpt_prefix': 'checkpoint',
        'reset_cycle': 20000,
    }

    agents = [ddpg_agent.Agent(**config) for _ in range(env.num_agents)]

    if args.train:
        scores = maddpg(agents, env, **config)
        plot(scores, args.png_path)

    if args.show:
        for agent_i, agent in enumerate(agents):
            agent.actor_local.load_state_dict(torch.load(
                'checkpoint_actor_{}.pth'.format(agent_i), lambda a, b: a))
            agent.critic_local.load_state_dict(torch.load(
                'checkpoint_critic_{}.pth'.format(agent_i), lambda a, b: a))
        show(agents, env)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env_path', help='The absolute path to Tennis.')
    parser.add_argument('png_path', help='The path to save the scores.png')
    parser.add_argument('--train', action='store_true', help='Run training.')
    parser.add_argument('--show', action='store_true',
                        help='To view the agents show.')
    args = parser.parse_args()
    main(args)

import sys

import torch
import numpy as np
import ddpg_agent
from Tennis import Tennis

def show(env, agents):
    state = env.reset(train_mode=False)
    for _ in range(1000):
        action = [agent.act(s, add_noise=False) for agent, s in zip(agents, state)]
        state, reward, done, _ = env.step(action)

if len(sys.argv) != 2:
    print("Usage: python3 show.py <absolute_path_to_Tennis>")
    exit(-1)

env = Tennis(sys.argv[1])

agents = [ddpg_agent.Agent(
    state_size=env.state_size,
    action_size=env.action_size,
    random_seed=1,
    gamma=0.99,
    update_cycle=400,
    update_times=10,
    buffer_size=int(1e6),
    batch_size=1024,
    warm_start_size=1024) for _ in range(env.num_agents)]

for agent_i, agent in enumerate(agents):
    agent.actor_local.load_state_dict(torch.load('checkpoint_actor_{}.pth'.format(agent_i), lambda a, b: a))
    agent.critic_local.load_state_dict(torch.load('checkpoint_critic_{}.pth'.format(agent_i), lambda a, b: a))

show(env, agents)

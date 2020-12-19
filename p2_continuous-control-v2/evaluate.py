def evaluate(env, agent):
    import numpy as np
    state = env.reset()
    score = None
    for i in range(1000):
        action = agent.act(state, add_noise=False)
        state, reward, done, _ = env.step(action)
        if score is None:
            score = np.zeros(len(reward))
        score += reward
        if np.all(done):
            break
    return score

import sys
from Reacher import Reacher
Reacher_path = sys.argv[1]
env_20 = Reacher(Reacher_path)

import ddpg_agent
reward_accum_steps = 20
agent_20 = ddpg_agent.Agent(
    state_size=33,
    action_size=4,
    random_seed=1,
    gamma=0.99,
    update_cycle=reward_accum_steps * 20,
    update_times=reward_accum_steps * 20 // 40,
    buffer_size=int(1e6),
    batch_size=1024,
    warm_start_size=1024)

import torch
agent_20.actor_local.load_state_dict(torch.load('checkpoint_20_actor.pth'))
agent_20.critic_local.load_state_dict(torch.load('checkpoint_20_critic.pth'))
score = evaluate(env_20, agent_20)
print("Score: ", score)
print("Average: ", sum(score) / len(score))

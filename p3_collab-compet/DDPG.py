import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from Tennis import Tennis

def ddpg(agent, env, n_episodes=10000, max_t=1000, window_size=100, ckpt_prefix='checkpoint', reward_accum_steps=150):
    num_parallel = 2
    
    scores_deque = deque(maxlen=window_size)
    scores = []
    max_t = max_t // reward_accum_steps * reward_accum_steps

    discounts = np.expand_dims(0.99 ** np.arange(reward_accum_steps), 1)
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        agent.reset()
        score = np.zeros([num_parallel])
        
        t = 0
        while t < max_t:
            # collect data
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []
            for _ in range(reward_accum_steps):
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                score += reward
                
                states.append(state)  # [accum_steps, num_parallel, state_size]
                actions.append(action)  # [accum_steps, num_parallel, action_size]
                rewards.append(reward)  # [accum_steps, num_parallel]
                next_states.append(next_state) # [accum_steps, num_parallel, state_size]
                dones.append(done) # [accum_steps, num_parallel]

                if any(done):
                    break
                
                state = next_state
                t += 1

            # calculate rewards
            rewards = np.array(rewards, dtype=np.float32)
            length = len(rewards)
            for accum_step_i in range(length):
                rewards[accum_step_i,:] = np.sum(rewards[accum_step_i:,:] * discounts[:length-accum_step_i,:], 0)
            
            # agent step
            for accum_step_i in range(length):
                for parallel_i in range(num_parallel):
                    agent.step(states[accum_step_i][parallel_i],
                               actions[accum_step_i][parallel_i],
                               rewards[accum_step_i][parallel_i],
                               next_states[accum_step_i][parallel_i],
                               dones[accum_step_i][parallel_i])

        scores_deque.append(score)
        scores.append(score)
        cur_mean = np.mean(score)
        moving_mean = np.mean(scores_deque)
        print('\rEpisode {}    Average Score: {:.2f}    Cur Score: {:.2f}                       '.format(
            i_episode, moving_mean, cur_mean))
        torch.save(agent.actor_local.state_dict(), ckpt_prefix + '_actor.pth')
        torch.save(agent.critic_local.state_dict(), ckpt_prefix + '_critic.pth')

        if len(scores_deque) == window_size and moving_mean >= 30.:
            print("Solved at episode {}!".format(i_episode - window_size + 1))
            break

    return scores

import sys
env_path = sys.argv[1]
env = Tennis(env_path)

import ddpg_agent
reward_accum_steps = 150
agent = ddpg_agent.Agent(
    state_size=env.state_size,
    action_size=env.action_size,
    random_seed=1,
    gamma=0.99,
    update_cycle=reward_accum_steps * 2,
    update_times=10,
    buffer_size=int(1e6),
    batch_size=1024,
    warm_start_size=1024)

scores = ddpg(agent, env, 10000, ckpt_prefix='checkpoint', reward_accum_steps=reward_accum_steps)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('scores.png')

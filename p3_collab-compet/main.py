import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from Tennis import Tennis

def ddpg(agents, env, n_episodes=10000, max_t=1000, window_size=100, ckpt_prefix='checkpoint', reward_accum_steps=150, reset_cycle=20000):
    num_parallel = 2
    
    scores_deque = deque(maxlen=window_size)
    scores = []
    max_t = max_t // reward_accum_steps * reward_accum_steps

    discounts = np.expand_dims(0.99 ** np.arange(reward_accum_steps), 1)
    sample_count = 0
    for i_episode in range(1, n_episodes+1):
        state = env.reset()

        if sample_count >= reset_cycle:
            [agent.reset() for agent in agents]
            sample_count = 0

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
                action = [agent.act(s) for agent, s in zip(agents, state)]
                next_state, reward, done, _ = env.step(action)
                score += reward
                
                states.append(state)  # [accum_steps, num_parallel, state_size]
                actions.append(action)  # [accum_steps, num_parallel, action_size]
                rewards.append(reward)  # [accum_steps, num_parallel]
                next_states.append(next_state) # [accum_steps, num_parallel, state_size]
                dones.append(done) # [accum_steps, num_parallel]

                if any(done):
                    t = max_t
                    break
                
                state = next_state
                t += 1

            # calculate rewards
            rewards = np.array(rewards, dtype=np.float32)
            length = len(rewards)
            for accum_step_i in range(length):
                rewards[accum_step_i,:] = np.sum(rewards[accum_step_i:,:] * discounts[:length-accum_step_i,:], 0)
            rewards = rewards + 0.1 * rewards[:,::-1]
            
            # agent step
            for accum_step_i in range(length):
                for parallel_i in range(num_parallel):
                    agents[parallel_i].step(
                            states[accum_step_i][parallel_i],
                            actions[accum_step_i][parallel_i],
                            rewards[accum_step_i][parallel_i],
                            next_states[accum_step_i][parallel_i],
                            dones[accum_step_i][parallel_i],
                            states[accum_step_i][(parallel_i + 1) % 2],
                            next_states[accum_step_i][(parallel_i + 1) % 2])
                sample_count += 1

        cur_max = np.max(score)
        scores_deque.append(cur_max)
        scores.append(cur_max)
        moving_mean = np.mean(scores_deque)
        print('[{}] Average Score: {:.4f} Cur: {:.4f}'.format(i_episode, moving_mean, cur_max), end='\n')
        for agent_i, agent in enumerate(agents):
            torch.save(agent.actor_local.state_dict(), ckpt_prefix + '_actor_{}.pth'.format(agent_i))
            torch.save(agent.critic_local.state_dict(), ckpt_prefix + '_critic_{}.pth'.format(agent_i))

        if len(scores_deque) == window_size and moving_mean >= 0.5:
            print("Solved at episode {}!".format(i_episode - window_size + 1))
            break

    return scores

# import sys
# env_path = sys.argv[1]
# env = Tennis(env_path)
env = Tennis('/home/bobo258/and/environment/Tennis_Linux_NoVis/Tennis.x86_64')

import ddpg_agent
reward_accum_steps = 1000
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

scores = ddpg(agents, env, 1000000, ckpt_prefix='checkpoint', reward_accum_steps=reward_accum_steps)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('scores.png')

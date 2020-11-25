from logging import getLogger

from unityagents import UnityEnvironment
import numpy as np
import gym

_log = getLogger('main')

class Reacher:
    def __init__(self, **kwargs):
        self.env = UnityEnvironment(file_name=kwargs.get('app_path', 'Reacher.app'))

        self.brain_name = self.env.brain_names[0]

        brain = self.env.brains[self.brain_name]
        self.action_size = brain.vector_action_space_size

        env_info = self.env.reset(train_mode=True)[self.brain_name]
        self.num_agents = len(env_info.agents)

        states = env_info.vector_observations
        self.state_size = states.shape[1]

        verbose = kwargs.get('verbose', 0)
        _log.info('Brain name: %s' % self.brain_name)
        _log.info('Number of agents: %d' % self.num_agents)
        _log.info('Action size: %d' % self.action_size)
        _log.info('State size: %d' % self.state_size)
        _log.info('The state for the first agent looks like: %s' % states[0])

    def reset(self):
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        return env_info.vector_observations

    def step(self, actions):
        env_info = self.env.step(actions)[self.brain_name]
        return (env_info.vector_observations, env_info.rewards, env_info.local_done, env_info)

    def show_env(self):
        env_info = self.env.reset(train_mode=False)[self.brain_name]     # reset the environment    
        states = env_info.vector_observations                  # get the current state (for each agent)
        scores = np.zeros(self.num_agents)                          # initialize the score (for each agent)
        while True:
            actions = np.random.randn(self.num_agents, self.action_size) # select an action (for each agent)
            actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
            print("actions: ", actions, end='\r', flush=True)
            env_info = self.env.step(actions)[self.brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            scores += env_info.rewards                         # update the score (for each agent)
            states = next_states                               # roll over states to next time step
            if np.any(dones):                                  # exit loop if episode finished
                break
        _log.info('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

class GymWrapper:
    def __init__(self, env_name, **kwargs):
        self.env = gym.make(env_name)

    def __del__(self):
        try:
            self.env.close()
        except:
            pass

    def show_env(self):
        self.env.reset()
        while True:
            self.env.render()
            action = self.env.action_space.sample()
            print("action: ", action, end='\r', flush=True)
            _, _, done, _ = self.env.step(action)
            if done:
                break

def get_env(env_name, **kwargs):
    if env_name == 'Reacher':
        return Reacher(**kwargs)
    else:
        return GymWrapper(env_name, **kwargs)

if __name__ == '__main__':
    env = Reacher()
    env.show_env()
    # env = get_env('Acrobot-v1')
    # env.show_env()
    # env = get_env('MountainCarContinuous-v0')
    # env.show_env()

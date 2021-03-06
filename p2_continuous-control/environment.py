from logging import getLogger

from unityagents import UnityEnvironment
import numpy as np
import gym

_log = getLogger('main')

class Environment:
    def __init__(self, **kwargs):
        self.state_size = None
        self.action_size = None
        raise NotImplementedError("Abstract Class")

    def reset(self):
        raise NotImplementedError("Abstract Class")

    def step(self, actions):
        raise NotImplementedError("Abstract Class")

class Reacher(Environment):
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
        _log.debug('Brain name: %s' % self.brain_name)
        _log.debug('Number of agents: %d' % self.num_agents)
        _log.debug('Action size: %d' % self.action_size)
        _log.debug('State size: %d' % self.state_size)
        _log.debug('The state for the first agent looks like: %s' % states[0])

    def reset(self, **kwargs):
        env_info = self.env.reset(train_mode=not kwargs.get('render', False))[self.brain_name]
        return env_info.vector_observations

    def step(self, actions):
        # actions = actions.data.numpy()
        env_info = self.env.step(actions)[self.brain_name]
        return (env_info.vector_observations, env_info.rewards, env_info.local_done, env_info)

    def show_env(self):
        states = self.reset(render=True)
        scores = np.zeros(self.num_agents)                          # initialize the score (for each agent)
        while True:
            actions = np.random.randn(self.num_agents, self.action_size) # select an action (for each agent)
            actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
            print("actions: ", actions, end='\r', flush=True)
            states, rewards, dones, _ = self.step(actions)
            scores += env_info.rewards                         # update the score (for each agent)
            if np.any(dones):                                  # exit loop if episode finished
                break
        _log.info('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

class GymWrapper(Environment):
    def __init__(self, env_name, **kwargs):
        self.env = gym.make(env_name)

    def __del__(self):
        try:
            self.env.close()
        except:
            pass

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        return self.env.step(actions)

    def show_env(self):
        self.env.reset()
        while True:
            self.env.render()
            action = self.env.action_space.sample()
            print("action: ", action, end='\r', flush=True)
            _, _, done, _ = self.env.step(action)
            if done:
                break

def get_env(env_name, **kwargs) -> Environment:
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

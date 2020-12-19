from unityagents import UnityEnvironment
import numpy as np

class Tennis:
    def __init__(self, path: str):
        env = UnityEnvironment(file_name=path)
        # get the default brain
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]

        env_info = env.reset(train_mode=True)[brain_name]

        num_agents = len(env_info.agents)
        print('Number of agents:', num_agents)

        action_size = brain.vector_action_space_size
        print('Size of each action:', action_size)

        states = env_info.vector_observations
        state_size = states.shape[1]
        print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
        print('The state for the first agent looks like:', states[0])

        self.env = env
        self.brain_name = brain_name
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents

    def __del__(self):
        self.env.close()

    def reset(self, train_mode=True):
        env_info = self.env.reset(train_mode=train_mode)[self.brain_name]
        return env_info.vector_observations

    def step(self, actions):
        # actions shape (num_agent, action_size)
        env_info = self.env.step(actions)[self.brain_name]
        rewards = env_info.rewards
        next_states = env_info.vector_observations
        dones = env_info.local_done
        return next_states, rewards, dones, env_info

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 Tennis.py <path_to_Tennis_app>")
        exit()
    env = Tennis(sys.argv[1])
    for _ in range(5):
        states = env.reset(train_mode=False)
        scores = np.zeros((env.num_agents,))
        step = 0
        while True:
            step += 1
            actions = np.random.randn(env.num_agents, env.action_size)
            states, rewards, dones, _ = env.step(actions)
            scores += rewards
            if any(dones):
                print("Done at step {}, reward: {}, all dones: {}".format(step, scores, all(dones)))
                print(states)
                break
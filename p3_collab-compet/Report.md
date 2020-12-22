---
title: Multi Agent RL - Collaboration and Competition
---

# Introduction
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

# Setup
1. Clone this repository and cd into this directory
`$ git clone https://github.com/drah/deep-reinforcement-learning.git`
`$ cd deep-reinforcement-learning/p3_collab-compet`
2. Install the provided environment and gym in your virtual env.
`$ pip3 install ../python/`
`$ pip3 install gym`

2. Download the environment 'Tennis' according your system and unzip it.
Max: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip
Linux: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip

# Train
`$ python3 main.py <absolute_path_to_Tennis> --train`
For Linux, <absolute_path_to_Tennis> is like <some_directory>/Tennis/Tennis.x86_64
For Mac, <absolute_path_to_Tennis> is like <some_directory>/Tennis.app

# Show
`$ python3 show.py <absolute_path_to_Tennis> --show`

# Method
The method used to solve this task is based on DDPG.
DDPG has 4 networks:
1. actor network: takes states as input and output actions.
    - consists of 3 fully connected layers, with 400(hidden_size1), 300(hidden_size2), 4(action_size) unit(s) respectively.
    - use elu as activation functions for the first 2 layers, and tanh for the output layer to make the output range in [-1, 1].
    - refer to Actor in model.py for details.
2. critic network: takes states and actions as inputs and output scores.
    - consists of 3 fully connected layers, with 400(hidden_size1), 300(hidden_size2), 1(score_size) unit(s) respectively.
    - use selu as activation functions for the first 2 layers.
    - refer to Critic in model.py for details.
    - note that critic network will have full observations, a.k.a the states of the two agents, in this task.
3. target actor network: a cloned actor network
    - used when we train the critic network
4. target critic network: a cloned critic network
    - used when we train the critic network
When the training starts, the agent will interact with the environment using the actor network to collect data
a.k.a (states, actions, rewards, next_states, dones). The agent will store these data into a memory buffer.
The agent will periodically sample batch data to train the critic network. For the loss function to train the
critic network, please refer to section 'udpate critic' in the method 'learn' in ddpg_agent.py.
Note that for making the training more stable, we clip the gradients of the critic by norm.

The actor network mainly learns from the critic network via following the gradients back-propagated from the critic.
Please refer to section 'udpate actor' in the method 'learn' in ddpg_agent.py.
We can imaging that the critic network should know that what action has higher score given this state, and use the
gradients to tell the actor network this information.

After training the critic and actor network for 1 step, we use soft-update to update the target actor network and
target critic network. Please refer to section 'update target networks' in the method 'learn' in ddpg_agent.py.

For exploration, we use the method described in Noisy Net, which adds noise to the actor parameters instead of adding
to the actions.

In this project, we highlight the following add-ons:
1. 2 agents instead of 1 with their own actors and critics.
2. The critic of each agent has full observations, a.k.a. both of the states.
3. The rewards of one agent will be added to the rewards of another agent.
4. Noise is added to the actor parameters every 20000 steps instead of every episode.
With these add-ons, the two agents will learn to collaborate and compete with each other in the environment.
Please see the result in the next section.

# Result
Please refer to dump.txt for the training progress.
The following is the plot of scores: the blue is the scores of each episode, while the orange is the average scores over 100 consecutive episodes.
![scores](https://github.com/drah/deep-reinforcement-learning/blob/master/p3_collab-compet/scores.png?raw=true)

In dump.txt, we can see that the task is solved when training 24597 episodes.

The video of the agents show.
[![](http://img.youtube.com/vi/--FFYKM8ofc/0.jpg)](http://www.youtube.com/watch?v=--FFYKM8ofc "")

# Future Work
Currently the two agents have their own critics respectively. From the above video, we can see that the behavior of the red one may be much like what we want the agent does. The future work may use one critic for the two agents to improve the performance.

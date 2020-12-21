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

[![](http://img.youtube.com/vi/--FFYKM8ofc/0.jpg)](http://www.youtube.com/watch?v=--FFYKM8ofc "")

# Setup
1. Install the provided environment and gym in your virtual env.
`$ pip3 install ../python/`
`$ pip3 install gym`

2. Download the environment 'Tennis' according your system and unzip it.
- Linux: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip
- Mac: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip

# Train
`$ python3 main.py <absolute_path_to_Tennis>`
- For Linux, <absolute_path_to_Tennis> is like <some_directory>/Tennis/Tennis.x86_64
- For Mac, <absolute_path_to_Tennis> is like <some_directory>/Tennis.app

# Show
`$ python3 show.py <absolute_path_to_Tennis>`

# Method
The method used to solve this task is based on DDPG. For the details of DDPG_agent, please refer to the introduction in p2_continuous-control-v2/README.md.
In this task, the following changes are made:
1. 2 agents instead of 1 with their own actors and critics.
2. The critic of each agent has full observations, a.k.a. both of the states.
3. The rewards of one agent will be added to the rewards of another agent.
4. Noise is added to the actor parameters every 20000 steps instead of every episode.

# Result
Please refer to dump.txt for the training progress.
The following is the plot of scores: the blue is the scores of each episode, while the orange is the average scores over 100 consecutive episodes.
![scores](https://github.com/drah/deep-reinforcement-learning/blob/master/p3_collab-compet/scores.png?raw=true)

In dump.txt, we can see that the task is solved when training 24597 episodes.

# Future Work
Currently the two agents have their own critics respectively. From the above video, we can see that the behavior of the red one may be much like what we want the agent does. The future work may use one critic for the two agents to improve the performance.

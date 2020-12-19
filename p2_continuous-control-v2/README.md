---
title: Continuous Control for Reacher
---

# Introduction
This project solves the task Reacher using DDPG based algorithm.
The task Reacher is to keep the 'arms' in the 'moving target area'.
The agent will receive states (vectors of 33 values representing positions, velocities, ...etc.)
and produce actions (vectors of 4 values representing how to move the arm).
We use the version Reacher_20 which runs 20 arms in parallel.
The task is considered solved if the agent get an averaged rewards more than 30 over 100 consecutive episodes.

# Setup
`$ pip3 install ../python/`
You can install it in your virtual env.
`$ pip3 install gym`
Download the Reacher from the following links and unzip it.
linux: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip
maxOs: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip

# Train
python3 DDPG.py <absolute_path_to_Reacher>
In my experiement, the task will be solved after about 110 episodes.
After training, check the plot called scores.png.
The checkpoints will also saved called checkpoint_20_actor.pth and checkpoint_20_critic.pth.

# Evaluate
python3 evaluate.py <absolute_path_to_Reacher>

# Algorithm
The algorithm is based on DDPG, with the changes including but not limited to:
1. decay learning rate
2. enlarge batch size
3. use l1 loss for critic loss instead of l2 loss
4. accumulate the rewards
5. decay weight
6. different activation functions
7. different random seed
8. add batch normalizations to network
9. add batch normalizations to rewards
10. multiple agents v.s. one agent
11. different buffer size
...

And, the changes resulting in significant improvement are:
1. adding noise to network parameters instead of actions (Noisy Net)
2. accumulate rewards

# Future Work
Dueling Network may be added to improve the performance further.

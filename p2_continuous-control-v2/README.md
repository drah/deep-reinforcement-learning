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

# Result
Please check scores.txt and scores.png.
We can see that the average rewards over 100 consecutive episodes has come to 30.06 after training 114 episodes.

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

## More Details
DDPG has 4 networks:
1. actor network: takes states as input and output actions.
    - consists of 3 fully connected layers, with 400(hidden_size1), 300(hidden_size2), 4(action_size) unit(s) respectively.
    - use elu as activation functions for the first 2 layers, and tanh for the output layer to make the output range in [-1, 1].
    - refer to Actor in model.py for details.
2. critic network: takes states and actions as inputs and output scores.
    - consists of 3 fully connected layers, with 400(hidden_size1), 300(hidden_size2), 1(score_size) unit(s) respectively.
    - use selu as activation functions for the first 2 layers.
    - refer to Critic in model.py for details.
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

For exploration, we use the method described in Noisy Net, which adds noise to the actor parameters at the begining of
every episode. And we don't add noise to the actions.


# Future Work
Dueling Network may be added to improve the performance further.

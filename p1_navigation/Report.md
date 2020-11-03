# Project 1: Navigation

## Goals:
- [x] Train an agent to walk in the environment Banana.
- [x] Get average rewards high than 13 over 100 consecutive episodes.
- [x] Solve the project in fewer than 1800 episodes.

## Achievements:
- Achieve 15.01 after training 865 episodes.

Please refer to the Navigation.ipynb for the following sections.
The section 'Navigation' remains for getting familiar with the environment.

## Introduction:
In this project, we are going to train an agent to collect yellow bananas in the environment.
The agent will get input states with state_size 37, and need to provide actions with action_size 4.
The goal of this project is that the agent is able to receive an average reward (over 100 episodes) of at least +13.

## Instructions:
This project is completed in the udacity workspace. Please refer to README.md for more details.

## Train:
Please refer to the section 'Train'.
In this section, a dqn_agent will be imported to interact with the environment.
The dqn_agent uses a simple model consisting of three fully connected as its Q network.
The hyperparameters for training are also listed.
During training, we save the weights every 1000 steps.
And if the mean score of the latest 100 episodes is higher than 15, the training will stop.

We can see that after training for 865 episodes, the agent has an average score 15.01.

And, we can see the increass of the score from the following plot.
The blue line stands for the scores of each episode.
The orange line stands for the mean scores of every 100 consective episodes.

The trained weights are also saved after training.

## Evaluation
Please refer to the section 'Evaluation'
In this section, a dqn_agent will use the trained weights to interact with the environment for 100 episodes.
We can see that the mean score of 100 episodes is 15.21.


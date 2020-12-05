from itertools import count
import logging
import numpy as np

from environment import Environment
from networks import Actor

__log = logging.getLogger('main')


def show_agent_play(env: Environment, actor: Actor, **kwargs):
    run_agent(env, actor, True, 1)

def evaluate(env: Environment, actor: Actor, **kwargs):
    run_agent(env, actor, False, kwargs.get('n_eval_episode', 100))

def run_agent(env: Environment, actor: Actor, render: bool, n_eval_episode: int, **kwargs):
    actor.eval()

    scores = []

    for _ in range(n_eval_episode):
        score = 0.

        states = env.reset(render=render)
        for step in count():
            actions = actor.act(states)
            actions = actions.detach().numpy()

            __log.debug("Actions: %s." % str(actions))
            states, rewards, dones, _ = env.step(actions)

            score += np.mean(rewards)

            if any(dones):
                __log.info("Done.")
                break

            if 'max_step' in kwargs and step >= kwargs['max_step']:
                __log.info("Break due to hit max_step.")
                break

        __log.info("Score: {}".format(score))
        scores.append(score)

    __log.info("Average: {}".format(np.mean(scores)))

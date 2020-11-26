from itertools import count
import logging
import numpy as np

from environment import Environment
from networks import Network

__log = logging.getLogger('main')


def show_agent_play(env: Environment, actor: Network, **kwargs):
    states = env.reset(render=True)
    scores = np.zeros(len(states), np.float32)
    for step in count():
        actions = actor.act(states)
        __log.info("Actions: %s." % str(actions))
        states, rewards, dones, _ = env.step(actions)
        scores += np.array(rewards, dtype=np.float32)
        if all(dones):
            __log.info("Done.")
            break
        if 'max_step' in kwargs and step >= kwargs['max_step']:
            __log.info("Break due to hit max_step.")
            break
    __log.info("Scores: %s" % str(scores))

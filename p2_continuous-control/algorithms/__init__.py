from .runner import show_agent_play
from .ddpg import DDPG

def get_algorithm(algorithm_name, **kwargs):
    return eval(algorithm_name)(**kwargs)
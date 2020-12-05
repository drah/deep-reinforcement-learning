from .algorithm import Algorithm
from .ddpg import DDPG
from .runner import show_agent_play
from .runner import evaluate

def get_algorithm(algorithm_name, **kwargs) -> Algorithm:
    return eval(algorithm_name)

from datetime import datetime
import os

from networks import Network


def soft_update(target: Network, source: Network, tao: float):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.detach_()
        target_param.copy_(target_param * (1.0 - tao) + param * tao)

def make_datetime_path(save_dir: str, save_name: str):
    return os.path.join(save_dir, save_name + datetime.now().strftime("_%Y%m%d_%H%M%S"))
from networks import Network


def soft_update(target: Network, source: Network, tao: float):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.detach_()
        target_param.copy_(target_param * (1.0 - tao) + param * tao)
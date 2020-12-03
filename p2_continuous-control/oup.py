import numpy as np


class RandomBase(object):
    def reset(self):
        raise NotImplementedError("Abstract Class")

    def sample(self):
        raise NotImplementedError("Abstract Class")


class Gaussian(RandomBase):
    def __init__(self, size: [int], mean=lambda: 0., std=lambda: 1.):
        self.size = size
        self.mean = mean
        self.std = std

    def sample(self):
        return np.random.normal(self.mean(), self.std(), self.size())


class OrnsteinUhlenbeckProcess(RandomBase):
    def __init__(self, size: [int], std: float, theta: float, delta_t: float, x0=None):
        self.size = size
        self.std = std
        self.theta = theta
        self.delta_t = delta_t
        self.x0 = x0
        self.mu = 0
        self.x_prev = None

        self.reset()

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)

    def sample(self):
        v1 = self.theta * (self.mu - self.x_prev) * self.delta_t
        v2 = self.std * np.sqrt(self.delta_t) * np.random.normal(size=self.size)
        x = self.x_prev + v1 + v2
        self.x_prev = x
        return x

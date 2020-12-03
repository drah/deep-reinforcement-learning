import random


class Memory:
    def __init__(self):
        raise NotImplementedError("Abstract Class")

    def push(self, item):
        raise NotImplementedError("Don't call me")

    def sample(self, n_sample: int):
        raise NotImplementedError("Don't call me")

    def __len__(self):
        raise NotImplementedError("Don't call me")


class ReplayBuffer(Memory):
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size

        from collections import deque
        self.memory = deque(maxlen=buffer_size)

    def push(self, item):
        self.memory.append(item)

    def sample(self, n_sample: int):
        sample = random.sample(self.memory, n_sample)
        n_content = len(sample[0])
        return [[sample[b][i] for b in range(n_sample)] for i in range(n_content)]

    def __len__(self):
        return len(self.memory)


def set_seed(seed: int):
    random.seed(seed)


def get_memory(memory_name: str, **kwargs) -> Memory:
    return eval(memory_name)(**kwargs)

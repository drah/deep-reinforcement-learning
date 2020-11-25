import random

class Memory:
    def __init__(self):
        raise NotImplementedError("Abstract Class")

    def push(self, item):
        pass

    def sample(self, n_sample: int):
        pass

class ReplayBuffer(Memory):
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size

        from collections import deque
        self.memory = deque(maxlen=buffer_size)

    def push(self, item):
        self.memory.append(item)

    def sample(self, n_sample: int):
        return random.sample(self.memory, n_sample)

def set_seed(seed: int):
    random.seed(seed)

def get_memory(memory_name: str, **kwargs) -> Memory:
    return eval(memory_name)(**kwargs)

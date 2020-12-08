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

    def push(self, *items):
        self.memory.append(items)

    def sample(self, n_sample: int):
        sample = random.sample(self.memory, n_sample)
        n_content = len(sample[0])
        return [[sample[b][i] for b in range(n_sample)] for i in range(n_content)]

    def __len__(self):
        return len(self.memory)


class ReplayBufferNumpy(Memory):
    def __init__(self, buffer_size: int):
        if buffer_size <= 0:
            raise Exception("Buffer size should be greater than 0.")
        self.buffer_size = buffer_size

        self.__is_initialized = False
        self.__index = None
        self.__size = 0
        self.__memories = None

    def init_memory(self, *items):
        if self.__is_initialized:
            raise Exception("Memory has been initialized before.")

        import numpy as np

        self.__memories = []
        for item in items:
            self.__memories.append(
                np.zeros(shape=[self.buffer_size, *np.array(item).shape], dtype=np.float32))

        self.__index = 0
        self.__size = 0
        self.__is_initialized = True

    def push(self, *items):
        if not self.__is_initialized:
            self.init_memory(*items)

        for i, item in enumerate(items):
            self.__memories[i][self.__index, ...] = item
        self.__index = (self.__index + 1) % self.buffer_size
        self.__size = min(self.__size + 1, self.buffer_size)

    def sample(self, n_sample: int):
        indices = random.sample(range(self.__size), n_sample)
        return [memory[indices] for memory in self.__memories]

    def __len__(self):
        return self.__size

    def get_memory_shapes(self):
        if not self.__is_initialized:
            return None
        return [memory.shape for memory in self.__memories]


def set_seed(seed: int):
    random.seed(seed)


def get_memory(memory_name: str, **kwargs) -> Memory:
    return eval(memory_name)(**kwargs)

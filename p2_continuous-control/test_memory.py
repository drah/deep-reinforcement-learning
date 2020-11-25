import unittest
from memory import set_seed
from memory import ReplayBuffer


class MemoryTest(unittest.TestCase):
    def setUp(self):
        set_seed(0)

    def test_ReplayBuffer(self):
        mem = ReplayBuffer(2)
        mem.push(1)
        mem.push(2)
        sample = mem.sample(2)
        self.assertTrue(sample[0] in [1, 2])
        self.assertTrue(sample[1] in [1, 2])
        mem.push(3)
        sample = mem.sample(2)
        self.assertTrue(sample[0] in [2, 3])
        self.assertTrue(sample[1] in [2, 3])
        mem.push(4)
        sample = mem.sample(2)
        self.assertTrue(sample[0] in [3, 4])
        self.assertTrue(sample[1] in [3, 4])

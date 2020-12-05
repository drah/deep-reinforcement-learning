import unittest

import numpy as np

from memory import set_seed
from memory import ReplayBuffer
from memory import ReplayBufferNumpy


class MemoryTest(unittest.TestCase):
    def setUp(self):
        set_seed(0)

    def test_ReplayBuffer(self):
        mem = ReplayBuffer(2)
        mem.push(1)
        mem.push(2)
        [sample] = mem.sample(2)
        self.assertEqual(sorted(sample), [1, 2])
        mem.push(3)
        [sample] = mem.sample(2)
        self.assertEqual(sorted(sample), [2, 3])
        mem.push(4)
        [sample] = mem.sample(2)
        self.assertEqual(sorted(sample), [3, 4])

    def test_ReplayBufferNumpy(self):
        mem = ReplayBufferNumpy(2)
        mem.push(np.arange(3), 3)
        self.assertEqual(len(mem), 1)
        self.assertEqual(mem.get_memory_shapes(), [(2, 3), (2,)])
        sample = mem.sample(1)
        self.assertEqual(sample[0].shape, (1, 3))
        self.assertEqual(sample[1].shape, (1,))

        mem.push(np.arange(3, 6), 4)
        self.assertEqual(len(mem), 2)

        sample = mem.sample(2)
        if np.all(sample[0] == np.array([[0, 1, 2], [3, 4, 5]])):
            a_equal = 0
        elif np.all(sample[0] == np.array([[3, 4, 5], [0, 1, 2]])):
            a_equal = 1
        else:
            self.assertFalse("Not Equal")

        if np.all(sample[1] == np.array([3, 4])):
            b_equal = 0
        elif np.all(sample[1] == np.array([4, 3])):
            b_equal = 1
        else:
            self.assertFalse("Not Equal")
        
        self.assertEqual(a_equal, b_equal)

        mem.push(np.arange(6, 9), 5)
        self.assertEqual(len(mem), 2)

        mem.push(np.arange(9, 12), 6)
        self.assertEqual(len(mem), 2)

        sample = mem.sample(2)
        if np.all(sample[0] == np.array([[6, 7, 8], [9, 10, 11]])):
            a_equal = 0
        elif np.all(sample[0] == np.array([[9, 10, 11], [6, 7, 8]])):
            a_equal = 1
        else:
            self.assertFalse("Not Equal")

        if np.all(sample[1] == np.array([5, 6])):
            b_equal = 0
        elif np.all(sample[1] == np.array([6, 5])):
            b_equal = 1
        else:
            self.assertFalse("Not Equal")
        
        self.assertEqual(a_equal, b_equal)

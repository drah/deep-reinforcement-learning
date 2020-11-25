import unittest
import torch
import numpy as np

from networks import SimpleFC


class NetworkTest(unittest.TestCase):
    def test_SimpleFC(self):
        batch_size = 8
        state_size = 32
        action_size = 4

        net = SimpleFC(state_size, action_size)

        fake_states = np.random.uniform(-1, 1, batch_size * state_size).reshape([batch_size, state_size]).astype(np.float32)
        fake_states = torch.from_numpy(fake_states)
        actions = net.act(fake_states)
        self.assertEqual(actions.shape[0], batch_size)
        self.assertEqual(actions.shape[1], action_size)
        numpy_values = actions.data.numpy()
        self.assertTrue(np.all(numpy_values <= 1.))
        self.assertTrue(np.all(numpy_values >= -1.))
        
        loss = actions.sum()
        loss.backward()
        self.assertTrue(net.fc1.weight.grad is not None)
        net.zero_grad()
    
        clone = net.clone()
        for a, b in zip(net.parameters(), clone.parameters()):
            self.assertTrue(np.all(a.data.numpy() == b.data.numpy()))
        
        with torch.no_grad():
            for a in net.parameters():
                a.copy_(a + 1.)
        
        for a, b in zip(net.parameters(), clone.parameters()):
            self.assertTrue(np.all(a.data.numpy() != b.data.numpy()))

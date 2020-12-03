from abc import abstractmethod
from torch import nn
from torch import save
from torch import load

class Network(nn.Module):
    @abstractmethod
    def __init__(self, input_shapes: [[int]], output_shapes: [[int]], **kwargs):
        super().__init__()
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        self.kwargs = kwargs

    def forward(self, x):
        raise NotImplementedError("Don't call me.")

    def clone(self) -> 'type(self)':
        net = type(self)(self.input_shapes, self.output_shapes, **self.kwargs)
        net.load_state_dict(self.state_dict())
        return net

    def save(self, save_path: str):
        save(self.state_dict(), save_path)

    def load(self, load_path: str):
        state_dict = load(load_path)
        self.load_state_dict(state_dict)

class Actor(Network):
    @abstractmethod
    def __init__(self, input_shapes: [[int]], output_shapes: [[int]], **kwargs):
        super().__init__(input_shapes, output_shapes, **kwargs)

    def act(self, states):
        raise NotImplementedError("Don't call me.")

class Critic(Network):
    @abstractmethod
    def __init__(self, input_shapes: [[int]], output_shapes: [[int]], **kwargs):
        super().__init__(input_shapes, output_shapes, **kwargs)

    def score(self, states, actions):
        raise NotImplementedError("Don't call me.")

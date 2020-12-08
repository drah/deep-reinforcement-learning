import torch
import numpy as np

def make_tensor(values) -> torch.Tensor:
    if isinstance(values, np.ndarray) and values.dtype is np.float32:
        values = torch.from_numpy(values)
    elif not isinstance(values, torch.Tensor):
        values = torch.tensor(values, dtype=torch.float32)
    return values

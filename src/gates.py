import torch
import torch.nn as nn


class InputGate(nn.Module):
    def __init__(self, d_in, n_atoms):
        super().__init__()
        self.proj = nn.Linear(d_in, n_atoms)

    def forward(self, x):
        return torch.relu(self.proj(x))

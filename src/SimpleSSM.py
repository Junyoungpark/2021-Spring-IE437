import torch
import torch as nn


class SimpleSSM(nn.Module):
    """
    Simple State-Space Model
    """

    def __init__(self, state_dim, action_dim):
        super(SimpleSSM, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x, u):
        inp = torch.cat([x, u], dim=-1)
        return self.net(inp)

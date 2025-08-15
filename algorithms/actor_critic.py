
import torch
import torch.nn as nn
from torch.distributions import Normal

# ====== Plain 3-layer Actor / Critic ======
class Actor3(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, act=nn.ReLU):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3_mu  = nn.Linear(hidden, out_dim)
        self.fc3_std = nn.Linear(hidden, out_dim)
        self.activation = act()  # renamed to avoid collision with method

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        h = self.activation(self.fc1(x))
        h = self.activation(self.fc2(h))
        mu = self.fc3_mu(h)
        log_std = torch.clamp(self.fc3_std(h), -5, 1)  # keep std in a reasonable range
        std = torch.exp(log_std)
        return mu, std

    def dist(self, x):
        mu, std = self.forward(x)
        return Normal(mu, std)

    @torch.no_grad()
    def sample_action(self, x):
        """Sample action and return (action, logprob). Useful for inference."""
        d = self.dist(x)
        a = d.sample()
        logp = d.log_prob(a).sum(-1)
        return a, logp


class Critic3(nn.Module):
    def __init__(self, in_dim, hidden, act=nn.ReLU):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3_v = nn.Linear(hidden, 1)
        self.activation = act()  # renamed

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        h = self.activation(self.fc1(x))
        h = self.activation(self.fc2(h))
        v = self.fc3_v(h)
        return v.squeeze(-1)  # [N]
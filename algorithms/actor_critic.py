import torch
import torch.nn as nn

# ====== Plane 3-layer Actor / Critic ======
class Actor3(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, act=nn.ReLU):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3_mu  = nn.Linear(hidden, out_dim)
        self.fc3_std = nn.Linear(hidden, out_dim)
        self.act = act()

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.act(self.fc2(h))
        mu = self.fc3_mu(h)
        log_std = torch.clamp(self.fc3_std(h), -5, 1)
        std = torch.exp(log_std)
        return mu, std

class Critic3(nn.Module):
    def __init__(self, in_dim, hidden, act=nn.ReLU):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3_v = nn.Linear(hidden, 1)
        self.act = act()

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.act(self.fc2(h))
        v = self.fc3_v(h)
        return v.squeeze(-1)  # [N]
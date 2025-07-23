import torch
import torch.nn as nn
import numpy as np


# ====================== 1. Utils ======================
def make_linspace_points(start, end, steps, device=None, dtype=torch.float32):
    axes = [torch.linspace(s, e, steps, device=device, dtype=dtype)
            for s, e in zip(start, end)]
    mesh = torch.meshgrid(*axes, indexing="ij")
    pts = torch.stack([m.reshape(-1) for m in mesh], dim=1)
    return pts  # (steps**D, D)

def get_mean_frequency_2d(coeff_2d: torch.Tensor):
    """Get the mean frequency of the 2D Fourier coefficients."""

    dim_u, dim_v = coeff_2d.shape
    max_freq = round(((dim_u - 1) ** 2 + (dim_v - 1) ** 2) ** (1 / 2))

    grid_i, grid_j = torch.meshgrid(
        torch.Tensor(range(dim_u)), torch.Tensor(range(dim_v)), indexing="ij"
    )
    bins = torch.round(torch.sqrt(grid_i**2 + grid_j**2)).to(torch.int)

    coeff_1d = torch.zeros(max_freq + 1)

    for i in range(max_freq + 1):
        coeff_1d[i] = coeff_2d[bins == i].sum()

    # Normalize
    coeff_1d[0] = 0  # ignore shift coming from 0 "Hz" frequency.
    coeff_1d /= coeff_1d.sum()

    # Compute mean frequency
    mean_freq = (coeff_1d * torch.Tensor(range(max_freq + 1))).sum() / max_freq

    return mean_freq.item()

def simplicity_score(c_vals: torch.Tensor):
    return (1.0 / c_vals).mean().item()

def uniform_init_(module, w_amp, b_amp=None):
    if b_amp is None: b_amp = w_amp
    if isinstance(module, nn.Linear):
        nn.init.uniform_(module.weight, -w_amp, w_amp)
        if module.bias is not None:
            nn.init.uniform_(module.bias, -b_amp, b_amp)

def freeze_rsnorm(model):
    from types import MethodType
    for m in model.modules():
        if m.__class__.__name__ == "RSNorm":   # 직접 isinstance(RSNorm) 해도 됨
            def forward_no_update(self, x):
                squeezed = 0
                if x.dim() == 1:
                    x = x.unsqueeze(0); squeezed = 1
                y = (x - self.mu) / torch.sqrt(self.var + self.eps)
                return y.squeeze(0) if squeezed else y
            m.forward = MethodType(forward_no_update, m)


# ====================== 2. Core function ======================
def compute_simplicity_for_net(
    net: nn.Module,
    input_dim: int,
    scalar_extractor,          # callable: output_tensor(B,...) -> scalar(B)
    grid_range=(-100, 100),
    steps=256,
    amp=1.0,                   # amplitude
    n_samples=100,
    device=None
):
    """
    net      : nn.Module (Actor or Critic)
    scalar_extractor: function that maps model output to 1D tensor [N]
    """
    device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    # 2D grid + zeros for the rest dims
    start = (grid_range[0], grid_range[0])
    end   = (grid_range[1], grid_range[1])
    grid2 = make_linspace_points(start, end, steps, device=device)          # (steps^2, 2)
    if input_dim > 2:
        pad = torch.zeros((grid2.size(0), input_dim-2), device=device)
        full_in = torch.cat([grid2, pad], dim=1)
    else:
        full_in = grid2

    c_list = []

    for _ in range(n_samples):
        # re-init
        net.apply(lambda m: uniform_init_(m, amp, amp))
        freeze_rsnorm(net)

        net.eval()
        with torch.no_grad():
            out = net(full_in)                  # arbitrary shape
            scalar = scalar_extractor(out)      # [steps^2]
            scalar = scalar.view(steps, steps)

        coeff_2d = torch.fft.rfft2(scalar).abs()
        coeff_2d[0, 0] = 0
        coeff_2d = coeff_2d[: coeff_2d.shape[0] // 2 + 1]

        c_val = get_mean_frequency_2d(coeff_2d)
        c_list.append(c_val)

    c_tensor = torch.tensor(c_list, dtype=torch.float32, device=device)
    return simplicity_score(c_tensor), c_tensor


# ====================== 3. PPOAgent Wrapper ======================
def compute_actor_critic_simplicity(PPOAgentCls,
                                    agent_kwargs,
                                    actor_output_idx=0,
                                    grid_range=(-1, 1),
                                    steps=100,
                                    amp=1.0,
                                    n_samples=100,
                                    device=None):
    """
    PPOAgentCls : simba.PPOAgent Class
    agent_kwargs: PPOAgent initialize factor dict
    actor_output_idx: dimension of mu from actor
    """
    device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    agent = PPOAgentCls(**agent_kwargs)

    actor = agent.actor if hasattr(agent, "actor") else agent.policy.actor
    critic = agent.critic if hasattr(agent, "critic") else agent.policy.critic

    input_dim_actor  = agent_kwargs["input_dim"] if "input_dim" in agent_kwargs else actor.input_fc.in_features
    input_dim_critic = agent_kwargs["input_dim"] if "input_dim" in agent_kwargs else critic.input_fc.in_features

    # --- Actor ---
    def actor_scalar_extractor(out):
        # out: (mu, std)
        mu = out[0] if isinstance(out, (tuple, list)) else out
        return mu[:, actor_output_idx]

    simp_actor, cs_actor = compute_simplicity_for_net(
        actor, input_dim_actor, actor_scalar_extractor,
        grid_range=grid_range, steps=steps, amp=amp,
        n_samples=n_samples, device=device
    )

    # --- Critic ---
    def critic_scalar_extractor(out):
        # out: [N,1]
        return out.view(-1)

    simp_critic, cs_critic = compute_simplicity_for_net(
        critic, input_dim_critic, critic_scalar_extractor,
        grid_range=grid_range, steps=steps, amp=amp,
        n_samples=n_samples, device=device
    )

    return {
        "simplicity_actor": simp_actor,
        "simplicity_critic": simp_critic,
        "c_values_actor": cs_actor,     # tensor of c(fθ)
        "c_values_critic": cs_critic
    }


# ====== main1 ======
def main1():
    from algorithms.simba import PPOAgent
    from environment.env import MiniSystem

    system = MiniSystem(
        user_num=2,
        RIS_ant_num=4,
        UAV_ant_num=4,
        if_dir_link=1,
        if_with_RIS=True,
        if_move_users=True,
        if_movements=True,
        reverse_x_y=(False, False),
        if_UAV_pos_state=True,
        reward_design='see',
        project_name='SIMBA/simba_see',
        step_num=100
    )

    agent_kwargs = dict(
        alpha=3e-4, beta=3e-3,
        input_dim=system.get_system_state_dim(),
        n_action=system.get_system_action_dim() - 2,
        lamda=0.95, gamma=0.99, eps_clip=0.2,
        layer_size=128, batch_size=100, K_epochs=10,
        noise='AWGN', num_block1=2, num_block2=4
    )

    res = compute_actor_critic_simplicity(
        PPOAgentCls=PPOAgent,
        agent_kwargs=agent_kwargs,
        actor_output_idx=0,
        grid_range=(-1, 1),
        steps=300,
        amp=1.0,
        n_samples=100
    )

    print("SimBa Actor simplicity :", res["simplicity_actor"])
    print("SimBa Critic simplicity:", res["simplicity_critic"])
    # print("SimBa Actor complexity :", res["c_values_actor"])
    # print("SimBa Critic complexity:", res["c_values_critic"])


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


# ====== main2 ======
def main2():
    # Hyperparameters
    in_dim   = 27
    hidden   = 128
    n_action = 20
    steps    = 100
    amp      = 1.0
    n_samples = 100
    grid_range = (-1, 1)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize Actor/ Critic
    actor = Actor3(in_dim=in_dim, hidden=hidden, out_dim=n_action).to(device)
    critic = Critic3(in_dim=in_dim, hidden=hidden).to(device)

    # --- Actor ---
    def actor_scalar_extractor(out):
        mu = out[0]  # (mu, std)
        return mu[:, 0]

    simp_actor, c_actor = compute_simplicity_for_net(
        net=actor,
        input_dim=in_dim,
        scalar_extractor=actor_scalar_extractor,
        grid_range=grid_range,
        steps=steps,
        amp=amp,
        n_samples=n_samples,
        device=device
    )

    # --- Critic ---
    def critic_scalar_extractor(out):
        return out.view(-1)  # [N]

    simp_critic, c_critic = compute_simplicity_for_net(
        net=critic,
        input_dim=in_dim,
        scalar_extractor=critic_scalar_extractor,
        grid_range=grid_range,
        steps=steps,
        amp=amp,
        n_samples=n_samples,
        device=device
    )

    print(f"Plane Actor simplicity : {simp_actor}")
    print(f"Plane Critic simplicity: {simp_critic}")
    # print("Plane Actor complexity:", c_actor.mean().item())
    # print("Plane Critic complexity:", c_critic.mean().item())



if __name__ == "__main__":
    main1()
    main2()
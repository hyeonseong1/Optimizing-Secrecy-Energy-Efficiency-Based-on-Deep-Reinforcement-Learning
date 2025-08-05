import torch

from algorithms.actor_critic import Actor3, Critic3
from metrics.compute_simpl.core_func import compute_simplicity_for_net


# ====== compute_simple_actor_critic ======
def compute_simpl_AC():
    # Hyperparameters
    in_dim   = 27   # 3
    hidden   = 128
    n_action = 20   # 2
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

    return {
        "simplicity_actor": simp_actor,
        "simplicity_critic": simp_critic,
        "c_values_actor": c_actor,
        "c_values_critic": c_critic,
    }
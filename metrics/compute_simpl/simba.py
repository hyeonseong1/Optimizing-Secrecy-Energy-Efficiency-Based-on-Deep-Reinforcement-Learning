import torch

from algorithms.simba import PPOAgent
from metrics.compute_simpl.core_func import compute_simplicity_for_net


# ====== compute_simpl_simba ======
def compute_simpl_simba():
    # Hyperparameters
    in_dim = 27           # 입력 차원
    hidden = 128          # layer_size
    n_action = 20         # action 차원
    steps = 300
    amp = 1.0
    n_samples = 100
    grid_range = (-100, 100)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize SimBa PPOAgent
    agent_kwargs = dict(
        alpha=3e-4,
        beta=3e-3,
        input_dim=in_dim,
        n_action=n_action,
        lamda=0.95,
        gamma=0.99,
        eps_clip=0.2,
        layer_size=hidden,
        batch_size=100,
        K_epochs=10,
        num_block1=2,
        num_block2=4
    )
    agent = PPOAgent(**agent_kwargs)

    actor = agent.actor if hasattr(agent, "actor") else agent.policy.actor
    critic = agent.critic if hasattr(agent, "critic") else agent.policy.critic

    actor = actor.to(device)
    critic = critic.to(device)

    # --- Actor ---
    def actor_scalar_extractor(out):
        mu = out[0] if isinstance(out, (tuple, list)) else out
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
        return out.view(-1)

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

    print(f"SimBa Actor simplicity : {simp_actor}")
    print(f"SimBa Critic simplicity: {simp_critic}")
    # print("SimBa Actor complexity :", c_actor.mean().item())
    # print("SimBa Critic complexity:", c_critic.mean().item())

    return {
        "simplicity_actor": simp_actor,
        "simplicity_critic": simp_critic,
        "c_values_actor": c_actor,
        "c_values_critic": c_critic,
    }
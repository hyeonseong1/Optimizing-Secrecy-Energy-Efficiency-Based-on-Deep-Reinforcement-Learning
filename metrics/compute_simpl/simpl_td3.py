import torch
from metrics.compute_simpl.core_func import compute_simplicity_for_net
from algorithms.td3 import ActorNetwork, CriticNetwork


class CriticStateWrapper(torch.nn.Module):
    """Fixed actions for CriticNetwork"""
    def __init__(self, critic: CriticNetwork, fixed_action: torch.Tensor):
        super().__init__()
        self.critic = critic
        # fixed_action: (1, n_actions)
        self.register_buffer("fixed_action", fixed_action)

    def forward(self, state):
        batch_size = state.shape[0]
        action = self.fixed_action.expand(batch_size, -1)
        return self.critic(state, action)


def compute_simpl_td3(
    in_dim=27,
    n_actions=2,
    layer1_size=400,
    layer2_size=300,
    layer3_size=256,
    layer4_size=128,
    alpha=1e-3,
    beta=1e-3,
    steps=100,
    amp=1.0,
    n_samples=100,
    grid_range=(-100, 100),
    device=None
):
    # device 설정
    device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_dims = in_dim

    # init Network
    actor = ActorNetwork(
        alpha=alpha,
        input_dims=input_dims,
        fc1_dims=layer1_size,
        fc2_dims=layer2_size,
        fc3_dims=layer3_size,
        fc4_dims=layer4_size,
        n_actions=n_actions,
        name='Actor_TD3'
    ).to(device)

    critic1 = CriticNetwork(
        beta=beta,
        input_dims=input_dims,
        fc1_dims=layer1_size,
        fc2_dims=layer2_size,
        fc3_dims=layer3_size,
        fc4_dims=layer4_size,
        n_actions=n_actions,
        name='Critic1_TD3'
    ).to(device)

    critic2 = CriticNetwork(
        beta=beta,
        input_dims=input_dims,
        fc1_dims=layer1_size,
        fc2_dims=layer2_size,
        fc3_dims=layer3_size,
        fc4_dims=layer4_size,
        n_actions=n_actions,
        name='Critic2_TD3'
    ).to(device)

    # Fixed action (zeros)
    fixed_action = torch.zeros((1, n_actions), device=device)

    # wrapper for state-only critics
    critic1_state = CriticStateWrapper(critic1, fixed_action).to(device)
    critic2_state = CriticStateWrapper(critic2, fixed_action).to(device)

    # --- Actor scalar extractor ---
    def actor_scalar_extractor(out):
        return out[:, 0]

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

    # --- Critic1 (state only) ---
    def critic_scalar_extractor(out):
        return out.view(-1)

    simp_critic1, c_critic1 = compute_simplicity_for_net(
        net=critic1_state,
        input_dim=in_dim,
        scalar_extractor=critic_scalar_extractor,
        grid_range=grid_range,
        steps=steps,
        amp=amp,
        n_samples=n_samples,
        device=device
    )

    # --- Critic2 (state only) ---
    simp_critic2, c_critic2 = compute_simplicity_for_net(
        net=critic2_state,
        input_dim=in_dim,
        scalar_extractor=critic_scalar_extractor,
        grid_range=grid_range,
        steps=steps,
        amp=amp,
        n_samples=n_samples,
        device=device
    )

    print(f"TD3 Actor simplicity : {simp_actor}")
    print(f"TD3 Critic1 simplicity: {simp_critic1}")
    print(f"TD3 Critic2 simplicity: {simp_critic2}")
    # print("TD3 Actor complexity :", c_actor.mean().item())
    # print("TD3 Critic1 complexity:", c_critic1.mean().item())
    # print("TD3 Critic2 complexity:", c_critic2.mean().item())

    return {
        "simplicity_actor": simp_actor,
        "simplicity_critic1": simp_critic1,
        "simplicity_critic2": simp_critic2,
        "c_values_actor": c_actor,
        "c_values_critic1": c_critic1,
        "c_values_critic2": c_critic2
    }
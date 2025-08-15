
import torch
from metrics.compute_simpl.core_func import compute_simplicity_for_net
from algorithms.ddpg import ActorNetwork, CriticNetwork


class CriticStateWrapper(torch.nn.Module):
    """Wrap CriticNetwork(state, action) -> Q(state, action) to a state-only module
    by feeding a fixed action. This lets us reuse compute_simplicity_for_net which
    expects a mapping f: R^{in_dim} -> R.
    """
    def __init__(self, critic: CriticNetwork, fixed_action: torch.Tensor):
        super().__init__()
        self.critic = critic
        # fixed_action: (1, n_actions)
        self.register_buffer("fixed_action", fixed_action)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]
        action = self.fixed_action.expand(batch, -1)
        return self.critic(state, action)


def compute_simpl_ddpg(
    in_dim: int = 27,
    n_actions: int = 2,
    layer1_size: int = 400,
    layer2_size: int = 300,
    layer3_size: int = 256,
    layer4_size: int = 128,
    alpha: float = 1e-3,
    beta: float = 1e-3,
    steps: int = 100,
    amp: float = 1.0,
    n_samples: int = 100,
    grid_range = (-100, 100),
    device: torch.device = None
):
    """Compute simplicity for a DDPG actor/critic, following td3.py style.

    Returns
    -------
    dict with keys:
      - simplicity_actor, simplicity_critic
      - c_values_actor, c_values_critic (raw complexity grid values)
    """
    device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_dims = (in_dim,)

    # Initialize DDPG networks (random weights)
    actor = ActorNetwork(
        alpha=alpha,
        input_dims=input_dims,
        fc1_dims=layer1_size,
        fc2_dims=layer2_size,
        fc3_dims=layer3_size,
        fc4_dims=layer4_size,
        n_actions=n_actions,
        name='Actor_DDPG'
    ).to(device)

    critic = CriticNetwork(
        beta=beta,
        input_dims=input_dims,
        fc1_dims=layer1_size,
        fc2_dims=layer2_size,
        fc3_dims=layer3_size,
        fc4_dims=layer4_size,
        n_actions=n_actions,
        name='Critic_DDPG'
    ).to(device)

    # Fix an action for the critic so we can treat it as a scalar-valued map of state.
    fixed_action = torch.zeros((1, n_actions), device=device)
    critic_state = CriticStateWrapper(critic, fixed_action).to(device)

    # --- Actor simplicity (use first action dim as scalar) ---
    def actor_scalar_extractor(out: torch.Tensor) -> torch.Tensor:
        # out shape: [N, n_actions] -> pick dim 0 as representative
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

    # --- Critic simplicity (state-only wrapper) ---
    def critic_scalar_extractor(out: torch.Tensor) -> torch.Tensor:
        # CriticStateWrapper returns [N, 1] -> flatten to [N]
        return out.view(-1)

    simp_critic, c_critic = compute_simplicity_for_net(
        net=critic_state,
        input_dim=in_dim,
        scalar_extractor=critic_scalar_extractor,
        grid_range=grid_range,
        steps=steps,
        amp=amp,
        n_samples=n_samples,
        device=device
    )

    print(f"DDPG Actor simplicity : {simp_actor}")
    print(f"DDPG Critic simplicity: {simp_critic}")

    return {
        "simplicity_actor": simp_actor,
        "simplicity_critic": simp_critic,
        "c_values_actor": c_actor,
        "c_values_critic": c_critic,
    }


if __name__ == "__main__":
    compute_simpl_ddpg()
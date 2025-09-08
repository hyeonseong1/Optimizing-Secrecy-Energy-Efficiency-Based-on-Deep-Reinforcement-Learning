import torch
from metrics.compute_simpl.core_func import compute_simplicity_for_net
from algorithms.ppo import PPOAgent

# ====== compute_simpl_ppo ======
def compute_simpl_ppo():
    # Hyperparameters (SimBa 포맷과 동일 키/값 구조 유지)
    in_dim    = 27            # 입력 차원
    hidden    = 128           # layer_size
    n_action  = 20            # action 차원
    steps     = 300
    amp       = 1.0
    n_samples = 100
    grid_range = (-100, 100)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize PPOAgent (두 번째 코드의 클래스 사용)
    agent_kwargs = dict(
        alpha=3e-4,          # actor lr
        beta=3e-3,           # critic lr
        input_dim=in_dim,
        n_action=n_action,
        lamda=0.95,
        gamma=0.99,
        eps_clip=0.2,
        layer1_size=hidden,
        layer2_size=hidden,
        batch_size=100,
        K_epochs=10,
        noise='AWGN',        # 또는 'OU'
    )
    agent = PPOAgent(**agent_kwargs)

    actor = agent.actor
    critic = agent.critic

    actor = actor.to(device)
    critic = critic.to(device)

    # --- Actor ---
    # PPOActor.forward -> (mu, std) 이므로 SimBa와 동일하게 첫 번째 차원만 취해 스칼라화
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

    print(f"PPO Actor simplicity : {simp_actor}")
    print(f"PPO Critic simplicity: {simp_critic}")
    # print("PPO Actor complexity :", c_actor.mean().item())
    # print("PPO Critic complexity:", c_critic.mean().item())

    return {
        "simplicity_actor": simp_actor,
        "simplicity_critic": simp_critic,
        "c_values_actor": c_actor,
        "c_values_critic": c_critic,
    }

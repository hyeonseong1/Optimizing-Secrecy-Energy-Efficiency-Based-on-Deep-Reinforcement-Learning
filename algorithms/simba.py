import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)

class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
            self.mu, self.sigma)


class AWGNActionNoise(object):
    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        x = np.random.normal(size=self.mu.shape) * self.sigma
        return x

import numpy as np

class RolloutBuffer(object):
    def __init__(self):
        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []
        self.terminals = []
        self.buffer_cnt = 0

    def store_transition(self, state, action, reward, state_, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(state_)
        self.terminals.append(1 - done)
        self.buffer_cnt += 1

    def sample_buffer(self, batch_size):
        if self.buffer_cnt < batch_size:
            print(f"[Warning] Not enough samples in buffer: {self.buffer_cnt}/{batch_size}")
            return None

        batch = np.random.choice(self.buffer_cnt, batch_size)

        states = np.array(self.states, dtype=np.float32)[batch]
        actions = np.array(self.actions, dtype=np.float32)[batch]
        rewards = np.array(self.rewards, dtype=np.float32)[batch]
        next_states = np.array(self.next_states, dtype=np.float32)[batch]
        terminals = np.array(self.terminals, dtype=np.float32)[batch]

        return states, actions, rewards, next_states, terminals

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.terminals.clear()
        self.buffer_cnt = 0

class RSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.register_buffer('mu',    torch.zeros(dim))
        self.register_buffer('var',   torch.zeros(dim))
        self.register_buffer('count', torch.zeros(1, dtype=torch.long))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D]
        squeezed = 0
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeezed = 1
        B, D = x.shape
        for i in range(B):
            self.count += 1
            delta = x[i].detach() - self.mu
            t = self.count.float()
            self.mu  += delta / t
            self.var  = torch.clamp((t - 1) / t * (self.var + delta.pow(2) / t), min=0.001)
        y = (x - self.mu) / torch.sqrt(self.var + self.eps)
        if squeezed:
            y = y.squeeze(0)
        return y

class ResidualFFBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.ln  = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.fc2(h)
        return x + h

class Actor(nn.Module):
    def __init__(self,
                 alpha,
                 input_dim,
                 hidden_dim,
                 n_action,
                 num_block):
        super(Actor, self).__init__()
        self.rsnorm = RSNorm(input_dim)
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()

        self.blocks = nn.ModuleList([
            ResidualFFBlock(hidden_dim)
            for _ in range(num_block)
        ])

        self.post_ln = nn.LayerNorm(hidden_dim)
        self.mu = nn.Linear(hidden_dim, n_action)
        self.std = nn.Linear(hidden_dim, n_action)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.rsnorm(state)
        x = self.input_fc(x)
        for blk in self.blocks:  # N × residual block
            x = blk(x)
        x = self.post_ln(x)  # final LayerNorm

        mu = self.mu(x)
        mu = torch.clamp(mu, -5, 5)

        log_std = self.std(x)
        log_std = torch.clamp(log_std, -5, 1)
        std = torch.exp(log_std)
        return mu, std # UAV coordination(3) or RIS beamforming(20)


class Critic(nn.Module):
    def __init__(self,
                 beta,
                 input_dim,
                 hidden_dim,
                 num_block):
        super().__init__()
        self.rsnorm = RSNorm(input_dim)
        self.input_fc = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            ResidualFFBlock(hidden_dim)
            for _ in range(num_block)
        ])

        self.post_ln = nn.LayerNorm(hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.rsnorm(obs)           # RSNorm
        x = self.input_fc(x)           # Linear(input→hidden1)
        for blk in self.blocks:        # N × residual block
            x = blk(x)
        x = self.post_ln(x)            # final LayerNorm
        v = self.value_head(x)         # value estimate [B,1]
        return v


class PPOAgent(object):
    def __init__(self,
                 alpha,
                 beta,
                 input_dim,
                 n_action,
                 lamda=0.95,
                 gamma=0.99,
                 eps_clip=0.2,
                 layer_size=256,
                 batch_size=64,
                 K_epochs=5,
                 noise='AWGN',
                 num_block1=2,
                 num_block2=4):
        self.input_dim = input_dim
        self.lamda = lamda
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.n_action = n_action
        self.batch_size = batch_size
        self.K_epochs = K_epochs
        self.noise_type = noise
        self.num_block1 = num_block1
        self.num_block2 = num_block2

        self.buffer = RolloutBuffer()

        # alpha for actor, beta for critic; learning rate hyperparameter
        self.actor = Actor(alpha, input_dim, layer_size, n_action, num_block1)
        self.critic = Critic(beta, input_dim, layer_size, num_block2)
        self.old_actor = copy.deepcopy(self.actor)

        if noise == 'OU':
            self.noise = OUActionNoise(mu=np.zeros(n_action))
        elif noise == 'AWGN':
            self.noise = AWGNActionNoise(mu=np.zeros(n_action))

    def act(self, state, greedy=0.5):
        state = torch.tensor(state, dtype=torch.float32).to(self.actor.device)
        # print(state)

        with torch.no_grad():
            mu, std = self.actor(state)
            # print(f'mu:{mu}, std:{std}')

            # Apply noise to mean if exploration is desired
            if greedy > 0:
                noise_tensor = torch.tensor(greedy * self.noise(), dtype=torch.float32).to(self.actor.device)
                mu = mu + noise_tensor

            # Create distribution and sample
            dist = torch.distributions.Normal(mu, std)
            actions_raw = dist.sample()
            actions = torch.tanh(actions_raw)  # Bound actions to [-1, 1]

        return actions.detach().cpu().numpy()

    def learn(self):
        # Get data from buffer
        sample = self.buffer.sample_buffer(self.batch_size)
        if sample is None:
            return  # Not enough data to train

        states, actions, rewards, states_, dones = sample

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.actor.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.actor.device)
        next_states = torch.tensor(states_, dtype=torch.float32).to(self.actor.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.actor.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.actor.device)
        done_tensor = torch.ones_like(dones)

        # Calculate advantages and returns
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)

            # TD errors
            deltas = rewards + self.gamma * next_values * done_tensor - values

            # Calculate GAE
            advantages = torch.zeros_like(deltas)
            gae = 0
            for t in reversed(range(len(rewards))):
                if done_tensor[t]:
                    gae = deltas[t]
                else:
                    gae = deltas[t] + self.gamma * self.lamda * gae
                advantages[t] = gae

            # Calculate returns for critic update
            returns = advantages + values

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
            advantages = torch.clamp(advantages, -3, 3)

            # Get old policy distribution parameters
            mu_old, std_old = self.old_actor(states)

            # Calculate log probabilities of old actions

            old_dist = torch.distributions.Normal(mu_old, std_old)
            # Use arctanh with numerical stability
            arctanh_actions = 0.5 * torch.log((1 + actions + 1e-6) / (1 - actions + 1e-6))
            old_log_probs = old_dist.log_prob(arctanh_actions).sum(1, keepdim=True)

        # PPO Update loop
        for _ in range(self.K_epochs):
            # Get current policy distribution parameters
            mu, std = self.actor(states)

            # Calculate log probabilities of actions under current policy
            current_dist = torch.distributions.Normal(mu, std)
            arctanh_actions = 0.5 * torch.log((1 + actions + 1e-6) / (1 - actions + 1e-6))
            current_log_probs = current_dist.log_prob(arctanh_actions).sum(1, keepdim=True)

            # Calculate policy ratio
            ratios = torch.exp(current_log_probs - old_log_probs)

            # Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Calculate actor loss with KL penalty
            kl_div = torch.distributions.kl_divergence(
                torch.distributions.Normal(mu_old, std_old),
                torch.distributions.Normal(mu, std)
            ).mean()

            actor_loss = -torch.min(surr1, surr2).mean() + 0.01 * kl_div

            # Update actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10)
            self.actor.optimizer.step()

            # Calculate critic loss
            critic_value = self.critic(states)
            critic_loss = F.mse_loss(critic_value, returns)

            # Update critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10)
            self.critic.optimizer.step()

        # Update old actor for next iteration
        self.old_actor = copy.deepcopy(self.actor)

        # Clear buffer after update
        self.buffer.clear()

    def store_transition(self, state, action, reward, state_, done):
        self.buffer.store_transition(state, action, reward, state_, done)
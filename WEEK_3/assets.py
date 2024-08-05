import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import Tuple, List

# 정책 신경망 (Policy Network)
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, int] = (64, 64), activation_fn: nn.Module = F.tanh):
        super(PolicyNetwork, self).__init__()
        self.input_layer = nn.Linear(state_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.mu_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.activation_fn = activation_fn

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.activation_fn(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation_fn(layer(x))
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x).clamp(-20, 2)
        std = log_std.exp()
        return mu, std

# 가치 신경망 (Value Network)
class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dims: Tuple[int, int] = (64, 64), activation_fn: nn.Module = F.tanh):
        super(ValueNetwork, self).__init__()
        self.input_layer = nn.Linear(state_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        self.activation_fn = activation_fn
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation_fn(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation_fn(layer(x))
        x = self.output_layer(x)
        return x

# 경험 버퍼 (Experience Buffer)
# 지난번에 보았던 Buffer와 같은 기능을 합니다! 

class ExperienceBuffer:
    def __init__(self):
        self.buffer: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = []

    def store(self, transition: Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]) -> None:
        self.buffer.append(transition)

    def sample(self) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        s, a, r, s_prime, done = map(np.array, zip(*self.buffer))
        self.buffer.clear()
        return (
            torch.FloatTensor(s),
            torch.FloatTensor(a),
            torch.FloatTensor(r).unsqueeze(1),
            torch.FloatTensor(s_prime),
            torch.FloatTensor(done).unsqueeze(1)
        )


    @property
    def size(self) -> int:
        return len(self.buffer)

# PPO 알고리즘 (PPO Algorithm)
# PPO의 장점은 TRPO와 다르게 간단하게 clipping으로 구현이 가능하다는 점에 있습니다. 아래를 잘보고 빈칸을 잘 채워주세요! 
class PPO:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, int] = (64, 64),
        activation_fn: nn.Module = F.relu,
        n_steps: int = 2048,
        n_epochs: int = 10,
        batch_size: int = 64,
        policy_lr: float = 0.0003,
        value_lr: float = 0.0003,
        gamma: float = 0.99,
        lmda: float = 0.95,
        clip_ratio: float = 0.2,
        vf_coef: float = 1.0,
        ent_coef: float = 0.01,
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dims, activation_fn).to(self.device)
        self.value = ValueNetwork(state_dim, hidden_dims, activation_fn).to(self.device)
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lmda = lmda
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

        self.policy_optimizer = torch.optim.AdamW(self.policy.parameters(), lr=policy_lr)
        self.value_optimizer = torch.optim.AdamW(self.value.parameters(), lr=value_lr)
        
        self.buffer = ExperienceBuffer()

    @torch.no_grad()
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[np.ndarray, torch.Tensor]:
        self.policy.train(training)
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        mu, std = self.policy(state)
        dist = Normal(mu, std)
        z = dist.sample() if training else mu
        action = torch.tanh(z)
        return action.cpu().numpy(), dist.log_prob(z).sum(dim=-1, keepdim=True)

    def update(self) -> None:
        self.policy.train()
        self.value.train()
        states, actions, rewards, next_states, dones = self.buffer.sample()
        states, actions, rewards, next_states, dones = map(lambda x: x.to(self.device), [states, actions, rewards, next_states, dones])
        
        with torch.no_grad():
            deltas = rewards + (1 - dones) * self.gamma * self.value(next_states) - self.value(states)
            advantages = torch.zeros_like(deltas).to(self.device)
            returns = torch.zeros_like(rewards).to(self.device)
            acc_advantage = 0
            acc_return = 0
            for t in reversed(range(len(rewards))):
                acc_advantage = deltas[t] + self.gamma * self.lmda * acc_advantage * (1 - dones[t])
                acc_return = rewards[t] + self.gamma * acc_return * (1 - dones[t])
                advantages[t] = acc_advantage
                returns[t] = acc_return
            
            mu, std = self.policy(states)
            dist = Normal(mu, std)
            log_prob_old = dist.log_prob(torch.atanh(actions)).sum(dim=-1, keepdim=True)
        
        dataset = TensorDataset(states, actions, returns, advantages, log_prob_old)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for _ in range(self.n_epochs):
            for batch in loader:
                s_batch, a_batch, r_batch, adv_batch, log_prob_old_batch = batch
                    
                value_loss = F.mse_loss(self.value(s_batch), r_batch)
                mu, std = self.policy(s_batch)
                dist = Normal(mu, std)
                log_prob = dist.log_prob(torch.atanh(a_batch)).sum(dim=-1, keepdim=True)
                ratio = (log_prob - log_prob_old_batch).exp()

                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_bonus = dist.entropy().mean()

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_bonus

                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                self.policy_optimizer.step()
                self.value_optimizer.step()

    def step(self, transition: Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]) -> None:
        self.buffer.store(transition)
        if self.buffer.size >= self.n_steps:
            self.update()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

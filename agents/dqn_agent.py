"""
Deep Q-Network (DQN) agent for Spot Instance optimization.

Implements Double DQN + Dueling Network + Prioritized Experience Replay:
- Double DQN: dùng online network chọn action, target network đánh giá value
  → giảm overestimation bias, agent học terminate khi thực sự cần
- Dueling Network: tách Value vs Advantage stream
  → học V(s) tốt hơn, đặc biệt khi nhiều action tương đương (DO_NOTHING)
- Prioritized Experience Replay (PER): ưu tiên sample transition có TD-error cao
  → học nhanh hơn từ những tình huống khó (spike, interrupt, overprov)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple
from collections import deque

from agents.networks import DuelingQNetwork, FactoredDuelingQNetwork
from agents.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


def _build_qnet(network_type: str, state_dim: int, action_dim: int, hidden_dim: int):
    """Factory: build Q-network by config.

    network_type:
      - "dueling":  flat DuelingQNetwork [state → action_dim]
      - "factored": FactoredDuelingQNetwork [state → (op + pool + hold) → action_dim]
    """
    if network_type == "factored":
        from envs.action_schema import N_POOL_OPS, N_POOLS
        expected = N_POOL_OPS * N_POOLS + 1
        assert action_dim == expected, \
            f"Factored net requires action_dim={expected}, got {action_dim}"
        return FactoredDuelingQNetwork(
            state_dim=state_dim, n_ops=N_POOL_OPS, n_pools=N_POOLS, hidden_dim=hidden_dim,
        )
    elif network_type == "dueling":
        return DuelingQNetwork(state_dim, action_dim, hidden_dim=hidden_dim)
    else:
        raise ValueError(f"Unknown network_type={network_type!r} (use 'dueling' or 'factored')")


class DQNAgent:
    """
    Double DQN + Dueling Network + Prioritized Experience Replay agent.

    Improvements over vanilla DQN:
    - Double DQN: giảm overestimation → agent học terminate đúng thời điểm
    - PER: sample ưu tiên transition khó → hội tụ nhanh hơn
    - use_per=False: fallback về uniform ReplayBuffer (cho compatibility)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 30000,
        batch_size: int = 512,
        buffer_size: int = 200000,
        target_update_freq: int = 500,
        hidden_dim: int = 512,
        use_per: bool = True,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        network_type: str = "dueling",   # "dueling" (flat) | "factored" (op × pool)
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device
        self.use_per = use_per
        self.network_type = network_type

        # Epsilon-greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        self.hidden_dim = hidden_dim

        # Networks — build via factory (supports flat Dueling or Factored Dueling)
        self.q_network = _build_qnet(network_type, state_dim, action_dim, hidden_dim).to(device)
        self.target_network = _build_qnet(network_type, state_dim, action_dim, hidden_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer — AdamW tốt hơn Adam về generalization (weight decay nhỏ)
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=learning_rate, weight_decay=1e-5)

        # LR scheduler — giảm LR khi training ổn định
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50000, gamma=0.5)

        # Replay buffer — PER mặc định
        if use_per:
            self.replay_buffer = PrioritizedReplayBuffer(
                buffer_size,
                alpha=per_alpha,
                beta_start=per_beta_start,
                beta_frames=epsilon_decay * 2,
            )
        else:
            self.replay_buffer = ReplayBuffer(buffer_size)

        # RNG for exploration
        self.rng = np.random.default_rng(np.random.randint(0, 2**31))

        # Metrics
        self.loss_history = deque(maxlen=100)

    def select_action(
        self,
        state: np.ndarray,
        training: bool = True,
        action_mask: np.ndarray = None,
    ) -> int:
        """
        Select action using epsilon-greedy policy with optional action masking.

        Args:
            state: Current state
            training: If False, always exploit (no exploration)
            action_mask: Boolean array [action_dim] — True = valid action.
                         If None, all actions are valid.

        Returns:
            Selected action (int)
        """
        # Build valid action list from mask
        if action_mask is not None:
            valid_actions = np.where(action_mask)[0]
            if len(valid_actions) == 0:
                valid_actions = np.arange(self.action_dim)  # fallback
        else:
            valid_actions = None

        if training and self.rng.random() < self.epsilon:
            # Explore: random valid action
            if valid_actions is not None:
                return int(self.rng.choice(valid_actions))
            return int(self.rng.integers(self.action_dim))
        else:
            # Exploit: best valid action according to Q-network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor).squeeze(0)  # [action_dim]

                if action_mask is not None:
                    # Mask invalid actions với -inf
                    mask_tensor = torch.BoolTensor(action_mask).to(self.device)
                    q_values = q_values.masked_fill(~mask_tensor, float('-inf'))

                return q_values.argmax().item()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Store transition in replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.replay_buffer.add(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        """
        Perform one training step using Double DQN + PER.

        Double DQN: online net chọn action, target net đánh giá value
        → tránh overestimate Q(s', a') → agent không giữ instance vô ích

        Returns:
            Loss value, or None if buffer not ready
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Ensure training mode
        self.q_network.train()

        # Sample batch — PER trả thêm indices và IS weights
        if self.use_per:
            states, actions, rewards, next_states, dones, indices, is_weights = \
                self.replay_buffer.sample(self.batch_size)
            is_weights_tensor = torch.FloatTensor(is_weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones = \
                self.replay_buffer.sample(self.batch_size)
            is_weights_tensor = None

        # Convert to tensors
        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)

        # Current Q-values: Q(s, a) từ online network
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target:
        # 1. Online net chọn next action: a* = argmax_a Q_online(s', a)
        # 2. Target net đánh giá: Q_target(s', a*)
        with torch.no_grad():
            # Bước 1: online net chọn action tốt nhất ở next_state
            next_actions = self.q_network(next_states).argmax(dim=1)
            # Bước 2: target net đánh giá Q-value của action đó
            next_q_values = self.target_network(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # TD errors để update PER priorities
        td_errors = (current_q_values - target_q_values).detach().cpu().numpy()

        # Huber loss với IS weights (PER correction)
        element_loss = nn.SmoothL1Loss(reduction='none')(current_q_values, target_q_values)
        if is_weights_tensor is not None:
            loss = (element_loss * is_weights_tensor).mean()
        else:
            loss = element_loss.mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        # Update PER priorities
        if self.use_per:
            self.replay_buffer.update_priorities(indices, np.abs(td_errors))

        # Update epsilon
        self.steps_done += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            np.exp(-self.steps_done / self.epsilon_decay)

        # Update target network
        if self.steps_done % self.target_update_freq == 0:
            self.update_target_network()

        # Log loss
        self.loss_history.append(loss.item())

        return loss.item()

    def update_target_network(self):
        """Sync target network with Q-network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'use_per': self.use_per,
            'network_type': self.network_type,
            'hidden_dim': self.hidden_dim,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
        }, path)

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']

    def get_metrics(self) -> dict:
        """Return training metrics."""
        return {
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'avg_loss': np.mean(self.loss_history) if self.loss_history else 0.0,
            'buffer_size': len(self.replay_buffer),
        }

"""
Deep Q-Network (DQN) agent for Spot Instance optimization.

Implements DQN with:
- Experience replay
- Target network
- Epsilon-greedy exploration
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple
from collections import deque

from agents.networks import QNetwork
from agents.replay_buffer import ReplayBuffer


class DQNAgent:
    """
    DQN agent implementation.

    TODO: Implement DQN training loop:
    - select_action: epsilon-greedy action selection
    - store_transition: add experience to replay buffer
    - train_step: sample batch and update Q-network
    - update_target_network: sync target network
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 100000,
        batch_size: int = 64,
        buffer_size: int = 100000,
        target_update_freq: int = 1000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize DQN agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Steps to decay epsilon
            batch_size: Batch size for training
            buffer_size: Replay buffer capacity
            target_update_freq: Steps between target network updates
            device: Device for PyTorch (cuda/cpu)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device

        # Epsilon-greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        # Networks
        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Metrics
        self.loss_history = deque(maxlen=100)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            training: If False, always exploit (no exploration)

        Returns:
            Selected action (int)
        """
        # TODO: Implement epsilon-greedy action selection
        # 1. With probability epsilon, select random action
        # 2. Otherwise, select action with max Q-value

        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.action_dim)
        else:
            # Exploit: best action according to Q-network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax(dim=1).item()

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
        Perform one training step (sample batch and update Q-network).

        Returns:
            Loss value, or None if buffer not ready
        """
        # TODO: Implement DQN training step
        # 1. Check if buffer has enough samples
        # 2. Sample random batch from replay buffer
        # 3. Compute target Q-values using target network
        # 4. Compute loss (MSE between Q-values and targets)
        # 5. Backprop and update Q-network
        # 6. Update epsilon
        # 7. Periodically update target network

        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = batch

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping (optional)
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

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
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
        }, path)

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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

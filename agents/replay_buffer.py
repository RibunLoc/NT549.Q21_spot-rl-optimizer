"""
Experience replay buffer for DQN.
"""

import numpy as np
from collections import deque


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.

    Stores (state, action, reward, next_state, done) transitions.
    """

    def __init__(self, capacity: int):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.rng = np.random.default_rng(np.random.randint(0, 2**31))

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Add experience to buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        """
        Sample random batch of experiences.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
            Each element is a numpy array or list
        """
        indices = self.rng.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Return current size of buffer."""
        return len(self.buffer)

    def clear(self):
        """Clear buffer."""
        self.buffer.clear()


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.

    TODO (optional extension): Implement prioritized replay
    - Sample experiences based on TD error
    - Higher priority for experiences with large error
    - Importance sampling weights to correct bias
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
    ):
        """
        Initialize prioritized replay buffer.

        Args:
            capacity: Maximum buffer size
            alpha: Prioritization exponent (0 = uniform, 1 = full priority)
            beta_start: Initial importance sampling weight
            beta_frames: Frames to anneal beta to 1.0
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.rng = np.random.default_rng(np.random.randint(0, 2**31))

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add experience with maximum priority."""
        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> tuple:
        """Sample batch proportional to priority, with IS weights."""
        n = len(self.buffer)
        priorities = self.priorities[:n]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = self.rng.choice(n, size=batch_size, replace=False, p=probs)

        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1

        weights = (n * probs[indices]) ** (-beta)
        weights /= weights.max()

        batch = [self.buffer[i] for i in indices]
        states      = np.array([e[0] for e in batch], dtype=np.float32)
        actions     = np.array([e[1] for e in batch], dtype=np.int64)
        rewards     = np.array([e[2] for e in batch], dtype=np.float32)
        next_states = np.array([e[3] for e in batch], dtype=np.float32)
        dones       = np.array([e[4] for e in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones, indices, weights.astype(np.float32)

    def update_priorities(self, indices, td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        priorities = np.abs(td_errors) + 1e-6   # epsilon để tránh priority = 0
        for idx, p in zip(indices, priorities):
            self.priorities[idx] = p ** self.alpha

    def __len__(self) -> int:
        return len(self.buffer)

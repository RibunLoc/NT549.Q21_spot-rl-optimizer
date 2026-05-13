"""
Experience replay buffer for DQN.
"""

import numpy as np
from collections import deque


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.

    Stores (state, action, reward, next_state, done, next_action_mask) transitions.
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
        next_action_mask: np.ndarray = None,
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
        mask = None if next_action_mask is None else np.asarray(next_action_mask, dtype=bool)
        self.buffer.append((state, action, reward, next_state, done, mask))

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
        masks = [exp[5] if len(exp) > 5 else None for exp in batch]

        if all(m is not None for m in masks):
            next_action_masks = np.array(masks, dtype=bool)
            return states, actions, rewards, next_states, dones, next_action_masks

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
        next_action_mask: np.ndarray = None,
    ):
        """Add experience with a moderately high priority.

        Dùng p95 thay vì max() vì max() có thể là outlier cực lớn, làm entry mới
        dominate sampling vô thời hạn tới khi được sample lại. p95 vẫn đủ cao để
        new transition được ưu tiên khám phá nhưng không vượt trội outliers.
        """
        if self.buffer:
            filled = self.priorities[:len(self.buffer)]
            init_priority = float(np.quantile(filled, 0.95))
            if init_priority <= 0:
                init_priority = 1.0
        else:
            init_priority = 1.0

        mask = None if next_action_mask is None else np.asarray(next_action_mask, dtype=bool)
        transition = (state, action, reward, next_state, done, mask)

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = init_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> tuple:
        """Sample batch proportional to priority, with IS weights."""
        n = len(self.buffer)
        priorities = self.priorities[:n].copy()
        # Guard: replace NaN/inf/zero với min positive để tránh probs.sum()==0
        priorities = np.where(np.isfinite(priorities) & (priorities > 0), priorities, 1e-6)
        probs = priorities ** self.alpha
        total = probs.sum()
        if total <= 0 or not np.isfinite(total):
            probs = np.ones(n, dtype=np.float64) / n
        else:
            probs /= total

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
        masks       = [e[5] if len(e) > 5 else None for e in batch]

        if all(m is not None for m in masks):
            next_action_masks = np.array(masks, dtype=bool)
            return (
                states, actions, rewards, next_states, dones,
                next_action_masks, indices, weights.astype(np.float32),
            )

        return states, actions, rewards, next_states, dones, indices, weights.astype(np.float32)

    def update_priorities(self, indices, td_errors: np.ndarray):
        """Update priorities based on TD errors.

        Store RAW |td_error| (pre-alpha). sample() applies ** alpha uniformly
        to all entries, nên cả add() và update_priorities() đều phải lưu raw
        priority để sampling consistent.
        """
        priorities = np.abs(td_errors) + 1e-6   # epsilon để tránh priority = 0
        for idx, p in zip(indices, priorities):
            self.priorities[idx] = p

    def __len__(self) -> int:
        return len(self.buffer)

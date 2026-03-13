"""
Neural network architectures for Q-function approximation.
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Multi-layer perceptron for Q-function approximation.

    Architecture:
    - Input: state vector
    - Hidden layers: 2-3 FC layers with ReLU
    - Output: Q-values for each action
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 128],
    ):
        """
        Initialize Q-network.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            hidden_dims: List of hidden layer dimensions
        """
        super(QNetwork, self).__init__()

        # Build layers
        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: State tensor (batch_size, state_dim)

        Returns:
            Q-values (batch_size, action_dim)
        """
        return self.network(state)


class LSTMQNetwork(nn.Module):
    """
    LSTM-based Q-network for capturing temporal dependencies.

    TODO (optional extension): Implement LSTM architecture
    - Use LSTM to process sequential states
    - Capture price trends and patterns over time
    - Better for time-series state features
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 2,
        fc_hidden_dim: int = 128,
    ):
        """
        Initialize LSTM Q-network.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            lstm_hidden_dim: LSTM hidden dimension
            lstm_num_layers: Number of LSTM layers
            fc_hidden_dim: Fully connected layer dimension
        """
        super(LSTMQNetwork, self).__init__()

        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, action_dim),
        )

    def forward(
        self,
        state: torch.Tensor,
        hidden: tuple = None,
    ) -> tuple:
        """
        Forward pass with LSTM.

        Args:
            state: State tensor (batch_size, seq_len, state_dim)
            hidden: LSTM hidden state (h, c)

        Returns:
            Q-values (batch_size, action_dim)
            Updated hidden state
        """
        # LSTM forward
        lstm_out, hidden = self.lstm(state, hidden)

        # Take last timestep output
        lstm_out = lstm_out[:, -1, :]

        # FC layers
        q_values = self.fc(lstm_out)

        return q_values, hidden

    def init_hidden(self, batch_size: int, device: str) -> tuple:
        """Initialize LSTM hidden state."""
        h0 = torch.zeros(
            self.lstm_num_layers, batch_size, self.lstm_hidden_dim
        ).to(device)
        c0 = torch.zeros(
            self.lstm_num_layers, batch_size, self.lstm_hidden_dim
        ).to(device)
        return (h0, c0)


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-network architecture.

    TODO (optional extension): Implement dueling architecture
    - Separate value and advantage streams
    - V(s): value of being in state s
    - A(s, a): advantage of action a in state s
    - Q(s, a) = V(s) + (A(s, a) - mean(A(s, :)))
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ):
        super(DuelingQNetwork, self).__init__()

        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: State tensor (batch_size, state_dim)

        Returns:
            Q-values (batch_size, action_dim)
        """
        features = self.feature(state)

        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Combine value and advantages
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values

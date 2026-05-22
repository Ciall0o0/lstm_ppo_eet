"""LSTM Actor-Critic network for elevator scheduling PPO."""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class LSTMActorCritic(nn.Module):
    """Shared LSTM encoder with independent Actor and Critic heads."""

    def __init__(
        self,
        state_dim: int = 73,
        action_dim: int = 3,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.0,
        actor_hidden: int = 64,
        critic_hidden: int = 64,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers

        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
        )

        self.actor = nn.Sequential(
            nn.Linear(lstm_hidden, actor_hidden),
            nn.ReLU(),
            nn.Linear(actor_hidden, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(lstm_hidden, critic_hidden),
            nn.ReLU(),
            nn.Linear(critic_hidden, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)
        for module in [self.actor, self.critic]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=0.01)
                    nn.init.constant_(layer.bias, 0.0)

    def forward(self, obs_seq: torch.Tensor, hidden: tuple | None = None):
        """Forward pass through LSTM and both heads.

        Args:
            obs_seq: (batch, seq_len, state_dim) observation sequence
            mask:   (batch, seq_len) bool mask for valid timesteps
            hidden: optional initial LSTM hidden state

        Returns:
            action_logits: (batch, seq_len, action_dim)
            values:        (batch, seq_len, 1)
            hidden:        final LSTM hidden state
        """
        batch_size = obs_seq.size(0)
        if hidden is None:
            h0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden,
                             device=obs_seq.device)
            c0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden,
                             device=obs_seq.device)
            hidden = (h0, c0)

        lstm_out, hidden = self.lstm(obs_seq, hidden)      # (B, T, H)
        action_logits = self.actor(lstm_out)                # (B, T, A)
        values = self.critic(lstm_out)                      # (B, T, 1)
        return action_logits, values, hidden

    def get_action(self, obs_seq: torch.Tensor, hidden: tuple | None = None,
                   deterministic: bool = False):
        """Sample an action from the policy.

        Returns:
            action:    (batch, seq_len) sampled actions
            log_prob:  (batch, seq_len) log probabilities
            value:     (batch, seq_len, 1) state values
            hidden:    final LSTM hidden state
        """
        action_logits, values, hidden = self.forward(obs_seq, hidden)
        dist = Categorical(logits=action_logits)

        if deterministic:
            action = action_logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)  # (B, T)
        return action, log_prob, values.squeeze(-1), hidden

    def evaluate_actions(self, obs_seq: torch.Tensor, actions: torch.Tensor,
                         hidden: tuple | None = None):
        """Evaluate given actions for PPO update.

        Returns:
            action_logits: (B, T, A)
            log_probs:     (B, T)
            values:        (B, T)
            entropy:       (B, T)
            hidden:        final LSTM state
        """
        action_logits, values, hidden = self.forward(obs_seq, hidden)
        dist = Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)    # (B, T)
        entropy = dist.entropy()              # (B, T)
        return action_logits, log_probs, values.squeeze(-1), entropy, hidden

    def get_initial_hidden(self, batch_size: int, device: torch.device):
        return (
            torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=device),
            torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=device),
        )

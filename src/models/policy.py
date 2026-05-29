"""LSTM Actor-Critic network for elevator scheduling PPO."""

import torch
import torch.nn as nn
from torch.distributions import Categorical


def _get_activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    elif name == "silu":
        return nn.SiLU()
    return nn.ReLU()


class LSTMActorCritic(nn.Module):
    """Shared LSTM encoder with independent Actor and Critic heads.

    Standard LSTM+PPO architecture: LSTM encoder produces a hidden state,
    which feeds into separate actor (action logits) and critic (value) MLPs.
    """

    def __init__(
        self,
        state_dim: int = 73,
        action_dim: int = 3,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.0,
        actor_hidden: int = 64,
        critic_hidden: int = 64,
        activation: str = "relu",
        actor_dropout: float = 0.0,
        critic_dropout: float = 0.0,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers

        dropout = lstm_dropout if lstm_layers > 1 else 0.0

        self.encoder = nn.LSTM(
            input_size=state_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(lstm_hidden)

        act_fn = _get_activation(activation)

        actor_layers = [nn.Linear(lstm_hidden, actor_hidden), act_fn]
        if actor_dropout > 0:
            actor_layers.append(nn.Dropout(p=actor_dropout))
        actor_layers.append(nn.Linear(actor_hidden, action_dim))
        self.actor = nn.Sequential(*actor_layers)

        critic_layers = [nn.Linear(lstm_hidden, critic_hidden), act_fn]
        if critic_dropout > 0:
            critic_layers.append(nn.Dropout(p=critic_dropout))
        critic_layers.append(nn.Linear(critic_hidden, 1))
        self.critic = nn.Sequential(*critic_layers)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.encoder.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)
        for module in [self.actor, self.critic]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0.0)

    def _init_hidden(self, batch_size: int, device: torch.device):
        h0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=device)
        c0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=device)
        return (h0, c0)

    def forward(self, obs_seq: torch.Tensor, hidden=None):
        batch_size = obs_seq.size(0)
        if hidden is None:
            hidden = self._init_hidden(batch_size, obs_seq.device)

        enc_out, hidden = self.encoder(obs_seq, hidden)
        if self.use_layer_norm:
            enc_out = self.layer_norm(enc_out)
        action_logits = self.actor(enc_out)
        values = self.critic(enc_out)
        return action_logits, values, hidden

    def get_action(self, obs_seq: torch.Tensor, hidden=None, deterministic: bool = False):
        action_logits, values, hidden = self.forward(obs_seq, hidden)
        dist = Categorical(logits=action_logits)

        if deterministic:
            action = action_logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action, log_prob, values.squeeze(-1), hidden

    def evaluate_actions(self, obs_seq: torch.Tensor, actions: torch.Tensor, hidden=None):
        action_logits, values, hidden = self.forward(obs_seq, hidden)
        dist = Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return action_logits, log_probs, values.squeeze(-1), entropy, hidden

    def get_initial_hidden(self, batch_size: int, device: torch.device):
        return self._init_hidden(batch_size, device)

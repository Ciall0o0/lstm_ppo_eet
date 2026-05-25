"""LSTM/GRU Actor-Critic network for elevator scheduling PPO."""

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
    """Shared recurrent encoder with independent Actor and Critic heads.

    Supports LSTM (default) and GRU encoders, configurable activations,
    residual projection, and optional LayerNorm in heads.
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
        encoder_type: str = "lstm",
        activation: str = "relu",
        use_head_layernorm: bool = False,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.encoder_type = encoder_type

        dropout = lstm_dropout if lstm_layers > 1 else 0.0

        if encoder_type == "gru":
            self.encoder = nn.GRU(
                input_size=state_dim,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                dropout=dropout,
                batch_first=True,
            )
        else:
            self.encoder = nn.LSTM(
                input_size=state_dim,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                dropout=dropout,
                batch_first=True,
            )

        self.ln = nn.LayerNorm(lstm_hidden)

        # Residual projection: state_dim → lstm_hidden
        self.res_proj = nn.Linear(state_dim, lstm_hidden) if state_dim != lstm_hidden else nn.Identity()

        act_fn = _get_activation(activation)

        actor_layers = [
            nn.Linear(lstm_hidden, actor_hidden),
            act_fn,
        ]
        if use_head_layernorm:
            actor_layers.insert(1, nn.LayerNorm(actor_hidden))
        actor_layers.append(nn.Linear(actor_hidden, action_dim))
        self.actor = nn.Sequential(*actor_layers)

        critic_layers = [
            nn.Linear(lstm_hidden, critic_hidden),
            act_fn,
        ]
        if use_head_layernorm:
            critic_layers.insert(1, nn.LayerNorm(critic_hidden))
        critic_layers.append(nn.Linear(critic_hidden, 1))
        self.critic = nn.Sequential(*critic_layers)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.encoder.named_parameters():
            if "weight" in name and "ih_" not in name and "hh_" not in name:
                # Non-recurrent weights (if any)
                nn.init.orthogonal_(param, gain=1.0)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)
        for module in [self.actor, self.critic]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
                    nn.init.constant_(layer.bias, 0.0)
        if isinstance(self.res_proj, nn.Linear):
            nn.init.kaiming_normal_(self.res_proj.weight, mode="fan_in", nonlinearity="relu")
            nn.init.constant_(self.res_proj.bias, 0.0)

    def _init_hidden(self, batch_size: int, device: torch.device):
        if self.encoder_type == "gru":
            h = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=device)
            return h
        else:
            h0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=device)
            c0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=device)
            return (h0, c0)

    def _residual(self, obs_seq: torch.Tensor, enc_out: torch.Tensor) -> torch.Tensor:
        proj = self.res_proj(obs_seq)
        return enc_out + proj

    def forward(self, obs_seq: torch.Tensor, hidden=None):
        batch_size = obs_seq.size(0)
        if hidden is None:
            hidden = self._init_hidden(batch_size, obs_seq.device)

        enc_out, hidden = self.encoder(obs_seq, hidden)
        enc_out = self._residual(obs_seq, enc_out)
        enc_out = self.ln(enc_out)
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

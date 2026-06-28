"""
ITASORL - Experiment B-v2 agent: a recurrent actor-critic with survival pressure.

Where Experiment B's agent (agent.py:RecurrentWorldModel) only PREDICTS its
sensory stream, this agent ACTS to stay alive. The trunk is deliberately the
same shape - encoder -> GRUCell(embed+act, hidden) -> h_t - so the recurrent
state h_t is directly comparable to Experiment B's, and the SAME probe harness
reads it out. The only thing that changed between B and B-v2 is the objective:
prediction (+ optional decoder auxiliary) PLUS a survival actor-critic.

Readout, not reward: world identity is never in the observation and never in the
reward. The probe (experiment_b2) is the only thing that ever sees the label.

Action handling (matches the env contract in world.py / patch_of_earth.py):
  - thrust in [0,1], turn in [-1,1]   -> a diagonal Gaussian over a raw latent,
    squashed by sigmoid/tanh into the env action. The policy-gradient log-prob is
    taken on the RAW latent (the squash is treated as part of the environment),
    which avoids tanh-Jacobian bookkeeping while staying a valid PG estimator.
  - eat / drink / emit (thresholded at >0.5 by the env) -> independent Bernoullis.

Two action representations are tracked on purpose:
  - raw_act   = [raw_continuous(2), bernoulli(3)]  -> used ONLY for log-prob/entropy
  - env_act   = [sigmoid, tanh, bernoulli]         -> what actually drives dynamics,
                                                      fed back as the GRU's prev-action
                                                      and as the decoder's conditioning.
"""

from __future__ import annotations

import torch
import torch.nn as nn

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0
N_CONT = 2  # thrust, turn


class RecurrentActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, embed: int = 64, hidden: int = 128,
                 world_model: bool = True) -> None:
        super().__init__()
        self.obs_dim, self.act_dim, self.hidden = obs_dim, act_dim, hidden
        self.n_cont = N_CONT
        self.n_bin = act_dim - N_CONT
        assert self.n_bin >= 0, "act_dim must be >= the 2 continuous dims"
        # Shared trunk - same shape as agent.py:RecurrentWorldModel so h_t is comparable.
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, embed), nn.ReLU(), nn.Linear(embed, embed), nn.ReLU()
        )
        self.cell = nn.GRUCell(embed + act_dim, hidden)
        # Actor head: continuous means (n_cont) ++ Bernoulli logits (n_bin).
        self.actor = nn.Linear(hidden, self.n_cont + self.n_bin)
        self.log_std = nn.Parameter(torch.zeros(self.n_cont))
        self.critic = nn.Linear(hidden, 1)
        self.world_model = bool(world_model)
        if self.world_model:
            self.decoder = nn.Sequential(
                nn.Linear(hidden + act_dim, embed), nn.ReLU(), nn.Linear(embed, obs_dim)
            )

    # --- recurrent core -----------------------------------------------------
    def initial_state(self, batch: int, device) -> torch.Tensor:
        return torch.zeros(batch, self.hidden, device=device)

    def step_state(self, obs: torch.Tensor, prev_act: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """obs (B,O), prev_act (B,A), h (B,H) -> h' (B,H). prev_act is the env action."""
        e = self.encoder(obs)
        return self.cell(torch.cat([e, prev_act], dim=-1), h)

    def _dist(self, h: torch.Tensor):
        out = self.actor(h)
        mu = out[..., :self.n_cont]
        logits = out[..., self.n_cont:]
        std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp().expand_as(mu)
        cont = torch.distributions.Normal(mu, std)
        bino = torch.distributions.Bernoulli(logits=logits)
        return cont, bino

    @staticmethod
    def to_env_action(raw_c: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Squash the raw continuous latent into the env's action ranges and append the binaries."""
        thrust = torch.sigmoid(raw_c[..., :1])       # -> [0,1]
        turn = torch.tanh(raw_c[..., 1:2])           # -> [-1,1]
        return torch.cat([thrust, turn, b], dim=-1)

    # --- online step (rollout collection) -----------------------------------
    @torch.no_grad()
    def act(self, obs: torch.Tensor, prev_act: torch.Tensor, h: torch.Tensor,
            deterministic: bool = False):
        """One control step. Returns (raw_act, env_act, logp, value, h')."""
        h = self.step_state(obs, prev_act, h)
        cont, bino = self._dist(h)
        if deterministic:
            raw_c = cont.mean
            b = (bino.probs > 0.5).float()
        else:
            raw_c = cont.sample()
            b = bino.sample()
        logp = cont.log_prob(raw_c).sum(-1) + bino.log_prob(b).sum(-1)
        value = self.critic(h).squeeze(-1)
        raw_act = torch.cat([raw_c, b], dim=-1)
        env_act = self.to_env_action(raw_c, b)
        return raw_act, env_act, logp, value, h

    # --- training (recompute over a stored trajectory) ----------------------
    def score_actions(self, obs_seq: torch.Tensor, act_in_seq: torch.Tensor,
                      raw_seq: torch.Tensor, h0: torch.Tensor):
        """Recompute log-probs, values, entropies and the recurrent states for a
        stored trajectory, with gradients. Shapes (B,T,*).

        act_in_seq[:, t] is the env action fed as the GRU's PREVIOUS action at step
        t (i.e. env_act at t-1, zeros at t=0). raw_seq is the raw sampled action used
        for the log-prob. Returns (logp, value, entropy, states), each (B,T) / (B,T,H).
        """
        B, T, _ = obs_seq.shape
        e = self.encoder(obs_seq)
        h = h0
        logps, values, ents, states = [], [], [], []
        for t in range(T):
            h = self.cell(torch.cat([e[:, t], act_in_seq[:, t]], dim=-1), h)
            states.append(h)
            cont, bino = self._dist(h)
            raw_c = raw_seq[:, t, :self.n_cont]
            b = raw_seq[:, t, self.n_cont:]
            logps.append(cont.log_prob(raw_c).sum(-1) + bino.log_prob(b).sum(-1))
            ents.append(cont.entropy().sum(-1) + bino.entropy().sum(-1))
            values.append(self.critic(h).squeeze(-1))
        return (torch.stack(logps, 1), torch.stack(values, 1),
                torch.stack(ents, 1), torch.stack(states, 1))

    def predict_next(self, states: torch.Tensor, env_act_seq: torch.Tensor) -> torch.Tensor:
        """Decoder auxiliary: predict obs_{t+1} from (h_t, env_act_t). (B,T,O)."""
        return self.decoder(torch.cat([states, env_act_seq], dim=-1))

    def world_model_loss(self, obs_seq: torch.Tensor, act_in_seq: torch.Tensor,
                         env_act_seq: torch.Tensor, mask: torch.Tensor, h0: torch.Tensor):
        """Pure next-step prediction loss over a trajectory (Experiment B's objective on
        THIS trunk), masked. Returns (loss, states). Used to train the prediction-only
        control agent so it is probed by the identical matched-pair readout."""
        B, T, _ = obs_seq.shape
        e = self.encoder(obs_seq)
        h = h0
        states = []
        for t in range(T):
            h = self.cell(torch.cat([e[:, t], act_in_seq[:, t]], dim=-1), h)
            states.append(h)
        states = torch.stack(states, 1)
        pred = self.predict_next(states, env_act_seq)
        m = mask[:, 1:].unsqueeze(-1)
        loss = (((pred[:, :-1] - obs_seq[:, 1:]) ** 2) * m).sum() / (m.sum() * obs_seq.shape[-1] + 1e-6)
        return loss, states

    # --- probing (no grad; the object Experiment B-v2 reads out) ------------
    @torch.no_grad()
    def states_for_probe(self, obs_seq: torch.Tensor, act_in_seq: torch.Tensor,
                         h0: torch.Tensor | None = None) -> torch.Tensor:
        """Recurrent states h_t over a trajectory, for the probe. (B,T,H)."""
        B, T, _ = obs_seq.shape
        e = self.encoder(obs_seq)
        h = h0 if h0 is not None else self.initial_state(B, obs_seq.device)
        states = []
        for t in range(T):
            h = self.cell(torch.cat([e[:, t], act_in_seq[:, t]], dim=-1), h)
            states.append(h)
        return torch.stack(states, 1)

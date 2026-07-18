"""
ITASORL - Experiment B, the agent: a compact recurrent world model (RSSM-lite).

The agent is trained ONLY to predict its own sensory stream (next-step
prediction). It is never told which world it inhabits and never rewarded for
world identity. The recurrent state h_t is the object Experiment B probes: if
world identity is linearly decodable from h_t, and that decoding is selective
(beats a randomized-label control), the agent has encoded it incidentally - the
core H4 claim ("readout, not reward").

A full Dreamer-style agent adds an actor-critic trained on survival reward; this
self-supervised world model is the natural first rung and isolates the probe
pipeline. The actor-critic swaps in later WITHOUT changing the probe harness,
because the harness only consumes recurrent states.

Two backends are provided so the pipeline always runs:
  - RecurrentWorldModel (PyTorch): the real, trainable agent.
  - Reservoir (numpy): an untrained random recurrent embedding. Probing it tests
    whether identity is *present* in a generic recurrent code (an Experiment-A
    adjacent check), and serves as a fallback when torch is unavailable. It does
    NOT support the "the agent learned to encode it" claim - only training does.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH = True
except Exception:  # torch optional
    TORCH = False


if TORCH:

    class RecurrentWorldModel(nn.Module):
        def __init__(self, obs_dim: int, act_dim: int, embed: int = 64, hidden: int = 128) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(obs_dim, embed), nn.ReLU(), nn.Linear(embed, embed), nn.ReLU()
            )
            self.cell = nn.GRUCell(embed + act_dim, hidden)
            self.decoder = nn.Sequential(
                nn.Linear(hidden, embed), nn.ReLU(), nn.Linear(embed, obs_dim)
            )
            self.hidden = hidden
            # Output convention of the decoder: False -> predict absolute next obs;
            # True -> predict the observation *change* (obs[t+1]-obs[t]). The flag
            # lives on the model because forward_rollout must integrate predictions
            # back into inputs without being told the mode at call time.
            self.delta = False

        def forward(self, obs, act):
            """obs (B,T,O), act (B,T,A) -> next-obs prediction (B,T,O), states (B,T,H)."""
            B, T, _ = obs.shape
            h = torch.zeros(B, self.hidden, device=obs.device)
            e = self.encoder(obs)
            states, preds = [], []
            for t in range(T):
                h = self.cell(torch.cat([e[:, t], act[:, t]], dim=-1), h)
                states.append(h)
                preds.append(self.decoder(h))
            return torch.stack(preds, 1), torch.stack(states, 1)

        def prediction_loss(self, obs, act):
            pred, states = self(obs, act)
            # pred[:, t] is the model's guess for obs[:, t+1] - absolute or delta,
            # per self.delta (the decoder's output convention; forward_rollout
            # integrates deltas back, so training must target the same quantity).
            tgt = (obs[:, 1:] - obs[:, :-1]) if self.delta else obs[:, 1:]
            loss = ((pred[:, :-1] - tgt) ** 2).mean()
            return loss, states

        def forward_rollout(self, obs, act, context):
            """Teacher-force on real obs for the first `context` steps, then imagine
            open-loop: feed the model's own predictions back as input, driven by the
            given actions alone. Returns preds (B,T,O) with pred[:, t] the guess for
            step t+1 (absolute or delta, per self.delta), same convention as forward.
            """
            B, T, _ = obs.shape
            if context < 1:
                raise ValueError(f"context must be >= 1 (got {context}): the first open-loop "
                                 "step reconstructs its input from the previous prediction")
            h = torch.zeros(B, self.hidden, device=obs.device)
            x = obs[:, 0]
            preds = []
            for t in range(T):
                if t < context:
                    x = obs[:, t]
                else:  # open loop: reconstruct the predicted obs at time t
                    prev = preds[-1]
                    x = x + prev if self.delta else prev
                h = self.cell(torch.cat([self.encoder(x), act[:, t]], dim=-1), h)
                preds.append(self.decoder(h))
            return torch.stack(preds, 1)

        def rollout_loss(self, obs, act, context):
            """Open-loop prediction error over the imagined horizon (the trained
            objective for the k-step / delta-rollout experiments)."""
            T = obs.shape[1]
            if not 1 <= context <= T - 2:
                raise ValueError(f"context must be in [1, T-2] (got {context}, T={T}): "
                                 "otherwise the scored horizon is empty and the loss is NaN")
            pred = self.forward_rollout(obs, act, context)
            P = pred[:, context:-1]
            tgt = (obs[:, context + 1:] - obs[:, context:-1]) if self.delta else obs[:, context + 1:]
            return ((P - tgt) ** 2).mean()


class Reservoir:
    """Untrained random recurrent embedding (echo-state style). Fallback / baseline."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128, radius: float = 0.9, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.Win = rng.normal(0, 1.0 / np.sqrt(obs_dim), (hidden, obs_dim))
        self.Wa = rng.normal(0, 1.0 / np.sqrt(act_dim), (hidden, act_dim))
        W = rng.normal(0, 1.0, (hidden, hidden))
        eig = np.max(np.abs(np.linalg.eigvals(W)))
        self.Wh = W * (radius / eig)  # scale spectral radius below 1 (stable echo state)
        self.hidden = hidden

    def states(self, obs: np.ndarray, act: np.ndarray) -> np.ndarray:
        """obs (B,T,O), act (B,T,A) -> states (B,T,H)."""
        B, T, _ = obs.shape
        h = np.zeros((B, self.hidden))
        out = np.empty((B, T, self.hidden))
        for t in range(T):
            h = np.tanh(obs[:, t] @ self.Win.T + act[:, t] @ self.Wa.T + h @ self.Wh.T)
            out[:, t] = h
        return out

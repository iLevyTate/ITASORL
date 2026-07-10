"""L3 dynamics-level surrogate: a learned velocity law `G_motion`.

The L3 surrogate world runs the authentic physics and the REAL observation model, but
replaces the analytic velocity update `(1 - drag*dt)*vel + a*dt` with a small learned net
trained on AUTHENTIC motion transitions. Crucially `G_motion` is NOT given the true drag,
so it must approximate the dynamics from `(vel, a)` alone - its systematic error is the L3
"generative fingerprint", and the net's capacity controls that error (validated: capacity
is a clean monotone knob for the L2-style residual oracle; see `docs/PREREGISTRATION_L3.md`
section 4 Stage-2 and the section-12 deviation).

Because the observations still come from the real sensor model applied to `G`'s motion, the
surrogate observations stay on the authentic manifold - there is no "synthetic vs real"
giveaway, unlike the retired observation-channel construction.
"""

from __future__ import annotations

import numpy as np


def collect_authentic_transitions(*, n_eps: int = 250, steps: int = 40, params=None,
                                  ray_steps: int = 5, seed0: int = 0):
    """Run authentic (drift-free) rollouts under the scripted policy and return the motion
    transitions `(X, Y)` where `X = [vel_x, vel_y, a_x, a_y]` (N, 4) and `Y = vel_next`
    (N, 2). Drag is deliberately excluded from `X` (see module docstring)."""
    from .patch_of_earth import PatchOfEarthV0
    from .world import SeedBundle, WorldParams
    from .experiment_b import scripted_policy
    X, Y = [], []
    for i in range(n_eps):
        w = PatchOfEarthV0(params or WorldParams())
        w.ray_steps = ray_steps
        w._log_motion = []
        w.reset(SeedBundle(world=seed0 + i, weather=seed0 + 7000 + i, ecology=seed0 + 13000 + i))
        rng = np.random.default_rng(seed0 + i)
        for _ in range(steps):
            w.step(scripted_policy(rng))
        for vel, a, _drag, vnext in w._log_motion:
            X.append([vel[0], vel[1], a[0], a[1]])
            Y.append([vnext[0], vnext[1]])
    return np.asarray(X, np.float32), np.asarray(Y, np.float32)


class GMotion:
    """A frozen learned velocity law, callable as the world's `_g_motion` hook:
    `(vel, a, drag) -> vel_next`. Ignores `drag` by design. Runs on CPU (single-step world
    inference); `capacity` (the net's hidden width) is the calibration difficulty knob."""

    def __init__(self, net, norm):
        import torch
        self._torch = torch
        self._net = net.to("cpu")            # inference net (no dropout/BN -> no eval mode needed)
        self._xm, self._xs, self._ym, self._ys = norm  # numpy normalization stats

    def __call__(self, vel, a, drag=None) -> np.ndarray:
        x = np.array([vel[0], vel[1], a[0], a[1]], np.float32)
        xn = (x - self._xm) / self._xs
        with self._torch.no_grad():
            yn = self._net(self._torch.from_numpy(xn)).numpy()
        return (yn * self._ys + self._ym).astype(float)


def train_g_motion(*, hidden: int = 8, n_eps: int = 250, steps: int = 40, epochs: int = 300,
                   lr: float = 1e-3, seed: int = 0, params=None, ray_steps: int = 5,
                   device: str = "cpu") -> GMotion:
    """Train `G_motion` on authentic transitions. `hidden` is the single difficulty knob;
    `device` = "cpu" or "cuda" for training (the returned net is moved to CPU for the
    per-step world inference). Returns a frozen `GMotion` callable."""
    import torch
    import torch.nn as nn
    X, Y = collect_authentic_transitions(n_eps=n_eps, steps=steps, params=params,
                                         ray_steps=ray_steps, seed0=seed)
    xm, xs = X.mean(0), X.std(0) + 1e-6
    ym, ys = Y.mean(0), Y.std(0) + 1e-6
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(4, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                        nn.ReLU(), nn.Linear(hidden, 2)).to(device)
    xt = torch.tensor((X - xm) / xs, device=device)
    yt = torch.tensor((Y - ym) / ys, device=device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        ((net(xt) - yt) ** 2).mean().backward()
        opt.step()
    return GMotion(net, (xm, xs, ym, ys))

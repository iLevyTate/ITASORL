"""
ITASORL - Experiment B-v2: survival-coupled incidental detection.

Experiment B asked whether a PREDICTION-only agent incidentally encodes world
identity. It does not (target AUROC ~= 0.50). The conjecture (FINDINGS.md "next
steps") is that what an objective does not REQUIRE, the agent does not represent.
B-v2 changes exactly one thing: the agent now ACTS to stay alive, in a world
whose dynamics drift (L2). Surviving well depends on coping with the drifting
drag, so representing the dynamics becomes instrumentally useful - and the SAME
probe asks whether world identity is now decodable from the recurrent state.

Readout, not reward: world identity is never in the observation or the reward.

Pipeline (all knobs in run_expB2.py):
  1. train_actor_critic    - recurrent A2C + GAE (+ optional world-model decoder aux)
  2. engagement_metric     - GATE: trained return must beat random/scripted, else the
                             null is uninformative (cf. the open-loop MSE check in FINDINGS)
  3. matched_pair_recurrent_rollout - shared authentic prefix, then branch authentic vs
                             drift from an IDENTICAL world+agent state; record h_t per branch.
                             Kills the lifetime/return confound (both branches share history).
  4. probe + leakage_audit_b2 - target/shuffled/speed AUROC on h_t (reusing experiment_b's
                             harness) and a battery proving the probe reads the artifact,
                             not reward/length/lifetime.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from .agent_ac import RecurrentActorCritic
from .experiment_a import grouped_auroc
from .experiment_b import (episode_features, episode_features_full, episode_features_var,
                           probe_auroc, scripted_policy)
from .patch_of_earth import PatchOfEarthV0
from .stats import auroc_ci
from .world import SeedBundle, WorldParams


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# Harsh metabolism so SURVIVAL ACTUALLY BITES: with the stock constants the agent
# never dies inside a tractable horizon, so "survival pressure" is only a marginal
# foraging-return signal. Lowering starting energy and raising the basal burn makes a
# non-forager starve in ~50 steps, so foraging - and thus coping with the drifting
# drag that governs movement - becomes life-or-death. Applied to EVERY B-v2 world
# (train, readout, baselines) for consistency. Experiments A/B build their own worlds
# and are untouched.
SURVIVAL_METAB = {"E0": 1.0, "basal_E": 0.4, "Hyd0": 8.0, "basal_Hyd": 0.005}
# Denser, larger, easier-to-reach food so the eat-reward is not impossibly sparse for
# a from-scratch policy (the exploration bottleneck). Drift still governs how thrust
# translates into reaching that food, so the dynamics stay survival-relevant.
SURVIVAL_FOOD = {"n_pellets": 24, "reach": 0.08, "pellet_r": 0.03}
# Engagement gate: trained TRUE return must beat both baselines by ENGAGE_MARGIN, with
# lifetime no worse than random by LIFE_TOL steps. Return is the discriminating signal;
# lifetime saturates at this food density. Frozen from the de-risk (see engagement_metric).
ENGAGE_MARGIN = 0.15
LIFE_TOL = 2.0
# Surrogate coupling mode for every world the B-v2/B-v3 pipeline builds. "ar1" is the
# pre-registered B-v2 volatility surrogate; "regime" is the B-v3 per-episode constant
# drag offset (identifiable + policy-relevant); "l3" is the learned-dynamics surrogate
# (a trained G_motion replaces the velocity law). Patched in place by run_expB2.py
# --drift-mode (parent AND each spawned worker), mirroring the SURVIVAL_* scarcity overrides.
DRIFT_MODE = "ar1"
# The frozen L3 velocity net, trained ONCE via setup_l3_surrogate() and shared across every
# surrogate world (drift_sigma>0) when DRIFT_MODE=="l3". Authentic worlds (drift_sigma=0)
# never receive it, so they stay byte-identical to authentic. None until setup runs.
_L3_GMOTION = None
# Held-out L3 fingerprint (spec 2026-07-14): a SECOND trained G, never seen by any
# agent during training, used only by transfer_readout. None until setup runs.
_L3_GMOTION_HELDOUT = None


def setup_l3_surrogate(**train_kwargs) -> None:
    """Train the shared L3 velocity net (`itasorl.surrogate_l3.train_g_motion`) once and
    install it, so every subsequent `make_world(..., drift_sigma>0)` in `l3` mode gets it.
    `hidden` (capacity) is the calibration difficulty knob; pass `device="cuda"` to train
    on GPU."""
    global _L3_GMOTION
    from .surrogate_l3 import train_g_motion
    _L3_GMOTION = train_g_motion(**train_kwargs)


def setup_l3_heldout_surrogate(**train_kwargs) -> None:
    """Train the held-out G (different capacity, same frozen recipe) and store it for
    transfer_readout. Does NOT touch the training surrogate _L3_GMOTION."""
    global _L3_GMOTION_HELDOUT
    from .surrogate_l3 import train_g_motion
    _L3_GMOTION_HELDOUT = train_g_motion(**train_kwargs)


def make_world(params: WorldParams | None, drift_sigma: float, ray_steps: int,
               food_override: dict | None = None) -> PatchOfEarthV0:
    w = PatchOfEarthV0(params or WorldParams(), drift_sigma=drift_sigma, drift_mode=DRIFT_MODE)
    w.ray_steps = ray_steps
    # food_override is an ADDITIVE merge (control-arm world-invariant layout); None ->
    # byte-identical to the frozen SURVIVAL_FOOD layout every other experiment depends on.
    for k, v in {**SURVIVAL_METAB, **SURVIVAL_FOOD, **(food_override or {})}.items():
        setattr(w, k, v)
    if DRIFT_MODE == "l3" and drift_sigma > 0.0 and _L3_GMOTION is not None:
        w._g_motion = _L3_GMOTION  # surrogate (drift_sigma>0) uses learned dynamics; authentic does not
    return w


def _food_potential(world) -> float:
    """Potential Phi(s) = -distance to the nearest available pellet (0 if none/terminal).
    Used for POTENTIAL-BASED shaping, which provably preserves the optimal policy - a
    learning aid, not reward hacking, and identical in authentic vs surrogate worlds."""
    avail = world.pellet_amt > 1e-9
    if not avail.any():
        return 0.0
    d2 = np.sum((world.pellets[avail] - world.pos) ** 2, axis=1)
    return -float(np.sqrt(d2.min()))


def _seeds(base: int) -> SeedBundle:
    return SeedBundle(world=base, weather=base + 7000, ecology=base + 13000)


# ---------------------------------------------------------------------------
# Online observation normalizer (Welford). Frozen after training so the probe
# and the matched-pair readout see the SAME normalization the agent trained with.
# ---------------------------------------------------------------------------
class RunningNorm:
    def __init__(self, dim: int) -> None:
        self.mean = np.zeros(dim, np.float64)
        self.var = np.ones(dim, np.float64)
        self.count = 1e-4
        self.frozen = False

    def update(self, x: np.ndarray) -> None:
        if self.frozen:
            return
        x = np.atleast_2d(x)
        n = x.shape[0]
        bm = x.mean(0)
        bv = x.var(0)
        tot = self.count + n
        delta = bm - self.mean
        self.mean += delta * n / tot
        self.var = (self.var * self.count + bv * n + delta**2 * self.count * n / tot) / tot
        self.count = tot

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (np.sqrt(self.var) + 1e-6)

    def freeze(self) -> "RunningNorm":
        self.frozen = True
        return self


# ---------------------------------------------------------------------------
# Episode collection over parallel envs (one independent recurrent sequence each).
# ---------------------------------------------------------------------------
def collect_episodes_ac(agent: RecurrentActorCritic, norm: RunningNorm, params, drift_sigma,
                        n_eps: int, max_steps: int, device: str, seed_base: int,
                        ray_steps: int = 5, deterministic: bool = False, update_norm: bool = True,
                        shaping_coef: float = 0.0, gamma: float = 0.99,
                        food_override: dict | None = None):
    """Run n_eps parallel envs to death/max_steps. Returns padded torch tensors on
    `device` for the A2C update plus per-episode scalars. h0 is zero per episode.

    shaping_coef>0 adds potential-based food-approach shaping to the TRAINING reward
    (optimal-policy-preserving); `ret`/lengths still report the TRUE task outcome."""
    A = agent.act_dim
    envs = [make_world(params, drift_sigma, ray_steps, food_override) for _ in range(n_eps)]
    obs = np.stack([e.reset(_seeds(seed_base + i)).obs for i, e in enumerate(envs)]).astype(np.float64)
    active = np.ones(n_eps, bool)
    h = agent.initial_state(n_eps, device)
    prev_env_act = torch.zeros(n_eps, A, device=device)
    phi_prev = np.array([_food_potential(e) for e in envs])

    seq_obs = [[] for _ in range(n_eps)]      # normalized obs fed to the agent
    seq_actin = [[] for _ in range(n_eps)]    # prev env action fed to the GRU
    seq_raw = [[] for _ in range(n_eps)]      # raw sampled action (for log-prob)
    seq_env = [[] for _ in range(n_eps)]      # env action applied (decoder conditioning)
    seq_rew = [[] for _ in range(n_eps)]      # TRAINING reward (true + shaping)
    seq_true = [[] for _ in range(n_eps)]     # TRUE reward only (for engagement/return)
    seq_drift = [[] for _ in range(n_eps)]    # per-step drag-drift state (sysid-aux target)
    speeds = [[] for _ in range(n_eps)]
    terminated = np.zeros(n_eps, bool)

    for _ in range(max_steps):
        if not active.any():
            break
        if update_norm:
            norm.update(obs[active])
        obs_n = norm(obs)
        obs_t = torch.as_tensor(obs_n, dtype=torch.float32, device=device)
        raw_act, env_act, _, _, h = agent.act(obs_t, prev_env_act, h, deterministic=deterministic)
        env_np = env_act.detach().cpu().numpy()
        for i in range(n_eps):
            if not active[i]:
                continue
            seq_obs[i].append(obs_n[i].astype(np.float32))
            seq_actin[i].append(prev_env_act[i].detach().cpu().numpy())
            seq_raw[i].append(raw_act[i].detach().cpu().numpy())
            seq_env[i].append(env_np[i])
            r = envs[i].step(env_np[i].astype(np.float32))
            shaped = r.reward
            if shaping_coef:
                phi_new = 0.0 if r.terminated else _food_potential(envs[i])
                shaped = r.reward + shaping_coef * (gamma * phi_new - phi_prev[i])
                phi_prev[i] = phi_new
            seq_rew[i].append(shaped)
            seq_true[i].append(r.reward)
            seq_drift[i].append(float(envs[i]._drift_w))   # drag-drift used for this step
            speeds[i].append(float(np.linalg.norm(envs[i].vel)))
            obs[i] = r.obs
            if r.terminated:
                active[i] = False
                terminated[i] = True
        prev_env_act = env_act
        # zero hidden + prev action for finished envs (clean episode boundary if reused)
        amask = torch.as_tensor(active, dtype=torch.float32, device=device).unsqueeze(-1)
        h = h * amask
        prev_env_act = prev_env_act * amask

    lengths = np.array([len(s) for s in seq_obs])
    Tmax = int(lengths.max())
    O, Adim = agent.obs_dim, A

    def pad(seqs, width):
        out = np.zeros((n_eps, Tmax, width), np.float32)
        for i, s in enumerate(seqs):
            if s:
                out[i, : len(s)] = np.asarray(s, np.float32)
        return out

    mask = np.zeros((n_eps, Tmax), np.float32)
    for i, n in enumerate(lengths):
        mask[i, :n] = 1.0
    rew = np.zeros((n_eps, Tmax), np.float32)        # shaped reward -> GAE target
    for i, s in enumerate(seq_rew):
        rew[i, : len(s)] = np.asarray(s, np.float32)
    drift = np.zeros((n_eps, Tmax), np.float32)      # per-step drag-drift -> sysid-aux target
    for i, s in enumerate(seq_drift):
        drift[i, : len(s)] = np.asarray(s, np.float32)
    true_ret = np.array([float(np.sum(s)) for s in seq_true])  # TRUE return -> engagement

    batch = {
        "obs": torch.as_tensor(pad(seq_obs, O), device=device),
        "act_in": torch.as_tensor(pad(seq_actin, Adim), device=device),
        "raw": torch.as_tensor(pad(seq_raw, Adim), device=device),
        "env_act": torch.as_tensor(pad(seq_env, Adim), device=device),
        "reward": torch.as_tensor(rew, device=device),
        "drift_w": torch.as_tensor(drift, device=device),
        "mask": torch.as_tensor(mask, device=device),
        "terminated": torch.as_tensor(terminated.astype(np.float32), device=device),
        "lengths": lengths,
        "ret": true_ret,
        "speed": np.array([np.mean(s) if s else 0.0 for s in speeds]),
    }
    return batch


def compute_gae(reward, value, mask, terminated, gamma, lam):
    """Per-episode GAE. reward/value/mask (B,T); terminated (B,). Bootstrap is 0 for
    episodes that ended in death, else the last in-episode value (truncation)."""
    B, T = reward.shape
    adv = torch.zeros_like(reward)
    last_idx = mask.sum(1).long().clamp(min=1) - 1  # final valid step per episode
    boot = torch.where(terminated > 0.5, torch.zeros(B, device=reward.device),
                       value.gather(1, last_idx.unsqueeze(1)).squeeze(1))
    gae = torch.zeros(B, device=reward.device)
    next_v = boot
    # next_mask = mask of step t+1 (0 at the final valid step). It gates the GAE
    # accumulator so the carry resets at the episode boundary; using the CURRENT step's
    # mask instead would leak the padded-step delta into the last valid step's advantage.
    next_mask = torch.zeros(B, device=reward.device)
    for t in reversed(range(T)):
        m = mask[:, t]
        delta = reward[:, t] + gamma * next_v - value[:, t]
        gae = delta + gamma * lam * next_mask * gae
        adv[:, t] = gae * m
        next_v = torch.where(m > 0.5, value[:, t], next_v)
        next_mask = m
    ret = adv + value
    return adv, ret


def train_actor_critic(drift_sigma: float, params=None, *, n_eps: int = 16, updates: int = 200,
                       embed: int = 64, hidden: int = 96, world_model: bool = True,
                       lr: float = 3e-4, gamma: float = 0.99, lam: float = 0.95,
                       ent_coef: float = 0.01, vf_coef: float = 0.5, wm_coef: float = 1.0,
                       shaping_coef: float = 0.5, max_steps: int = 80, ray_steps: int = 5,
                       seed: int = 0, device: str | None = None, log_every: int = 0,
                       sysid_aux: bool = False, sysid_coef: float = 1.0):
    """Train the survival actor-critic. sysid_aux adds a CEILING-control auxiliary loss
    that regresses h_t onto the scalar drag-drift - a positive control that deliberately
    breaks readout-not-reward to measure whether the trunk CAN linearly encode world
    identity. Run it separately from the headline; never fold its target into the H_B2 verdict."""
    device = device or default_device()
    torch.manual_seed(seed)
    np.random.seed(seed)
    probe = make_world(params, drift_sigma, ray_steps)
    obs_dim, act_dim = probe.obs_spec.size, probe.action_spec.size
    agent = RecurrentActorCritic(obs_dim, act_dim, embed, hidden, world_model,
                                 sysid_aux=sysid_aux).to(device)
    opt = torch.optim.Adam(agent.parameters(), lr=lr)
    norm = RunningNorm(obs_dim)
    history = []
    for u in range(updates):
        seed_base = 100_000 + seed * 10_000 + u * n_eps
        batch = collect_episodes_ac(agent, norm, params, drift_sigma, n_eps, max_steps,
                                    device, seed_base, ray_steps, deterministic=False,
                                    shaping_coef=shaping_coef, gamma=gamma)
        logp, value, ent, states = agent.score_actions(batch["obs"], batch["act_in"], batch["raw"],
                                                        agent.initial_state(n_eps, device))
        with torch.no_grad():
            adv, ret = compute_gae(batch["reward"], value, batch["mask"], batch["terminated"], gamma, lam)
            am = batch["mask"].sum()
            adv = (adv - (adv * batch["mask"]).sum() / am) / ((adv * batch["mask"]).std() + 1e-6)
        m = batch["mask"]
        pg = -(logp * adv * m).sum() / am
        vloss = (((value - ret) ** 2) * m).sum() / am
        entropy = (ent * m).sum() / am
        loss = pg + vf_coef * vloss - ent_coef * entropy
        if world_model:
            pred = agent.predict_next(states, batch["env_act"])  # predict obs_{t+1} from (h_t, a_t)
            wm = (((pred[:, :-1] - batch["obs"][:, 1:]) ** 2) * m[:, 1:].unsqueeze(-1)).sum() / (m[:, 1:].sum() * obs_dim + 1e-6)
            loss = loss + wm_coef * wm
        if sysid_aux:  # CEILING control: supervise h_t -> drag-drift (breaks readout-not-reward)
            sid_pred = agent.predict_sysid(states)
            sid = (((sid_pred - batch["drift_w"]) ** 2) * m).sum() / (am + 1e-6)
            loss = loss + sysid_coef * sid
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        opt.step()
        history.append(float(batch["ret"].mean()))
        if log_every and (u % log_every == 0 or u == updates - 1):
            print(f"   update {u:4d}  mean_return={np.mean(history[-log_every:]):+.3f}  "
                  f"len={batch['lengths'].mean():.0f}  ent={float(entropy):.2f}")
    norm.freeze()
    return agent, norm, history


# ---------------------------------------------------------------------------
# Engagement gate - the trained policy must out-survive random / scripted.
# ---------------------------------------------------------------------------
def _baseline_return(kind: str, params, drift_sigma, n_eps, max_steps, ray_steps, seed_base):
    rng = np.random.default_rng(seed_base)
    rets, lens = [], []
    for i in range(n_eps):
        w = make_world(params, drift_sigma, ray_steps)
        w.reset(_seeds(seed_base + i))
        R, t = 0.0, 0
        for _ in range(max_steps):
            if kind == "random":
                a = np.array([rng.uniform(0, 1), rng.uniform(-1, 1), float(rng.random() < 0.5),
                              float(rng.random() < 0.5), 0.0], np.float32)
            else:
                a = scripted_policy(rng)
            r = w.step(a)
            R += r.reward
            t += 1
            if r.terminated:
                break
        rets.append(R)
        lens.append(t)
    return float(np.mean(rets)), float(np.mean(lens))


def engagement_metric(agent, norm, params, drift_sigma, *, n_eps: int = 64, max_steps: int = 80,
                      ray_steps: int = 5, device: str | None = None, seed_base: int = 900_000) -> dict:
    device = device or default_device()
    b = collect_episodes_ac(agent, norm, params, drift_sigma, n_eps, max_steps, device,
                            seed_base, ray_steps, deterministic=True, update_norm=False)
    trained_ret, trained_len = float(b["ret"].mean()), float(b["lengths"].mean())
    rnd_ret, rnd_len = _baseline_return("random", params, drift_sigma, n_eps, max_steps, ray_steps, seed_base + 1)
    scr_ret, scr_len = _baseline_return("scripted", params, drift_sigma, n_eps, max_steps, ray_steps, seed_base + 2)
    # Engagement = the trained policy forages MEANINGFULLY better than both baselines on
    # TRUE return, with lifetime not worse. Return (not lifetime) is the discriminating
    # signal: at the frozen food density lifetime saturates (even a random agent survives
    # ~68/80), so requiring strictly-longer lifetime flips on ~1 step of noise. The margin
    # cleanly separates a real forager (de-risk: +0.43 over random) from a non-learner
    # (the v2 pilot: +0.04). Calibrated on de-risk data, frozen for the confirmatory run.
    better_return = trained_ret >= max(rnd_ret, scr_ret) + ENGAGE_MARGIN
    not_worse_life = trained_len >= rnd_len - LIFE_TOL
    return {
        "trained_return": trained_ret, "random_return": rnd_ret, "scripted_return": scr_ret,
        "trained_len": trained_len, "random_len": rnd_len, "scripted_len": scr_len,
        "better_return": better_return, "not_worse_life": not_worse_life,
        "engaged": better_return and not_worse_life,
    }


def survival_return(agent, norm, params, drift_sigma, *, n_eps: int = 64, max_steps: int = 80,
                    ray_steps: int = 5, device: str | None = None, seed_base: int = 920_000) -> float:
    """Mean TRUE (deterministic) return of a frozen agent in a world at `drift_sigma`.
    Used for the manipulation check: cross-evaluating an agent at a drift it was NOT
    trained on. A fixed seed_base keeps the authentic world layout identical across eval
    drifts, so the only difference is the drag regime - if return drops, the artifact is
    survival-relevant (and a chance world-identity probe is then genuinely informative)."""
    device = device or default_device()
    b = collect_episodes_ac(agent, norm, params, drift_sigma, n_eps, max_steps, device,
                            seed_base, ray_steps, deterministic=True, update_norm=False)
    return float(b["ret"].mean())


# ---------------------------------------------------------------------------
# Matched-pair recurrent readout - the keystone confound control with a recurrent
# agent. Shared authentic prefix, snapshot world + agent state, then branch
# authentic vs drift from the IDENTICAL state. Records h_t per branch for the probe.
# ---------------------------------------------------------------------------
def _run_branch(agent, norm, world, h0, prev_act0, branch_steps, device):
    """Step `world` with the (frozen, deterministic) agent for branch_steps, carrying
    the recurrent state. Returns per-step h_t (T,H), speeds, rewards, and `alive`."""
    h = h0.clone()
    prev = prev_act0.clone()
    obs = world.observe().astype(np.float64)
    Hs, spd, rew = [], [], []
    alive = True
    for _ in range(branch_steps):
        obs_t = torch.as_tensor(norm(obs)[None], dtype=torch.float32, device=device)
        _, env_act, _, _, h = agent.act(obs_t, prev, h, deterministic=True)
        Hs.append(h[0].detach().cpu().numpy())
        env_np = env_act[0].detach().cpu().numpy().astype(np.float32)
        r = world.step(env_np)
        spd.append(float(np.linalg.norm(world.vel)))
        rew.append(float(r.reward))
        obs = r.obs.astype(np.float64)
        prev = env_act
        if r.terminated:
            alive = False
            break
    return np.asarray(Hs, np.float32), spd, rew, alive


def matched_pair_recurrent_rollout(agent, norm, params, drift_sigma, *, n_pairs: int = 60,
                                   prefix_steps: int = 20, branch_steps: int = 24, ray_steps: int = 5,
                                   device: str | None = None, seed_base: int = 700_000) -> tuple[list, list]:
    """Returns (auth_eps, surr_eps): per-branch dicts with H (T,Hdim), speed, reward_sum,
    length, lifetime. Authentic vs surrogate branch from a bit-identical shared state."""
    device = device or default_device()
    auth_eps, surr_eps = [], []
    for p in range(n_pairs):
        seeds = _seeds(seed_base + p)
        # shared authentic prefix (drift OFF) with the frozen agent, carrying h_t
        base = make_world(params, 0.0, ray_steps)
        base.reset(seeds)
        h = agent.initial_state(1, device)
        prev = torch.zeros(1, agent.act_dim, device=device)
        obs = base.observe().astype(np.float64)
        for _ in range(prefix_steps):
            obs_t = torch.as_tensor(norm(obs)[None], dtype=torch.float32, device=device)
            _, env_act, _, _, h = agent.act(obs_t, prev, h, deterministic=True)
            r = base.step(env_act[0].detach().cpu().numpy().astype(np.float32))
            obs = r.obs.astype(np.float64)
            prev = env_act
            if r.terminated:
                break
        snapshot = base.get_state()
        h_branch, prev_branch = h.clone(), prev.clone()

        # authentic branch: fresh authentic world restored to the snapshot
        aw = make_world(params, 0.0, ray_steps)
        aw.reset(seeds)
        aw.set_state(snapshot)
        Ha, spa, rea, alivea = _run_branch(agent, norm, aw, h_branch, prev_branch, branch_steps, device)

        # surrogate branch: drift world restored to the SAME snapshot (drift activates here).
        # Keep the surrogate's own reset-drawn drift_w: the snapshot comes from the
        # drift-free prefix world (drift_w=0.0), and in regime mode step() never redraws,
        # so a plain set_state would silently disable the drag offset for the whole branch
        # (mp_target collapsed to exactly 0.5). Under ar1 the reset value is 0.0, so this
        # merge is a bit-for-bit no-op there.
        sw = make_world(params, drift_sigma, ray_steps)
        sw.reset(seeds)
        regime_w = float(sw.get_state()["drift_w"])
        sw.set_state({**snapshot, "drift_w": regime_w})
        Hs, sps, res, alives = _run_branch(agent, norm, sw, h_branch, prev_branch, branch_steps, device)

        # Truncate BOTH branches of the pair to equal length so episode length /
        # lifetime cannot leak the label (the matched-pair confound guard, by construction).
        L = min(len(Ha), len(Hs))
        if L == 0:
            continue
        full = int(L == branch_steps)
        auth_eps.append({"H": Ha[:L], "label": 0, "speed": float(np.mean(spa[:L])),
                         "reward_sum": float(np.sum(rea[:L])), "length": L, "lifetime": full})
        surr_eps.append({"H": Hs[:L], "label": 1, "speed": float(np.mean(sps[:L])),
                         "reward_sum": float(np.sum(res[:L])), "length": L, "lifetime": full})
    return auth_eps, surr_eps


# ---------------------------------------------------------------------------
# Probe + leakage audit (reuse experiment_b.probe_auroc / grouped_auroc).
# ---------------------------------------------------------------------------
def _episode_feature(H: np.ndarray) -> np.ndarray:
    """Per-episode probe input = [mean recurrent state, final recurrent state].
    Matches experiment_b.episode_features semantics for a single variable-length episode."""
    return np.concatenate([H.mean(0), H[-1]])


def probe_world_identity(auth_eps: list, surr_eps: list, seed: int = 0) -> dict:
    eps = auth_eps + surr_eps
    X = np.stack([_episode_feature(e["H"]) for e in eps])
    y = np.array([e["label"] for e in eps])
    spd = np.array([e["speed"] for e in eps])
    # The two members of pair i share seed, prefix, and branch state, so they are NOT
    # independent episodes: give them one group id so GroupKFold never splits a pair
    # across folds. (Split twins let the probe read the train twin's label off the
    # near-identical test twin, biasing AUROC toward 0 whenever the pair count is not
    # a multiple of n_splits.)
    g = np.concatenate([np.arange(len(auth_eps)), np.arange(len(surr_eps))])
    rng = np.random.default_rng(seed)
    return {
        "target": grouped_auroc(X, y, g),                                   # H4: decode world identity
        "shuffled": grouped_auroc(X, rng.permutation(y), g),                # negative control
        "speed": grouped_auroc(X, (spd > np.median(spd)).astype(int), g),   # positive control
        "n": len(eps),
    }


def leakage_audit_b2(auth_eps: list, surr_eps: list, margin: float = 0.1,
                     groups: np.ndarray | None = None) -> dict:
    """Confound battery: world identity must NOT be decodable from reward/length/lifetime.
    A clean target probe with these near 0.5 proves it reads the artifact, not 'I lived
    longer'. `margin` is the tolerated |AUROC-0.5|; the run gate is this default 0.1
    (see readout's docstring for why tighter would false-alarm on the matched-pair path).
    `groups`: pass a shared pair id per pair member when the episodes are matched pairs
    (see probe_world_identity); default treats episodes as independent."""
    eps = auth_eps + surr_eps
    y = np.array([e["label"] for e in eps])
    g = groups if groups is not None else np.arange(len(eps))

    def channel(key):
        v = np.array([e[key] for e in eps], np.float64).reshape(-1, 1)
        return grouped_auroc(v, y, g)

    audit = {k: channel(k) for k in ("reward_sum", "length", "lifetime")}
    audit["max_abs_dev"] = max(abs(a - 0.5) for a in audit.values())
    audit["margin"] = margin
    audit["clean"] = audit["max_abs_dev"] < margin
    return audit


def transfer_probe(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray,
                   return_scores: bool = False):
    """Held-out fingerprint transfer: fit the STANDARD linear probe family (same
    scaler+logistic pipeline grouped_auroc uses) once on the training pools, then
    score a FROZEN AUROC on disjoint test pools. No CV: train and test worlds are
    disjoint by construction, and the frozen score is the estimand (does the
    direction learned on the trained fingerprint generalize to an unseen one).
    return_scores=True also returns (yte, p_te) so callers can bootstrap a CI
    with itasorl.stats.auroc_ci (no refit), mirroring _auroc_with_ci."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        nan = float("nan")
        return (nan, yte, np.array([])) if return_scores else nan
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    auc = float(roc_auc_score(yte, p))
    return (auc, yte, p) if return_scores else auc


# ---------------------------------------------------------------------------
# Control agents: same trunk, different objective, probed by the IDENTICAL readout.
#   untrained_agent     - random init -> the MECHANICAL floor (drift perturbs inputs,
#                         so any recurrent state separates the worlds somewhat).
#   train_predictor_only- Experiment B's objective (next-step prediction, scripted
#                         policy) on this trunk -> tests prediction WITHOUT survival.
# "Survival pressure induces encoding" requires survival > predictor-only > untrained.
# ---------------------------------------------------------------------------
def untrained_agent(params, drift_sigma, ray_steps, hidden, embed, world_model, device, seed=0):
    torch.manual_seed(seed)
    probe = make_world(params, drift_sigma, ray_steps)
    agent = RecurrentActorCritic(probe.obs_spec.size, probe.action_spec.size, embed, hidden, world_model).to(device)
    norm = RunningNorm(probe.obs_spec.size)
    # a short normalizer warmup so the probe sees sensibly-scaled inputs
    collect_episodes_ac(agent, norm, params, drift_sigma, 8, 40, device, 555_000, ray_steps,
                        deterministic=False, update_norm=True)
    return agent.train(False), norm.freeze()


def _collect_scripted(agent, norm, params, drift_sigma, n_eps, max_steps, device, seed_base, ray_steps):
    """Collect episodes driven by the fixed scripted policy (Experiment B's policy),
    for training the prediction-only control. Returns padded tensors + mask."""
    A = agent.act_dim
    envs = [make_world(params, drift_sigma, ray_steps) for _ in range(n_eps)]
    rngs = [np.random.default_rng(seed_base + i) for i in range(n_eps)]
    obs = np.stack([e.reset(_seeds(seed_base + i)).obs for i, e in enumerate(envs)]).astype(np.float64)
    active = np.ones(n_eps, bool)
    seq_obs = [[] for _ in range(n_eps)]
    seq_actin = [[] for _ in range(n_eps)]
    seq_env = [[] for _ in range(n_eps)]
    prev = [np.zeros(A, np.float32) for _ in range(n_eps)]
    for _ in range(max_steps):
        if not active.any():
            break
        norm.update(obs[active])
        obs_n = norm(obs)
        for i in range(n_eps):
            if not active[i]:
                continue
            a = scripted_policy(rngs[i])
            seq_obs[i].append(obs_n[i].astype(np.float32))
            seq_actin[i].append(prev[i])
            seq_env[i].append(a)
            r = envs[i].step(a)
            obs[i] = r.obs
            prev[i] = a
            if r.terminated:
                active[i] = False
    lengths = np.array([len(s) for s in seq_obs])
    Tmax = int(lengths.max())

    def pad(seqs, w):
        out = np.zeros((n_eps, Tmax, w), np.float32)
        for i, s in enumerate(seqs):
            if s:
                out[i, : len(s)] = np.asarray(s, np.float32)
        return out
    mask = np.zeros((n_eps, Tmax), np.float32)
    for i, n in enumerate(lengths):
        mask[i, :n] = 1.0
    return (torch.as_tensor(pad(seq_obs, agent.obs_dim), device=device),
            torch.as_tensor(pad(seq_actin, A), device=device),
            torch.as_tensor(pad(seq_env, A), device=device),
            torch.as_tensor(mask, device=device))


def train_predictor_only(drift_sigma, params=None, *, n_eps=16, updates=200, embed=64, hidden=96,
                         lr=1e-3, max_steps=80, ray_steps=5, seed=0, device=None, log_every=0):
    device = device or default_device()
    torch.manual_seed(seed)
    np.random.seed(seed)
    probe = make_world(params, drift_sigma, ray_steps)
    agent = RecurrentActorCritic(probe.obs_spec.size, probe.action_spec.size, embed, hidden, True).to(device)
    opt = torch.optim.Adam(agent.parameters(), lr=lr)
    norm = RunningNorm(probe.obs_spec.size)
    for u in range(updates):
        obs, act_in, env_act, mask = _collect_scripted(agent, norm, params, drift_sigma, n_eps,
                                                        max_steps, device, 200_000 + seed * 9000 + u * n_eps, ray_steps)
        loss, _ = agent.world_model_loss(obs, act_in, env_act, mask, agent.initial_state(n_eps, device))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        opt.step()
        if log_every and (u % log_every == 0 or u == updates - 1):
            print(f"   pred-only update {u:4d}  wm_loss={float(loss):.4f}")
    return agent.train(False), norm.freeze()


# ---------------------------------------------------------------------------
# PRIMARY readout - non-matched pooled episodes, fixed length, trained policy.
# This is Experiment B's exact frame (independent authentic vs surrogate pools,
# probe a PERSISTENT world-identity direction in h_t), so the result is directly
# comparable to Exp B's 0.50 null. Fixed length + drop-on-early-death keeps episode
# length constant across pools, so length/lifetime cannot leak the label.
# ---------------------------------------------------------------------------
def collect_pool(agent, norm, params, drift_sigma, n_eps, steps, device, seed_base, ray_steps,
                 return_anchors: bool = False):
    """Collect up to n_eps episodes of EXACTLY `steps` length (drop early deaths) with
    the frozen deterministic agent. Returns H (k,steps,Hdim), speeds (k,).

    With return_anchors=True also returns per-episode mean energy, mean nearest-pellet
    distance, mean drag, and the per-episode summed reward. energy/food are the readout
    CEILING: decoding them shows the recurrent state is linearly readable, so a chance
    world-identity AUROC is genuine absence of encoding, not a dead probe. (They are NOT
    the world label: energy/food vary inside authentic and surrogate alike.) reward_sum
    feeds the pooled leakage audit: L3 dynamics shift movement-cost -> reward, so the
    headline probe must be shown NOT to be reading 'how much it ate' instead of identity.
    Also returns the PER-TIMESTEP behavior traces (k, steps, 4) - the same
    speed/energy/food/drag accumulators the anchor means are taken over - so the
    behavior-mediation audit can run its per-timestep control offline."""
    Hs, spd, energy, food, drag, reward, traces = [], [], [], [], [], [], []
    for i in range(n_eps):
        w = make_world(params, drift_sigma, ray_steps)
        w.reset(_seeds(seed_base + i))
        h = agent.initial_state(1, device)
        prev = torch.zeros(1, agent.act_dim, device=device)
        obs = w.observe().astype(np.float64)
        Hrow, sp, en, fd, dg, px, py, hd, died = [], [], [], [], [], [], [], [], False
        rw = 0.0
        for _ in range(steps):
            obs_t = torch.as_tensor(norm(obs)[None], dtype=torch.float32, device=device)
            _, env_act, _, _, h = agent.act(obs_t, prev, h, deterministic=True)
            Hrow.append(h[0].detach().cpu().numpy())
            r = w.step(env_act[0].detach().cpu().numpy().astype(np.float32))
            sp.append(float(np.linalg.norm(w.vel)))
            en.append(float(w.E / w.Emax))
            fd.append(-_food_potential(w))           # >=0 distance to nearest pellet
            dg.append(float(w._drift_w))             # instantaneous drag-drift state (0 in authentic)
            px.append(float(w.pos[0]))               # absolute position: diverges across worlds
            py.append(float(w.pos[1]))               # under differing velocity laws, so it must be
            hd.append(float(w.heading))              # available to the mediation control
            rw += float(r.reward)                     # summed homeostatic reward (never detection)
            obs = r.obs.astype(np.float64)
            prev = env_act
            if r.terminated:
                died = True
                break
        if not died and len(Hrow) == steps:
            Hs.append(np.asarray(Hrow, np.float32))
            spd.append(float(np.mean(sp)))
            energy.append(float(np.mean(en)))
            food.append(float(np.mean(fd)))
            drag.append(float(np.mean(dg)))
            reward.append(rw)
            traces.append(np.stack([sp, en, fd, dg, px, py, hd], axis=1).astype(np.float32))
    H = np.stack(Hs) if Hs else np.zeros((0, steps, agent.hidden), np.float32)
    if return_anchors:
        Bt = np.stack(traces) if traces else np.zeros((0, steps, 7), np.float32)
        return (H, np.asarray(spd), np.asarray(energy), np.asarray(food),
                np.asarray(drag), np.asarray(reward), Bt)
    return H, np.asarray(spd)


def _auroc_with_ci(X, y, seed: int = 0, groups: np.ndarray | None = None) -> tuple[float, float, float]:
    """5-fold grouped CV AUROC plus a stratified-bootstrap 95% CI from its out-of-fold
    predictions (no model refit). `groups` defaults to one group per row (independent
    episodes); matched-pair callers pass a shared pair id for the two members so
    GroupKFold never splits a pair across folds."""
    if groups is None:
        groups = np.arange(len(y))
    auc, yv, pv = grouped_auroc(X, y, groups, return_oof=True)
    if yv.size == 0:
        return auc, float("nan"), float("nan")
    lo, hi = auroc_ci(yv, pv, seed=seed)
    return auc, lo, hi


def pooled_readout(agent, norm, params, drift_sigma, *, n_eps=110, steps=24, ray_steps=5,
                   device=None, seed=0, dump_path=None, leak_margin=0.1, return_pools=False) -> dict:
    """Experiment-B-style probe: decode world identity across independent episodes.
    Reports the headline `target` (LEVEL features) with a bootstrap CI, plus two
    additive readouts that probe a VOLATILITY signature - `target_var` (dispersion
    features) and `target_full` (level ++ dispersion) - and per-probe `selectivity`
    (target minus a shuffled-label baseline on the SAME feature set, which cancels the
    overfitting bias a wider feature set incurs at n~220). Also reports anchor CEILINGS
    (energy, food-distance) that show the recurrent state is readable, a POOLED leakage
    audit (`pool_reward_leak`/`pool_leak_clean`: world identity must NOT be decodable from
    summed reward, so the headline reads the artifact not 'how much it ate'), and per-world
    survivor/death counts (`deaths_auth`/`deaths_surr`) that bound the survivorship
    asymmetry from dropping early deaths. If `dump_path` is set, persists the raw recurrent
    states AND per-episode reward so both probes can be recomputed offline (no GPU)."""
    device = device or default_device()
    Ha, spa, ena, fda, dra, rwa, bta = collect_pool(agent, norm, params, 0.0, n_eps, steps,
                                                    device, 800_000, ray_steps,
                                                    return_anchors=True)
    Hs, sps, ens, fds, drs, rws, bts = collect_pool(agent, norm, params, drift_sigma, n_eps,
                                                    steps, device, 850_000, ray_steps,
                                                    return_anchors=True)
    if dump_path is not None:
        d = os.path.dirname(dump_path)
        if d:
            os.makedirs(d, exist_ok=True)
        np.savez_compressed(dump_path, Ha=Ha, Hs=Hs, spa=spa, sps=sps, ena=ena, ens=ens,
                            fda=fda, fds=fds, dra=dra, drs=drs, ra=rwa, rs=rws, bta=bta,
                            bts=bts, drift_sigma=np.float64(drift_sigma), steps=np.int64(steps))
    nan = float("nan")
    if len(Ha) < 5 or len(Hs) < 5:
        out = {"target": nan, "target_lo": nan, "target_hi": nan,
               "target_var": nan, "target_var_lo": nan, "target_var_hi": nan,
               "target_full": nan, "target_full_lo": nan, "target_full_hi": nan,
               "shuffled": nan, "shuffled_var": nan, "shuffled_full": nan,
               "selectivity": nan, "selectivity_var": nan, "selectivity_full": nan,
               "speed": nan, "anchor_energy": nan, "anchor_food": nan, "ceiling_drag": nan,
               "pool_reward_leak": nan, "pool_leak_clean": False, "pool_leak_max_dev": nan,
               "pool_n_eps": n_eps, "deaths_auth": n_eps - len(Ha), "deaths_surr": n_eps - len(Hs),
               "n": len(Ha) + len(Hs), "too_few_survivors": True}
        return (out, (Ha, Hs)) if return_pools else out
    H = np.concatenate([Ha, Hs])
    X = episode_features(H)                          # reuse Exp B's feature builder verbatim
    Xv = episode_features_var(H)                     # dispersion (volatility signature)
    Xf = episode_features_full(H)                    # level ++ dispersion
    y = np.concatenate([np.zeros(len(Ha)), np.ones(len(Hs))]).astype(int)
    spd = np.concatenate([spa, sps])
    en = np.concatenate([ena, ens])
    fd = np.concatenate([fda, fds])
    rng = np.random.default_rng(seed)
    tgt, t_lo, t_hi = _auroc_with_ci(X, y, seed=seed)
    tgt_v, tv_lo, tv_hi = _auroc_with_ci(Xv, y, seed=seed)
    tgt_f, tf_lo, tf_hi = _auroc_with_ci(Xf, y, seed=seed)
    # One shared label permutation across feature sets so selectivity gaps reflect
    # feature-set overfitting bias, not permutation noise. selectivity = target - shuffled
    # is the estimand that survives the L0>0.5 probe-bias offset seen in some seeds.
    y_perm = rng.permutation(y)
    shuf = probe_auroc(X, y_perm)
    shuf_v = probe_auroc(Xv, y_perm)
    shuf_f = probe_auroc(Xf, y_perm)
    # Drag-tracking ceiling: decode high- vs low-drift-drag episodes from h_t WITHIN the
    # surrogate pool only (all surrogate, so this is NOT the world label). High here +
    # chance target = the interesting null (the state tracks the dynamics moment-to-moment
    # but forms no persistent world-identity direction). Chance here = the agent isn't
    # tracking drag at all, so a chance target would be uninformative.
    if len(Hs) >= 10 and float(np.ptp(drs)) > 1e-6:
        ceiling_drag = probe_auroc(episode_features(Hs), (drs > np.median(drs)).astype(int))
    else:
        ceiling_drag = float("nan")
    # POOLED leakage audit (the headline endpoint's missing confound gate). Reuse the
    # matched-pair battery verbatim on the pooled episodes: world identity must NOT be
    # decodable from summed reward. length/lifetime are constant (=steps for every
    # full-length survivor), so those channels sit at ~0.5 by construction and reward_sum
    # is the one live channel. Near-chance here = the pooled target reads the dynamics
    # artifact, not a reward confound induced by the surrogate.
    auth_eps = [{"label": 0, "reward_sum": float(r), "length": steps, "lifetime": steps} for r in rwa]
    surr_eps = [{"label": 1, "reward_sum": float(r), "length": steps, "lifetime": steps} for r in rws]
    pool_leak = leakage_audit_b2(auth_eps, surr_eps, margin=leak_margin)
    out = {
        "target": tgt, "target_lo": t_lo, "target_hi": t_hi,
        "target_var": tgt_v, "target_var_lo": tv_lo, "target_var_hi": tv_hi,
        "target_full": tgt_f, "target_full_lo": tf_lo, "target_full_hi": tf_hi,
        "shuffled": shuf, "shuffled_var": shuf_v, "shuffled_full": shuf_f,
        "selectivity": tgt - shuf, "selectivity_var": tgt_v - shuf_v,
        "selectivity_full": tgt_f - shuf_f,
        "speed": probe_auroc(X, (spd > np.median(spd)).astype(int)),
        "anchor_energy": probe_auroc(X, (en > np.median(en)).astype(int)),
        "anchor_food": probe_auroc(X, (fd > np.median(fd)).astype(int)),
        "ceiling_drag": ceiling_drag,
        "pool_reward_leak": pool_leak["reward_sum"], "pool_leak_clean": bool(pool_leak["clean"]),
        "pool_leak_max_dev": pool_leak["max_abs_dev"], "pool_leakage": pool_leak,
        "pool_n_eps": n_eps, "deaths_auth": n_eps - len(Ha), "deaths_surr": n_eps - len(Hs),
        "n": len(y), "n_auth": len(Ha), "n_surr": len(Hs), "too_few_survivors": False,
    }
    return (out, (Ha, Hs)) if return_pools else out


def transfer_readout(agent, norm, params, drift_sigma, Ha_train, Hs_train, *,
                     n_eps=110, steps=24, ray_steps=5, device=None, seed=0,
                     dump_path=None, heldout=None, seed_base_auth=860_000,
                     seed_base_surr=870_000) -> dict:
    """Unseen-fingerprint transfer channel (spec 2026-07-14). Fits the standard
    linear probe on the TRAINED-fingerprint pools (Ha_train vs Hs_train, i.e. the
    exact pools pooled_readout probed), then scores it FROZEN on a fresh authentic
    pool vs a pool collected under the resolved held-out surrogate (the `heldout=`
    argument, or _L3_GMOTION_HELDOUT if none was passed). Fresh authentic pool:
    the probe must never be tested on authentic episodes it trained on. Restores
    the training surrogate in a finally block so the global can never leak into
    later evals."""
    global _L3_GMOTION
    heldout = heldout if heldout is not None else _L3_GMOTION_HELDOUT
    if heldout is None:
        raise RuntimeError("transfer_readout: pass heldout= or call setup_l3_heldout_surrogate() first")
    device = device or default_device()
    saved = _L3_GMOTION
    try:
        _L3_GMOTION = heldout
        # auth pool deliberately collected inside the swap: authentic worlds never
        # attach a G (make_world guards drift_sigma>0), and keeping both collections
        # here keeps swap/restore in one place - do not "simplify" it out.
        Ha2, _ = collect_pool(agent, norm, params, 0.0, n_eps, steps, device, seed_base_auth, ray_steps)
        H7, _ = collect_pool(agent, norm, params, drift_sigma, n_eps, steps, device, seed_base_surr, ray_steps)
    finally:
        _L3_GMOTION = saved
    if dump_path is not None:
        d = os.path.dirname(dump_path)
        if d:
            os.makedirs(d, exist_ok=True)
        np.savez_compressed(dump_path, Ha2=Ha2, H7=H7,
                            drift_sigma=np.float64(drift_sigma), steps=np.int64(steps))
    nan = float("nan")
    out = {"transfer_n_auth": int(len(Ha2)), "transfer_n_surr": int(len(H7)),
           "transfer_deaths_auth": int(n_eps - len(Ha2)),
           "transfer_deaths_surr": int(n_eps - len(H7)),
           "transfer_target": nan, "transfer_lo": nan, "transfer_hi": nan}
    if len(Ha_train) < 5 or len(Hs_train) < 5 or len(Ha2) < 5 or len(H7) < 5:
        return out
    Xtr = episode_features(np.concatenate([Ha_train, Hs_train]))
    ytr = np.concatenate([np.zeros(len(Ha_train)), np.ones(len(Hs_train))]).astype(int)
    Xte = episode_features(np.concatenate([Ha2, H7]))
    yte = np.concatenate([np.zeros(len(Ha2)), np.ones(len(H7))]).astype(int)
    auc, yv, pv = transfer_probe(Xtr, ytr, Xte, yte, return_scores=True)
    out["transfer_target"] = auc
    if pv.size:
        out["transfer_lo"], out["transfer_hi"] = auroc_ci(yv, pv, seed=seed)
    return out


def common_garden_rollout(agent, norm, params, drift_sigma, *, n_pairs=110,
                          prefix_steps=20, tail_steps=24, ray_steps=5,
                          device=None, seed_base=930_000) -> tuple[list, list]:
    """Common-garden channel (spec 2026-07-14): PAIRED episodes from identical
    seeds run their prefix in the authentic vs the surrogate world, then BOTH
    continue under authentic dynamics (fresh authentic world restored from each
    prefix's final snapshot, drift_w forced to 0). Returns (auth_tails,
    surr_tails): lists of (tail_steps, hidden) arrays, tail-only states. A pair
    is dropped if EITHER member dies in prefix or tail (symmetric, so
    survivorship cannot create asymmetry)."""
    device = device or default_device()
    auth_tails, surr_tails = [], []
    for p in range(n_pairs):
        seeds = _seeds(seed_base + p)
        pair = []
        for dsig in (0.0, drift_sigma):
            w = make_world(params, dsig, ray_steps)
            w.reset(seeds)
            h = agent.initial_state(1, device)
            prev = torch.zeros(1, agent.act_dim, device=device)
            obs = w.observe().astype(np.float64)
            died = False
            for _ in range(prefix_steps):
                obs_t = torch.as_tensor(norm(obs)[None], dtype=torch.float32, device=device)
                _, env_act, _, _, h = agent.act(obs_t, prev, h, deterministic=True)
                r = w.step(env_act[0].detach().cpu().numpy().astype(np.float32))
                obs = r.obs.astype(np.float64)
                prev = env_act
                if r.terminated:
                    died = True
                    break
            if died:
                pair.append(None)
                continue
            tail = make_world(params, 0.0, ray_steps)   # common garden: authentic dynamics
            # reset only populates tail._rng for the key filter below;
            # set_state then overwrites the whole state - do not remove
            tail.reset(seeds)
            snap = w.get_state()
            # authentic tail world has no "drift" RNG; strip surplus keys so set_state
            # can restore only the RNG slots that exist in the tail world
            snap["rng"] = {k: v for k, v in snap["rng"].items() if k in tail._rng}
            tail.set_state({**snap, "drift_w": 0.0})
            Ht, _, _, alive = _run_branch(agent, norm, tail, h, prev, tail_steps, device)
            pair.append(Ht if (alive and len(Ht) == tail_steps) else None)
        if pair[0] is not None and pair[1] is not None:
            auth_tails.append(pair[0])
            surr_tails.append(pair[1])
    return auth_tails, surr_tails


def cg_probe(auth_tails: list, surr_tails: list, *, late_k: int = 8, seed: int = 0) -> dict:
    """Probe tail-only states for the PREFIX world. cg_tail_target uses the full
    tail's [mean h, final h]; cg_latetail_target repeats it on the last late_k
    steps only - the persistence-decay check (a reactive signal washes out along
    the tail; a persistent representation does not)."""
    n = len(auth_tails)
    out = {"cg_n_pairs": n, "cg_tail_target": float("nan"), "cg_tail_lo": float("nan"),
           "cg_tail_hi": float("nan"), "cg_latetail_target": float("nan")}
    if n < 5:
        return out
    y = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)
    # Matched pairs share the prefix world state: same group id for both members so
    # GroupKFold keeps each pair in one fold (see probe_world_identity).
    g = np.concatenate([np.arange(n), np.arange(n)])
    X = np.stack([_episode_feature(H) for H in auth_tails + surr_tails])
    k = min(late_k, auth_tails[0].shape[0])
    Xl = np.stack([_episode_feature(H[-k:]) for H in auth_tails + surr_tails])
    tgt, lo, hi = _auroc_with_ci(X, y, seed=seed, groups=g)
    out.update(cg_tail_target=tgt, cg_tail_lo=lo, cg_tail_hi=hi,
               cg_latetail_target=grouped_auroc(Xl, y, g))
    return out


def save_agent_bundle(path: str, agent: RecurrentActorCritic, norm: RunningNorm) -> None:
    """Persist a frozen agent + its frozen obs normalizer with the constructor args
    needed to rebuild it. A few MB; prevents ever again losing trained agents."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    assert isinstance(agent.encoder[0], torch.nn.Linear), "encoder[0] must be the embed Linear"
    # embed is not a stored attribute; encoder[0] is the first Linear by
    # construction (agent_ac.py) - if the encoder ever gains a pre-layer,
    # this recovery breaks loudly via the assert below
    torch.save({"state_dict": agent.state_dict(),
                "ctor": {"obs_dim": agent.obs_dim, "act_dim": agent.act_dim,
                         "embed": agent.encoder[0].out_features, "hidden": agent.hidden,
                         "world_model": agent.world_model, "sysid_aux": agent.sysid_aux},
                "norm": {"mean": norm.mean, "var": norm.var, "count": norm.count}}, path)


def load_agent_bundle(path: str, device: str = "cpu"):
    """Rebuild (agent, norm) from save_agent_bundle output. Returns them frozen."""
    # weights_only=False is required: the bundle holds a plain dict of numpy
    # arrays (norm state), which weights_only=True rejects. Own artifacts only.
    blob = torch.load(path, map_location=device, weights_only=False)
    agent = RecurrentActorCritic(**blob["ctor"]).to(device)
    agent.load_state_dict(blob["state_dict"])
    norm = RunningNorm(blob["ctor"]["obs_dim"])
    norm.mean, norm.var, norm.count = blob["norm"]["mean"], blob["norm"]["var"], blob["norm"]["count"]
    return agent.train(False), norm.freeze()


def readout(agent, norm, params, drift_sigma, *, n_pairs=60, prefix_steps=20, branch_steps=24,
            ray_steps=5, device=None, seed_base=700_000, seed=0, leak_margin=0.1) -> dict:
    """Run the SECONDARY matched-pair recurrent readout (detectability-style) and return
    probe + leakage in one dict. The leakage gate stays at |dev|<0.1: this is the matched-
    pair path where reward_sum legitimately differs under drift, so a tighter gate would
    false-alarm on real return differences rather than probe contamination. The numeric
    leakage_max_dev is always reported for transparency."""
    a, s = matched_pair_recurrent_rollout(agent, norm, params, drift_sigma, n_pairs=n_pairs,
                                          prefix_steps=prefix_steps, branch_steps=branch_steps,
                                          ray_steps=ray_steps, device=device, seed_base=seed_base)
    pr = probe_world_identity(a, s, seed=seed)
    g = np.concatenate([np.arange(len(a)), np.arange(len(s))])
    lk = leakage_audit_b2(a, s, margin=leak_margin, groups=g)
    return {**pr, "leakage_clean": lk["clean"], "leakage_max_dev": lk["max_abs_dev"],
            "leakage": lk, "n_pairs": len(a)}


if __name__ == "__main__":
    # tiny smoke test - confirms the whole 3-agent apparatus executes end to end.
    # (tiny scale -> numbers are NOT the result, only proof the machinery runs)
    dev = default_device()
    D = 0.45
    print(f"Experiment B-v2 smoke test  (device = {dev}, drift={D})")
    P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
    RS, NP, PF, BR = 4, 20, 12, 16

    untr_a, untr_n = untrained_agent(P, D, RS, hidden=64, embed=64, world_model=True, device=dev, seed=0)
    r_untr = readout(untr_a, untr_n, P, D, n_pairs=NP, prefix_steps=PF, branch_steps=BR, ray_steps=RS, device=dev)

    pred_a, pred_n = train_predictor_only(D, P, n_eps=8, updates=10, hidden=64, max_steps=40,
                                          ray_steps=RS, seed=0, device=dev)
    r_pred = readout(pred_a, pred_n, P, D, n_pairs=NP, prefix_steps=PF, branch_steps=BR, ray_steps=RS, device=dev)

    surv_a, surv_n, _ = train_actor_critic(D, P, n_eps=8, updates=10, hidden=64, max_steps=40,
                                           ray_steps=RS, seed=0, device=dev, log_every=5)
    eng = engagement_metric(surv_a, surv_n, P, D, n_eps=16, max_steps=40, ray_steps=RS, device=dev)
    r_surv = readout(surv_a, surv_n, P, D, n_pairs=NP, prefix_steps=PF, branch_steps=BR, ray_steps=RS, device=dev)

    p_untr = pooled_readout(untr_a, untr_n, P, D, n_eps=40, steps=BR, ray_steps=RS, device=dev)
    p_pred = pooled_readout(pred_a, pred_n, P, D, n_eps=40, steps=BR, ray_steps=RS, device=dev)
    p_surv = pooled_readout(surv_a, surv_n, P, D, n_eps=40, steps=BR, ray_steps=RS, device=dev)

    print(f"   engagement: trained={eng['trained_return']:+.3f} random={eng['random_return']:+.3f} "
          f"scripted={eng['scripted_return']:+.3f}  engaged={eng['engaged']}")
    print("   PRIMARY pooled readout - target AUROC (~0.5 = no persistent encoding, Exp B frame):")
    print("     (ceiling = energy/food/drag; high drag-ceiling + chance target = tracks dynamics, no identity)")
    for name, r in (("untrained ", p_untr), ("predictor ", p_pred), ("survival  ", p_surv)):
        print(f"     {name}: target={r['target']:.3f} [{r.get('target_lo', float('nan')):.3f},"
              f"{r.get('target_hi', float('nan')):.3f}]  ceiling(energy={r.get('anchor_energy', float('nan')):.3f} "
              f"food={r.get('anchor_food', float('nan')):.3f} drag={r.get('ceiling_drag', float('nan')):.3f})  "
              f"speed(+)={r['speed']:.3f}  shuffled={r['shuffled']:.3f}  (n={r['n']})")
    print("   SECONDARY matched-pair readout - target AUROC (|dev from 0.5| = detectability):")
    for name, r in (("untrained ", r_untr), ("predictor ", r_pred), ("survival  ", r_surv)):
        print(f"     {name}: target={r['target']:.3f}  speed(+)={r['speed']:.3f}  "
              f"leak_clean={r['leakage_clean']} (n={r['n_pairs']})")

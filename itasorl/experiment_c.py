"""ITASORL - Experiment C world-coupled seams (fitness + detection panel).

The pure generational engine lives in ``itasorl.neuroevolution`` (mutation,
threshold-triggered reproduction, the ``evolve`` loop) and knows nothing about
worlds. This module supplies the two injected callables that ``evolve`` needs to
become an actual experiment, reusing the B-v2 rollout harness:

  * ``mixed_world_fitness`` -> the ``evaluate_fitness`` seam. Scores each policy
    by its lifetime foraging return over a MIXED authentic+surrogate lifetime
    (docs/PREREGISTRATION_C.md sec. 4). No within-life gradient step (Option P,
    sec. 3): a policy is just evaluated, never trained.
  * ``common_garden_panel`` -> the ``observe`` seam. A fixed-horizon common-garden
    detection measurement identical every generation (sec. 5): world-identity AUROC
    off the pooled recurrent tail states, with survival reported as a SEPARATE
    series and an L0 authentic-vs-authentic floor.

Determinism (the milestone-1 bar, sec. 13): rollouts use ``deterministic=True``
(mode actions, no policy sampling) and ``update_norm=False`` (the obs normalizer
never mutates mid-eval), so a fixed seed reproduces to the bit.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .agent_ac import RecurrentActorCritic
from .experiment_a import grouped_auroc
from .experiment_b2 import (RunningNorm, _episode_feature, cg_probe,
                            collect_episodes_ac, common_garden_rollout,
                            leakage_audit_b2)
from .stats import mean_ci
from .world import WorldParams

Population = list[RecurrentActorCritic]


def _t_ci90(x: np.ndarray) -> tuple[float, float]:
    """90% two-sided t interval on the mean of ``x`` (df=n-1). Student-t if scipy is
    present, else a normal approximation; [nan, nan] for n<2 (no variance)."""
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 2:
        return (float("nan"), float("nan"))
    mean = float(x.mean())
    se = float(x.std(ddof=1)) / np.sqrt(n)
    try:
        from scipy.stats import t as _student_t
        crit = float(_student_t.ppf(0.95, n - 1))
    except Exception:  # pragma: no cover - scipy optional
        crit = 1.645
    return (mean - crit * se, mean + crit * se)


def emergence_contrast(
    gen0_auroc,
    final_treat_auroc,
    final_ctrl_auroc,
    *,
    sesoi: float = 0.05,
    auroc_floor: float = 0.65,
    n_boot: int = 10_000,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """The pre-registered emergence estimand (docs/PREREGISTRATION_C.md sec. 6/10).

    Per lineage seed, ``Delta = AUROC(final) - AUROC(gen0)``; the reported effect is
    the treatment-minus-control contrast of those deltas. The three inputs are arrays
    over lineage seeds (both arms share the gen-0 population, so gen0 cancels and the
    contrast equals ``final_treat - final_ctrl`` - but we keep the delta-of-deltas
    form so it stays correct if the arms ever diverge at gen 0). Seeds are the
    replication unit (Colas et al.), so both intervals are ACROSS-SEED.

    ``emergence_claim`` fires only when ALL three pre-registered conditions hold:
    the (t-based) contrast CI excludes 0, the mean contrast reaches the SESOI, and
    the treatment's mean final AUROC clears the ``auroc_floor``. A pilot with 3 seeds
    will usually not clear the CI bar (df=2 is wide by design) - that is expected;
    the pieces are all returned so the confirmatory-scale decision can be made on
    the raw evidence, not a single boolean.
    """
    gen0 = np.asarray(gen0_auroc, dtype=float)
    ft = np.asarray(final_treat_auroc, dtype=float)
    fc = np.asarray(final_ctrl_auroc, dtype=float)
    delta_treat = ft - gen0
    delta_ctrl = fc - gen0
    contrast = delta_treat - delta_ctrl
    mean_contrast = float(contrast.mean())
    t_lo, t_hi = _t_ci90(contrast)
    seed = int(rng.integers(0, 2**31 - 1)) if rng is not None else 0
    _, b_lo, b_hi = mean_ci(contrast, level=0.90, n_boot=n_boot, seed=seed)
    mean_final_treat = float(ft.mean())

    ci_excludes_zero = bool(np.isfinite(t_lo) and t_lo > 0.0)
    meets_sesoi = bool(mean_contrast >= sesoi)
    meets_floor = bool(mean_final_treat >= auroc_floor)
    return {
        "delta_treat": delta_treat.tolist(),
        "delta_ctrl": delta_ctrl.tolist(),
        "contrast": contrast.tolist(),
        "mean_contrast": mean_contrast,
        "t_ci90": [t_lo, t_hi],
        "boot_ci90": [b_lo, b_hi],
        "mean_final_treat_auroc": mean_final_treat,
        "n_seeds": int(contrast.size),
        "ci_excludes_zero": ci_excludes_zero,
        "meets_sesoi": meets_sesoi,
        "meets_auroc_floor": meets_floor,
        "emergence_claim": bool(ci_excludes_zero and meets_sesoi and meets_floor),
    }


def mixed_world_fitness(
    population: Population,
    *,
    drift_sigma: float,
    n_eps_per_world: int,
    max_steps: int,
    seed_base: int,
    params: WorldParams | None = None,
    ray_steps: int = 5,
    device: str = "cpu",
    norm: RunningNorm | None = None,
    food_override: dict | None = None,
) -> np.ndarray:
    """Per-policy fitness = mean lifetime foraging return over a mixed-world lifetime.

    Each policy lives ``n_eps_per_world`` authentic (drift_sigma=0) and
    ``n_eps_per_world`` surrogate (drift_sigma>0) episodes; fitness is the mean
    TRUE return (``batch["ret"]``) pooled across both. Authentic and surrogate
    share ``seed_base`` so the two lifetimes differ only in the velocity law - the
    one primitive that separates the worlds (sec. 4). Deterministic given the seed.

    ``food_override`` selects the pellet layout (sec. 11): None -> the treatment's
    frozen sparse/far layout; a dense/near dict -> the world-INVARIANT control arm
    where detecting the world buys no fitness. It rides identically on both the
    authentic and surrogate legs, so it never itself becomes a world cue.
    """
    fits = np.empty(len(population), dtype=float)
    for i, agent in enumerate(population):
        agent.train(False)
        n = norm or RunningNorm(agent.obs_dim)
        rets = []
        for dsig in (0.0, drift_sigma):
            batch = collect_episodes_ac(
                agent, n, params, dsig, n_eps=n_eps_per_world, max_steps=max_steps,
                device=device, seed_base=seed_base, ray_steps=ray_steps,
                deterministic=True, update_norm=False, food_override=food_override,
            )
            rets.append(np.asarray(batch["ret"], dtype=float))
        fits[i] = float(np.concatenate(rets).mean())
    return fits


def _pooled_common_garden(
    sample: Population, drift_sigma: float, *, params, ray_steps, device, norm,
    n_pairs, prefix_steps, tail_steps, seed_base,
) -> tuple[list, list, list, list]:
    """Run the shared-prefix authentic-vs-``drift_sigma`` common garden for every
    sampled policy and POOL the surviving tail states (and per-pair tail stats,
    for the gate battery) into one detection sample."""
    auth: list = []
    surr: list = []
    auth_st: list = []
    surr_st: list = []
    for agent in sample:
        agent.train(False)
        n = norm or RunningNorm(agent.obs_dim)
        a, s, ast, sst = common_garden_rollout(
            agent, n, params, drift_sigma, n_pairs=n_pairs, prefix_steps=prefix_steps,
            tail_steps=tail_steps, ray_steps=ray_steps, device=device, seed_base=seed_base,
            return_stats=True,
        )
        auth.extend(a)
        surr.extend(s)
        auth_st.extend(ast)
        surr_st.extend(sst)
    return auth, surr, auth_st, surr_st


def _survival_by_world(
    sample: Population, drift_sigma: float, *, params, ray_steps, device, norm,
    n_eps, max_steps, seed_base,
) -> dict[str, float]:
    """Death rate and mean lifetime per world, pooled over the sample. A SEPARATE
    series from detection (sec. 5) - computed from its own foraging rollouts and
    never fed to the probe."""
    out: dict[str, float] = {}
    for label, dsig in (("auth", 0.0), ("surr", drift_sigma)):
        term: list = []
        lens: list = []
        for agent in sample:
            agent.train(False)
            n = norm or RunningNorm(agent.obs_dim)
            b = collect_episodes_ac(
                agent, n, params, dsig, n_eps=n_eps, max_steps=max_steps, device=device,
                seed_base=seed_base, ray_steps=ray_steps, deterministic=True, update_norm=False,
            )
            term.append(b["terminated"].cpu().numpy().reshape(-1))
            lens.append(np.asarray(b["lengths"]).reshape(-1))
        out[f"death_rate_{label}"] = float(np.concatenate(term).mean())
        out[f"mean_len_{label}"] = float(np.concatenate(lens).mean())
    return out


def common_garden_panel(
    population: Population,
    gen: int = 0,
    *,
    drift_sigma: float,
    n_pairs: int,
    prefix_steps: int,
    tail_steps: int,
    seed_base: int,
    params: WorldParams | None = None,
    ray_steps: int = 5,
    device: str = "cpu",
    norm: RunningNorm | None = None,
    sample_size: int | None = None,
    late_k: int = 8,
    surv_n_eps: int = 2,
    surv_max_steps: int | None = None,
    surv_seed_base: int | None = None,
) -> dict[str, Any]:
    """One fixed-horizon common-garden detection panel (docs/PREREGISTRATION_C.md sec. 5).

    Identical every generation and independent of who lived or died: a fixed set of
    matched authentic-vs-surrogate pairs is run for a fixed ``sample`` of the current
    population and the world-identity probe reads ``[mean h, final h]`` off the pooled
    tail states. Survival (death rate / mean lifetime per world) is measured on its
    OWN foraging rollouts and returned under a separate ``survival`` key, never pooled
    into the detection estimate. An L0 authentic-vs-authentic floor is measured every
    generation. Deterministic given the seeds.
    """
    sample = population if sample_size is None else population[:sample_size]
    surv_max_steps = surv_max_steps or (prefix_steps + tail_steps)
    surv_seed_base = surv_seed_base if surv_seed_base is not None else seed_base + 1_000_000
    cg_kw = dict(params=params, ray_steps=ray_steps, device=device, norm=norm,
                 n_pairs=n_pairs, prefix_steps=prefix_steps, tail_steps=tail_steps,
                 seed_base=seed_base)

    auth, surr, a_st, s_st = _pooled_common_garden(sample, drift_sigma, **cg_kw)
    det = cg_probe(auth, surr, late_k=late_k)
    a0, s0, _, _ = _pooled_common_garden(sample, 0.0, **cg_kw)
    l0 = cg_probe(a0, s0, late_k=late_k)
    survival = _survival_by_world(
        sample, drift_sigma, params=params, ray_steps=ray_steps, device=device, norm=norm,
        n_eps=surv_n_eps, max_steps=surv_max_steps, seed_base=surv_seed_base,
    )
    # Gate battery on the SAME pooled tails (PREREGISTRATION_C sec. 7 gates 4-5):
    #   gate 4 - leakage: world identity must not be decodable from tail reward
    #     (length/lifetime are constant for surviving pairs by construction, so
    #     those channels sit at 0.5 structurally; reward_sum is the live channel);
    #   gate 5 - positive control: the probe must read a known signal (median
    #     tail speed) at >= 0.75, or a chance detection value is uninformative.
    # Pair members share a group id (matched pairs, same rule as cg_probe).
    leak: dict[str, Any] = {}
    speed_control = float("nan")
    if len(a_st) >= 5:
        g = np.concatenate([np.arange(len(a_st)), np.arange(len(s_st))])
        leak = leakage_audit_b2(a_st, s_st, groups=g)
        X = np.stack([_episode_feature(H) for H in auth + surr])
        spd = np.array([e["speed"] for e in a_st + s_st])
        if float(np.ptp(spd)) > 1e-9:
            speed_control = grouped_auroc(X, (spd > np.median(spd)).astype(int), g)
    return {
        "gen": int(gen),
        "cg_n_pairs": det["cg_n_pairs"],
        "cg_tail_target": det["cg_tail_target"],
        "cg_tail_lo": det["cg_tail_lo"],
        "cg_tail_hi": det["cg_tail_hi"],
        "cg_latetail_target": det["cg_latetail_target"],
        "l0_auroc": l0["cg_tail_target"],
        "l0_latetail": l0["cg_latetail_target"],
        "l0_n_pairs": l0["cg_n_pairs"],
        "survival": survival,
        "leakage": {k: v for k, v in leak.items() if k != "margin"},
        "leak_clean": bool(leak.get("clean", False)),
        "speed_control": float(speed_control),
        "speed_control_pass": bool(speed_control >= 0.75) if np.isfinite(speed_control) else False,
    }

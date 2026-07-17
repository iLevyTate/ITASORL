"""Experiment C world-coupled seams: mixed-world fitness + common-garden panel.

These are the two injected seams that turn the pure generational loop
(``itasorl.neuroevolution.evolve``) into an actual world experiment. Unlike the
neuroevolution primitives they DO spin up PatchOfEarth rollouts, so every test
here is held at tiny scale (population of 2-3, a handful of episodes, short
horizons) - the numbers are never the result, only proof the machinery runs and
is bit-reproducible (the milestone-1 bar, docs/PREREGISTRATION_C.md sec. 13).
"""

from functools import partial

import numpy as np
import torch

from itasorl.agent_ac import RecurrentActorCritic
from itasorl.patch_of_earth import PatchOfEarthV0

_W = PatchOfEarthV0()
_OBS, _ACT = _W.obs_spec.size, _W.action_spec.size


def _world_pop(n: int) -> list[RecurrentActorCritic]:
    """n tiny policies sized to the real world (obs=146, act=5), each seeded
    distinctly so they forage differently - the raw material selection acts on."""
    pop = []
    for i in range(n):
        torch.manual_seed(500 + i)
        pop.append(RecurrentActorCritic(_OBS, _ACT, embed=4, hidden=4, world_model=False))
    return pop


def test_mixed_world_fitness_shape_and_determinism():
    """One finite fitness per agent, bit-identical across two calls on a fixed seed."""
    from itasorl.experiment_c import mixed_world_fitness

    pop = _world_pop(3)
    kw = dict(drift_sigma=0.02, n_eps_per_world=2, max_steps=6, seed_base=310_000)
    f1 = mixed_world_fitness(pop, **kw)
    f2 = mixed_world_fitness(pop, **kw)
    assert isinstance(f1, np.ndarray) and f1.shape == (3,)
    assert np.all(np.isfinite(f1))
    assert np.array_equal(f1, f2), "fixed seed must give bit-identical fitness"


def test_make_world_food_override_is_additive_and_default_byte_identical():
    """The control arm needs a world-INVARIANT (dense/near) pellet layout while the
    treatment keeps the frozen sparse/far one. `food_override` supplies that as an
    ADDITIVE merge on top of SURVIVAL_FOOD, so only the named fields move and the
    default path (None / omitted) is byte-identical to the frozen behavior every
    other experiment depends on (sec. 11 - the control is the claim)."""
    from itasorl.experiment_b2 import make_world
    from itasorl.world import WorldParams

    P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
    base = make_world(P, 0.0, 5)
    assert base.reach == 0.08 and base.n_pellets == 24, "frozen SURVIVAL_FOOD default"

    ovr = make_world(P, 0.0, 5, food_override={"reach": 0.5, "n_pellets": 60})
    assert ovr.reach == 0.5 and ovr.n_pellets == 60, "override reaches the layout"
    assert ovr.pellet_r == base.pellet_r, "unnamed fields keep the frozen value (additive)"

    none = make_world(P, 0.0, 5, food_override=None)
    assert none.reach == base.reach and none.n_pellets == base.n_pellets, \
        "food_override=None must be byte-identical to omission"


def test_mixed_world_fitness_threads_food_override_default_byte_identical():
    """`mixed_world_fitness` must pass `food_override` all the way down to the
    rollout: the default is byte-identical to omission (protects the frozen infra),
    and a genuinely different layout produces a different fitness vector (proving it
    is not silently dropped at the seam)."""
    from itasorl.experiment_c import mixed_world_fitness

    pop = _world_pop(3)
    kw = dict(drift_sigma=0.02, n_eps_per_world=2, max_steps=6, seed_base=312_000)
    f_omit = mixed_world_fitness(pop, **kw)
    f_none = mixed_world_fitness(pop, food_override=None, **kw)
    assert np.array_equal(f_omit, f_none), "food_override=None must equal omission to the bit"

    f_dense = mixed_world_fitness(pop, food_override={"reach": 0.5, "n_pellets": 60}, **kw)
    assert not np.array_equal(f_omit, f_dense), "override must actually reach the rollout"


def test_mixed_world_fitness_scores_agents_independently():
    """Permuting the population permutes the fitness identically - each policy is
    scored on the same worlds with no cross-agent state leakage."""
    from itasorl.experiment_c import mixed_world_fitness

    pop = _world_pop(3)
    kw = dict(drift_sigma=0.02, n_eps_per_world=2, max_steps=6, seed_base=311_000)
    f = mixed_world_fitness(pop, **kw)
    perm = [2, 0, 1]
    fp = mixed_world_fitness([pop[i] for i in perm], **kw)
    assert np.array_equal(fp, f[perm])


_PANEL_KW = dict(drift_sigma=0.02, n_pairs=6, prefix_steps=5, tail_steps=6,
                 seed_base=930_000)


def test_emergence_contrast_delta_of_deltas_and_claim_rule():
    """The pre-registered estimand (sec. 6/10): per-lineage
    Delta = AUROC(final) - AUROC(gen0), then the treatment-minus-control contrast.
    Because both arms share the gen-0 population per lineage, gen0 cancels and the
    contrast reduces to final_treat - final_ctrl. The emergence claim fires only
    when ALL three pre-registered conditions hold: contrast CI excludes 0, mean
    contrast >= SESOI(0.05), and mean treatment final AUROC >= 0.65."""
    from itasorl.experiment_c import emergence_contrast

    gen0 = [0.50, 0.50, 0.50]
    final_treat = [0.72, 0.75, 0.70]
    final_ctrl = [0.52, 0.50, 0.51]
    r = emergence_contrast(gen0, final_treat, final_ctrl, rng=np.random.default_rng(0))

    assert np.allclose(r["contrast"], [0.72 - 0.52, 0.75 - 0.50, 0.70 - 0.51])
    assert abs(r["mean_contrast"] - np.mean(r["contrast"])) < 1e-12
    assert r["ci_excludes_zero"] is True
    assert r["meets_sesoi"] is True
    assert r["meets_auroc_floor"] is True
    assert r["emergence_claim"] is True


def test_emergence_contrast_null_when_arms_equal():
    """When the arms end at the same detection level the contrast is ~0: no SESOI,
    CI spans 0, no emergence claim - the guard against a false positive."""
    from itasorl.experiment_c import emergence_contrast

    gen0 = [0.50, 0.50, 0.50]
    same = [0.55, 0.50, 0.52]
    r = emergence_contrast(gen0, same, same, rng=np.random.default_rng(0))
    assert r["contrast"] == [0.0, 0.0, 0.0]
    assert r["mean_contrast"] == 0.0
    assert r["meets_sesoi"] is False
    assert r["emergence_claim"] is False


def test_emergence_contrast_bootstrap_is_reproducible():
    """Fixed rng -> bit-identical bootstrap CI (the determinism bar)."""
    from itasorl.experiment_c import emergence_contrast

    a = emergence_contrast([0.5, 0.5, 0.5], [0.7, 0.65, 0.72], [0.5, 0.52, 0.49],
                           rng=np.random.default_rng(7))
    b = emergence_contrast([0.5, 0.5, 0.5], [0.7, 0.65, 0.72], [0.5, 0.52, 0.49],
                           rng=np.random.default_rng(7))
    assert a["boot_ci90"] == b["boot_ci90"]


def test_common_garden_panel_reports_detection_and_survival_separately():
    """The panel returns a detection AUROC AND a survival series, structurally
    separated so survival can never leak into the detection estimate (sec. 5)."""
    from itasorl.experiment_c import common_garden_panel

    pop = _world_pop(2)
    rec = common_garden_panel(pop, 0, **_PANEL_KW)
    # detection lives at the top level ...
    assert "cg_tail_target" in rec
    auroc = rec["cg_tail_target"]
    assert np.isnan(auroc) or 0.0 <= auroc <= 1.0
    # ... survival is a SEPARATE nested series, never folded into detection.
    assert isinstance(rec["survival"], dict)
    for k in ("death_rate_auth", "death_rate_surr", "mean_len_auth", "mean_len_surr"):
        assert k in rec["survival"]
    assert "cg_tail_target" not in rec["survival"]


def test_common_garden_panel_includes_l0_floor():
    """An authentic-vs-authentic sub-panel is measured every generation; its AUROC
    is the drift-in-the-apparatus floor (must stay at chance in a real run)."""
    from itasorl.experiment_c import common_garden_panel

    rec = common_garden_panel(_world_pop(2), 0, **_PANEL_KW)
    assert "l0_auroc" in rec


def test_common_garden_panel_is_bit_reproducible():
    """Fixed seed -> identical panel record (the milestone-1 determinism bar)."""
    from itasorl.experiment_c import common_garden_panel

    pop = _world_pop(2)
    r1 = common_garden_panel(pop, 0, **_PANEL_KW)
    r2 = common_garden_panel(pop, 0, **_PANEL_KW)
    assert r1 == r2


def test_evolve_runs_with_world_seams_bit_reproducibly():
    """Milestone-1 acceptance: the whole generational loop wired to the REAL
    world-coupled fitness and common-garden panel reproduces to the bit on a fixed
    seed - identical fitness history, identical panel series, identical final pop."""
    from itasorl.experiment_c import common_garden_panel, mixed_world_fitness
    from itasorl.neuroevolution import evolve

    def run():
        fit = partial(mixed_world_fitness, drift_sigma=0.02, n_eps_per_world=2,
                      max_steps=6, seed_base=320_000)
        obs = partial(common_garden_panel, drift_sigma=0.02, n_pairs=6, prefix_steps=5,
                      tail_steps=6, seed_base=930_000)
        return evolve(_world_pop(3), fit, generations=2, threshold=-10.0, sigma=0.02,
                      rng=np.random.default_rng(0), observe=obs)

    p1, h1 = run()
    p2, h2 = run()

    assert len(h1) == 2
    assert all("panel" in r and "survival" in r["panel"] for r in h1)
    assert [r["mean_fitness"] for r in h1] == [r["mean_fitness"] for r in h2]
    assert ([r["panel"]["cg_tail_target"] for r in h1]
            == [r["panel"]["cg_tail_target"] for r in h2])
    for a, b in zip(p1, p2):
        assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))

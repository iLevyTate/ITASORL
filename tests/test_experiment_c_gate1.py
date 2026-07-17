"""Experiment C gate-1 (world-only-exploitability): the scripted-controller probe.

Gate 1 (docs/PREREGISTRATION_C.md sec. 7) asks whether world identity is
FITNESS-relevant: a policy that could condition on the world must out-earn a
world-blind one in the treatment layout, while tying it in the world-invariant
control. We measure that with a scripted momentum-to-target controller (a
constant thrust toward a straight-ahead target) instead of a learned policy, so
the payoff isolates the ONE primitive that separates the worlds - the velocity
update at patch_of_earth.py:177 - with no training noise and bit-reproducibly.

These tests hold everything at tiny scale (short horizons, a handful of thrusts);
the numbers are never the result, only proof the instrument runs, is
deterministic, and has the mechanical property gate 1 relies on.
"""

import numpy as np

from itasorl.patch_of_earth import PatchOfEarthV0
from itasorl.world import WorldParams

# Thrust-dominant, spatially-uniform-drag MACHINERY config: gravity off and
# k_land==k_water, so the scripted reach is a clean 1-D thrust-vs-drag problem
# and the only thing that separates the worlds is the drag regime. This is a
# stand-in that lets the tests DEMONSTRATE the instrument (a real, robust gap in
# the treatment; exactly zero in the from-rest control); the real gate-1
# certification runs on world P with the frozen L3 map.
_MACH_PARAMS = WorldParams(gravity=0.0, k_land=0.3, k_water=0.3)
_THRUSTS_FINE = np.linspace(0.0, 1.0, 21)
_SEEDS = [7000, 7001, 7002, 7003]


def test_positive_thrust_beats_zero_thrust_toward_target():
    """A constant forward thrust must land closer to a straight-ahead target than
    no thrust at all - the payoff rewards momentum toward the goal, so a nonzero
    thrust yields a higher (less negative) payoff. Also bit-reproducible."""
    from itasorl.experiment_c_gate1 import Layout, scripted_reach_payoff

    layout = Layout(reach_range=0.15, horizon=12, name="treatment")
    w = PatchOfEarthV0(drift_sigma=0.0, drift_mode="regime")

    p_thrust = scripted_reach_payoff(w, thrust=0.8, layout=layout, seed=4100)
    p_rest = scripted_reach_payoff(w, thrust=0.0, layout=layout, seed=4100)
    assert p_thrust > p_rest, (p_thrust, p_rest)

    # deterministic: same world, same thrust, same seed -> identical payoff.
    p_again = scripted_reach_payoff(w, thrust=0.8, layout=layout, seed=4100)
    assert p_thrust == p_again


_THRUSTS = np.linspace(0.0, 1.0, 11)


def test_value_of_world_identity_treatment_has_positive_gap():
    """In the treatment layout (far target, long horizon) the drag law bites, so
    the optimal thrust DIFFERS between worlds and a world-conditional controller
    out-earns the single best world-blind thrust: gap > 0. That gap is the value
    of knowing the world - exactly what selection could exploit (sec. 7)."""
    from itasorl.experiment_c_gate1 import Layout, value_of_world_identity

    treat = Layout(reach_range=0.25, horizon=20, name="treatment")
    r = value_of_world_identity(treat, seed=7000, drift_sigma=1.0,
                                drift_mode="regime", thrusts=_THRUSTS, ray_steps=5)
    assert r["gap"] >= 0.0, "value of information can never be negative (Jensen)"
    assert r["gap"] > 0.0, "world identity must pay in the treatment"
    assert r["best_auth_thrust"] != r["best_surr_thrust"], "optimal thrust is world-dependent"

    r2 = value_of_world_identity(treat, seed=7000, drift_sigma=1.0,
                                 drift_mode="regime", thrusts=_THRUSTS, ray_steps=5)
    assert r["gap"] == r2["gap"], "fixed seed must give a bit-identical gap"


def test_value_of_world_identity_control_is_world_invariant_from_rest():
    """In the control layout (from rest, horizon 1) the first integrated step is
    a*dt regardless of drag (patch_of_earth.py:177), so the two worlds' payoff
    curves are bit-identical and the gap is EXACTLY zero - detection buys no
    fitness by construction, the yoked control gate 1 needs (sec. 7)."""
    from itasorl.experiment_c_gate1 import Layout, value_of_world_identity

    ctrl = Layout(reach_range=0.05, horizon=1, name="control")
    r = value_of_world_identity(ctrl, seed=7000, drift_sigma=1.0,
                                drift_mode="regime", thrusts=_THRUSTS, ray_steps=5)
    assert np.array_equal(r["auth_curve"], r["surr_curve"]), "from-rest curves must be identical"
    assert r["gap"] == 0.0


def _mach_kw(**over):
    kw = dict(seeds=_SEEDS, drift_sigma=1.5, drift_mode="regime", thrusts=_THRUSTS_FINE,
              params=_MACH_PARAMS, ray_steps=5, margin=0.005, tol=0.005, n_boot=500)
    kw.update(over)
    return kw


def test_gate1_passes_when_treatment_exploitable_and_control_neutral():
    """The full gate: a far/long treatment reach where world-optimal thrust
    diverges (gap CI strictly above margin) AND a from-rest control where the gap
    is exactly zero (gap CI below tol) -> gate 1 passes."""
    from itasorl.experiment_c_gate1 import Layout, gate1_exploitability

    treat = Layout(reach_range=0.3, horizon=30, name="treatment")
    ctrl = Layout(reach_range=0.05, horizon=1, name="control")
    r = gate1_exploitability(treatment=treat, control=ctrl,
                             rng=np.random.default_rng(0), **_mach_kw())
    assert r["treatment_gap_mean"] > 0.005
    assert r["control_gap_mean"] == 0.0
    assert r["passes_treatment"] is True
    assert r["passes_control"] is True
    assert r["passes_gate1"] is True


def test_gate1_fails_treatment_when_payoff_is_world_neutral():
    """Guard against a false-positive gate: if the treatment layout is actually
    world-neutral (here a from-rest layout, gap 0), the treatment side must FAIL
    and so must the overall gate - a world-blind policy could capture that payoff."""
    from itasorl.experiment_c_gate1 import Layout, gate1_exploitability

    neutral = Layout(reach_range=0.05, horizon=1, name="not-really-treatment")
    r = gate1_exploitability(treatment=neutral, control=neutral,
                             rng=np.random.default_rng(0), **_mach_kw())
    assert r["passes_treatment"] is False
    assert r["passes_gate1"] is False


def test_gate1_bootstrap_ci_is_reproducible():
    """Fixed rng -> identical bootstrap CI (the milestone-1 determinism bar)."""
    from itasorl.experiment_c_gate1 import Layout, gate1_exploitability

    treat = Layout(reach_range=0.3, horizon=30, name="treatment")
    ctrl = Layout(reach_range=0.05, horizon=1, name="control")
    kw = _mach_kw()
    r1 = gate1_exploitability(treatment=treat, control=ctrl, rng=np.random.default_rng(0), **kw)
    r2 = gate1_exploitability(treatment=treat, control=ctrl, rng=np.random.default_rng(0), **kw)
    assert r1["treatment_ci90"] == r2["treatment_ci90"]
    assert r1["control_ci90"] == r2["control_ci90"]


def test_l3_surrogate_replaces_law_while_authentic_stays_analytic():
    """The real gate-1 certification runs against the frozen L3 map, not the cheap
    regime stand-in. Under drift_mode='l3' the SURROGATE world's velocity update is
    replaced by the learned G_motion, but the AUTHENTIC world (no g_motion) must fall
    back to the analytic law (patch_of_earth.py:174-177). This pins that invariant -
    the authentic thrust->payoff curve is bit-identical whether we ask for it in
    l3-mode (auth has no g_motion) or in pure regime-mode at drift_sigma=0 - AND the
    surrogate genuinely differs (the learned map moved the payoff), with a
    non-negative, bit-reproducible gap."""
    from itasorl.experiment_c_gate1 import Layout, value_of_world_identity
    from itasorl.surrogate_l3 import train_g_motion

    # a tiny, fast frozen map (small capacity/epochs); the numbers are not the result,
    # only proof the L3 wiring is correct.
    g = train_g_motion(hidden=4, n_eps=30, steps=20, epochs=30, seed=0)
    treat = Layout(reach_range=0.25, horizon=20, name="treatment")

    r_l3 = value_of_world_identity(treat, seed=7000, drift_sigma=1.0, drift_mode="l3",
                                   thrusts=_THRUSTS, ray_steps=5, g_motion=g)
    r_analytic = value_of_world_identity(treat, seed=7000, drift_sigma=0.0,
                                         drift_mode="regime", thrusts=_THRUSTS, ray_steps=5)

    # authentic side is unchanged by l3 mode: the fallback at patch_of_earth.py:177 holds.
    assert np.array_equal(r_l3["auth_curve"], r_analytic["auth_curve"])
    # the learned map actually replaced the law on the surrogate side.
    assert not np.array_equal(r_l3["surr_curve"], r_l3["auth_curve"])
    assert r_l3["gap"] >= 0.0

    r_again = value_of_world_identity(treat, seed=7000, drift_sigma=1.0, drift_mode="l3",
                                      thrusts=_THRUSTS, ray_steps=5, g_motion=g)
    assert r_l3["gap"] == r_again["gap"]

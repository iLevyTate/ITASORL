"""Functional regression tests for the Experiment A oracle and leakage gate.

These guard the core detection methodology end to end: an L1 grid is detectable
while L0 is at chance, the detectability ceiling falls as the grid approaches the
sensor noise, and the leakage audit catches a contaminated (confounded) reward.
Deterministic - run_experiment_a and generate_clean use fixed seeds.
"""

import pytest

from experiment_a import generate_clean, run_experiment_a

pytest.importorskip("sklearn")

SIGMA = 0.01


def _clean():
    return generate_clean(n_pairs=20, steps=20, seed0=1000, ray_steps=10)


def test_L1_detectable_while_L0_at_chance():
    clean = _clean()
    l0 = run_experiment_a(clean, "L0", delta=0.06, sigma=SIGMA)
    l1 = run_experiment_a(clean, "L1", delta=0.06, sigma=SIGMA)
    assert l1["oracle_auroc"] > 0.9   # a clearly-resolved grid is detectable
    assert l0["oracle_auroc"] < 0.7   # an identical world is ~chance
    assert l1["oracle_auroc"] > l0["oracle_auroc"]


def test_detectability_ceiling_falls_as_delta_approaches_sigma():
    clean = _clean()
    coarse = run_experiment_a(clean, "L1", delta=0.06, sigma=SIGMA)["oracle_auroc"]
    fine = run_experiment_a(clean, "L1", delta=SIGMA, sigma=SIGMA)["oracle_auroc"]
    assert coarse > fine


def test_leakage_gate_catches_reward_contamination():
    clean = _clean()
    clean_run = run_experiment_a(clean, "L1", delta=0.06, sigma=SIGMA)
    dirty_run = run_experiment_a(clean, "L1", delta=0.06, sigma=SIGMA, contaminate=0.05)
    assert clean_run["leakage_pass"] is True    # no confound
    assert dirty_run["leakage_pass"] is False   # reward offset is caught

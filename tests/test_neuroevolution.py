"""Experiment C neuroevolution primitives: deterministic mutation + selection.

Pure functions with no world/rollout dependency, so their determinism (the
milestone-1 bar) is testable without spinning up a PatchOfEarth.
"""

import numpy as np
import torch

from itasorl.agent_ac import RecurrentActorCritic
from itasorl.neuroevolution import mutate_policy, reproduce


def _fresh_agent() -> RecurrentActorCritic:
    torch.manual_seed(0)
    return RecurrentActorCritic(20, 5, embed=8, hidden=8, world_model=True)


def _tagged_pop(n: int) -> list[RecurrentActorCritic]:
    """n tiny agents, each stamped with a unique fingerprint on the critic bias
    so lineages can be tracked through reproduction (sigma=0 preserves the tag)."""
    pop = []
    for i in range(n):
        torch.manual_seed(100 + i)
        a = RecurrentActorCritic(6, 5, embed=4, hidden=4, world_model=False)
        with torch.no_grad():
            a.critic.bias.fill_(float(i))
        pop.append(a)
    return pop


def _tag(agent: RecurrentActorCritic) -> int:
    return int(round(agent.critic.bias.item()))


def _params_equal(a: RecurrentActorCritic, b: RecurrentActorCritic) -> bool:
    return all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))


def test_mutation_perturbs_weights():
    parent = _fresh_agent()
    child = mutate_policy(parent, sigma=0.02, rng=np.random.default_rng(1))
    # At least one weight tensor must differ from the parent under sigma>0.
    changed = any(
        not torch.equal(p, c)
        for p, c in zip(parent.parameters(), child.parameters())
    )
    assert changed, "sigma>0 must perturb at least one weight tensor"


def test_mutation_is_bit_reproducible():
    """Same parent + same keyed seed -> child identical to the bit (milestone-1 bar)."""
    parent = _fresh_agent()
    c1 = mutate_policy(parent, sigma=0.02, rng=np.random.default_rng(7))
    c2 = mutate_policy(parent, sigma=0.02, rng=np.random.default_rng(7))
    assert _params_equal(c1, c2)


def test_mutation_depends_on_seed():
    """Different keyed streams must produce different children (rng is consumed)."""
    parent = _fresh_agent()
    c1 = mutate_policy(parent, sigma=0.02, rng=np.random.default_rng(7))
    c2 = mutate_policy(parent, sigma=0.02, rng=np.random.default_rng(8))
    assert not _params_equal(c1, c2)


def test_sigma_zero_is_identity_copy():
    parent = _fresh_agent()
    child = mutate_policy(parent, sigma=0.0, rng=np.random.default_rng(1))
    assert _params_equal(parent, child)


def test_parent_is_not_mutated():
    parent = _fresh_agent()
    before = [p.clone() for p in parent.parameters()]
    _ = mutate_policy(parent, sigma=0.5, rng=np.random.default_rng(1))
    assert all(torch.equal(b, p) for b, p in zip(before, parent.parameters()))


def test_qualifier_injects_offspring():
    """A single above-threshold individual must place a mutated offspring into the
    next generation (differential reproduction is the engine of selection)."""
    pop = _tagged_pop(4)
    energies = np.array([0.0, 0.0, 9.0, 0.0])  # only index 2 qualifies
    nxt = reproduce(pop, energies, threshold=1.0, sigma=0.05,
                    rng=np.random.default_rng(0))
    # Some slot must now differ from the parent that originally sat there.
    moved = any(
        not all(torch.equal(p, q) for p, q in zip(pop[i].parameters(), nxt[i].parameters()))
        for i in range(len(pop))
    )
    assert moved, "a qualifier must inject a mutated offspring somewhere"


def test_capacity_is_conserved():
    pop = _tagged_pop(6)
    energies = np.array([5.0, 0.0, 5.0, 5.0, 0.0, 5.0])  # 4 qualifiers
    nxt = reproduce(pop, energies, threshold=1.0, sigma=0.02,
                    rng=np.random.default_rng(3))
    assert len(nxt) == len(pop)


def test_no_qualifiers_is_identity():
    pop = _tagged_pop(5)
    energies = np.zeros(5)  # nobody reaches threshold
    nxt = reproduce(pop, energies, threshold=1.0, sigma=0.02,
                    rng=np.random.default_rng(3))
    for a, b in zip(pop, nxt):
        assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))


def test_reproduction_is_bit_reproducible():
    pop = _tagged_pop(6)
    energies = np.array([5.0, 0.0, 5.0, 0.0, 5.0, 0.0])
    n1 = reproduce(pop, energies, threshold=1.0, sigma=0.02, rng=np.random.default_rng(11))
    n2 = reproduce(pop, energies, threshold=1.0, sigma=0.02, rng=np.random.default_rng(11))
    for a, b in zip(n1, n2):
        assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))


def test_only_qualifier_lineage_grows():
    """sigma=0 so tags are preserved: the sole qualifier's tag must gain a copy and
    some non-qualifier's tag must be displaced; no brand-new lineage may appear."""
    pop = _tagged_pop(5)  # tags 0..4
    energies = np.array([0.0, 0.0, 0.0, 9.0, 0.0])  # only tag 3 qualifies
    nxt = reproduce(pop, energies, threshold=1.0, sigma=0.0,
                    rng=np.random.default_rng(0))
    before = sorted(_tag(a) for a in pop)
    after = sorted(_tag(a) for a in nxt)
    assert before == [0, 1, 2, 3, 4]
    assert after.count(3) == 2, "qualifier lineage must gain exactly one copy"
    assert set(after).issubset(set(before)), "no non-existent lineage may appear"
    assert len(after) == 5


def test_selection_raises_mean_fitness_over_generations():
    """World-free gate-2 proxy: with a heritable fitness (fitness == tag) and the
    top half qualifying each generation, iterating reproduce must drive mean
    fitness up - the selection engine works before any world is attached."""
    rng = np.random.default_rng(0)
    pop = _tagged_pop(10)  # tags 0..9, heritable under sigma=0
    mean0 = float(np.mean([_tag(a) for a in pop]))
    for _ in range(40):
        tags = np.array([_tag(a) for a in pop], dtype=float)
        threshold = float(np.median(tags))
        pop = reproduce(pop, tags, threshold=threshold, sigma=0.0, rng=rng)
    mean_final = float(np.mean([_tag(a) for a in pop]))
    assert mean_final > mean0 + 1.0, (mean0, mean_final)


def _tag_fitness(pop):
    return np.array([_tag(a) for a in pop], dtype=float)


def test_evolve_records_rising_fitness_series():
    """The generational loop returns a per-generation fitness history whose mean
    climbs under selection (the gate-2 series, here on heritable tags)."""
    from itasorl.neuroevolution import evolve

    pop = _tagged_pop(10)  # tags 0..9
    final_pop, history = evolve(
        pop, _tag_fitness, generations=30,
        threshold=4.5, sigma=0.0, rng=np.random.default_rng(0),  # ABSOLUTE threshold
    )
    assert len(history) == 30
    assert len(final_pop) == 10
    assert history[-1]["mean_fitness"] > history[0]["mean_fitness"] + 1.0


def test_evolve_is_bit_reproducible():
    """Fixed seed -> identical fitness history AND identical final population."""
    from itasorl.neuroevolution import evolve

    def run():
        return evolve(_tagged_pop(10), _tag_fitness, generations=15,
                      threshold=4.5, sigma=0.01, rng=np.random.default_rng(42))

    p1, h1 = run()
    p2, h2 = run()
    assert [r["mean_fitness"] for r in h1] == [r["mean_fitness"] for r in h2]
    for a, b in zip(p1, p2):
        assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))


def test_evolve_calls_observe_once_per_generation():
    """The detection panel runs every generation on that generation's population,
    on a seam separate from fitness (sec. 5 survival-vs-detection separation)."""
    from itasorl.neuroevolution import evolve

    seen = []

    def observe(pop, g):
        seen.append((g, len(pop)))
        return {"auroc": 0.5}

    _, history = evolve(_tagged_pop(8), _tag_fitness, generations=5,
                        threshold=4.5, sigma=0.0, rng=np.random.default_rng(0),
                        observe=observe)
    assert seen == [(0, 8), (1, 8), (2, 8), (3, 8), (4, 8)]
    assert all(r["panel"] == {"auroc": 0.5} for r in history)

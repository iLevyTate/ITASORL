"""ITASORL - Experiment C neuroevolution primitives (pure, world-independent).

Experiment C switches the mechanism that could produce world-identity detection
from within-life gradient learning (A/B/L3) to Darwinian selection across
generations. These are the two heritable atoms it needs, kept free of any world
or rollout dependency so their determinism is testable in milliseconds:

  * ``mutate_policy`` - Gaussian perturbation of a policy's weights (the heritable
    material under Option P), drawn from a keyed PRNG stream so a child is
    reproducible to the bit from ``(parent, seed)``.

Design (docs/PREREGISTRATION_C.md sec. 3, 4): within-life gradient learning is
OFF, so a generational gain in detection is inherited, not learned.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

import numpy as np
import torch

from .agent_ac import RecurrentActorCritic

Population = list[RecurrentActorCritic]
FitnessFn = Callable[[Population], np.ndarray]
ObserveFn = Callable[[Population, int], dict[str, Any]]


def mutate_policy(
    parent: RecurrentActorCritic, sigma: float, rng: np.random.Generator
) -> RecurrentActorCritic:
    """Return a mutated COPY of ``parent`` (parent left untouched).

    Adds i.i.d. Gaussian noise of scale ``sigma`` to every weight tensor, drawn
    from ``rng`` (the keyed mutation stream) so a child is reproducible to the
    bit from ``(parent, seed)``. ``sigma == 0`` is an exact copy. Parameter
    iteration order is registration order (stable), which the determinism
    guarantee relies on.
    """
    child = copy.deepcopy(parent)
    with torch.no_grad():
        for p in child.parameters():
            noise = rng.standard_normal(size=tuple(p.shape)) * sigma
            p.add_(torch.as_tensor(noise, dtype=p.dtype))
    return child


def reproduce(
    population: list[RecurrentActorCritic],
    energies: np.ndarray,
    *,
    threshold: float,
    sigma: float,
    rng: np.random.Generator,
) -> list[RecurrentActorCritic]:
    """One generation of threshold-triggered reproduction at fixed carrying capacity.

    Semantics (docs/PREREGISTRATION_C.md sec. 4, fork 4): every individual whose
    lifetime ``energy >= threshold`` spawns ONE mutated offspring; each offspring
    displaces a uniformly-random OTHER slot, so the carrying capacity ``N =
    len(population)`` is conserved exactly. Selection is implicit and NOT
    truncation/fitness-proportionate: fitter lineages cross the threshold more
    often and so leave more copies. Generations OVERLAP - a parent persists unless
    it is the slot chosen for displacement (a reversible modelling choice; a
    non-overlapping full-turnover variant would fill every slot from qualifiers).

    Deterministic given ``rng`` (the keyed mutation+selection stream): qualifiers
    are processed in ascending index order and each draws its displacement slot
    then its weight perturbation from the same stream.
    """
    n = len(population)
    energies = np.asarray(energies)
    qualifiers = np.flatnonzero(energies >= threshold)
    nxt = list(population)
    for parent_idx in qualifiers:
        others = [j for j in range(n) if j != parent_idx]
        if not others:  # capacity 1: nowhere to place a child without erasing the parent
            continue
        slot = int(rng.choice(others))
        nxt[slot] = mutate_policy(population[parent_idx], sigma, rng)
    return nxt


def evolve(
    population: Population,
    evaluate_fitness: FitnessFn,
    *,
    generations: int,
    threshold: float,
    sigma: float,
    rng: np.random.Generator,
    observe: ObserveFn | None = None,
) -> tuple[Population, list[dict[str, Any]]]:
    """Run the Experiment C generational loop for ``generations`` generations.

    Each generation: score the CURRENT population's lifetime fitness via the
    injected ``evaluate_fitness`` (the mixed-world lifetime plugs in here), record
    a fitness summary (the gate-2 "does selection move fitness" series), optionally
    run the injected ``observe`` panel (the common-garden DETECTION measurement,
    kept on a separate seam so survival/fitness can never leak into it - sec. 5),
    then apply threshold-triggered reproduction.

    Deterministic given ``rng``. Returns ``(final_population, history)`` where
    ``history[g]`` is that generation's record.
    """
    pop = list(population)
    history: list[dict[str, Any]] = []
    for g in range(generations):
        energies = np.asarray(evaluate_fitness(pop), dtype=float)
        record: dict[str, Any] = {
            "gen": g,
            "mean_fitness": float(energies.mean()),
            "max_fitness": float(energies.max()),
            "n_qualifiers": int(np.count_nonzero(energies >= threshold)),
        }
        if observe is not None:
            record["panel"] = observe(pop, g)
        history.append(record)
        pop = reproduce(pop, energies, threshold=threshold, sigma=sigma, rng=rng)
    return pop, history

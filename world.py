"""
ITASORL - World protocol, observation/action specs, and the surrogate ladder.

Companion to docs/ITASORL_world_spec.md ("A Patch of Earth"). This module defines the
*contract* (the World protocol + byte-identical obs/action specs) and the
*surrogate ladder* (L0-L4) as composable wrappers, plus the matched-pair
shared-world rollout that is the keystone confound control (spec sec. 11).

Heavy simulation internals (the sec. 3-8 dynamics) and the learned/adversarial
surrogates (L3/L4) are left as documented stubs. The structural skeleton, the
spec-identity guarantee, L0/L1, and the matched-pair harness are complete.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Observation / action specification - the byte-identical contract (spec sec. 9)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChannelSpec:
    """One named, contiguous slice of the flat observation vector."""
    name: str
    size: int
    enabled: bool = True  # channel mask: disabled channels are zero-filled, NOT removed


@dataclass(frozen=True)
class ObsSpec:
    """Fixed-length, egocentric observation layout (spec sec. 9).

    Layout - names, sizes, order, dtype - must be identical between the authentic
    world and every surrogate. `with_mask` zero-fills disabled channels without
    changing length, so masking is invisible to the format contract.
    """
    channels: tuple[ChannelSpec, ...]
    dtype: str = "float32"

    @property
    def size(self) -> int:
        return sum(c.size for c in self.channels)

    def slices(self) -> dict[str, slice]:
        out: dict[str, slice] = {}
        i = 0
        for c in self.channels:
            out[c.name] = slice(i, i + c.size)
            i += c.size
        return out

    def empty(self) -> np.ndarray:
        return np.zeros(self.size, dtype=self.dtype)

    def assemble(self, parts: dict[str, np.ndarray]) -> np.ndarray:
        """Build the flat vector, zero-filling masked or absent channels."""
        v = self.empty()
        sl = self.slices()
        for c in self.channels:
            if c.enabled and c.name in parts:
                v[sl[c.name]] = np.asarray(parts[c.name], dtype=self.dtype).reshape(-1)
        return v

    def with_mask(self, enabled: dict[str, bool]) -> "ObsSpec":
        chans = tuple(replace(c, enabled=enabled.get(c.name, c.enabled)) for c in self.channels)
        return replace(self, channels=chans)

    def identity_hash(self) -> str:
        """Stable hash of the *format* (names/sizes/order/dtype) - NOT the mask.

        Authentic and surrogate must share this hash; the harness asserts it.
        """
        payload = json.dumps(
            {"dtype": self.dtype, "channels": [(c.name, c.size) for c in self.channels]},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class ActionSpec:
    """Continuous action layout (spec sec. 9): thrust+turn, eat, drink, emit-scent."""
    names: tuple[str, ...]
    low: tuple[float, ...]
    high: tuple[float, ...]

    @property
    def size(self) -> int:
        return len(self.names)

    def identity_hash(self) -> str:
        payload = json.dumps({"names": self.names, "low": self.low, "high": self.high}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


# Default specs for the maximally-rich configuration (spec sec. 9).
DEFAULT_OBS_SPEC = ObsSpec(
    channels=(
        ChannelSpec("vision", 24 * 5),  # 24 rays x (distance, 3 reflectance, radial velocity)
        ChannelSpec("smell", 4 * 3),    # 4 chemo-channels x (concentration, grad_x, grad_y)
        ChannelSpec("intero", 14),      # velocity(2) heading(2) E Hyd Tb wetness slope(2) temp light accel(2)
    ),
)

DEFAULT_ACTION_SPEC = ActionSpec(
    names=("thrust", "turn", "eat", "drink", "emit"),
    low=(0.0, -1.0, 0.0, 0.0, 0.0),
    high=(1.0, 1.0, 1.0, 1.0, 1.0),
)


# ---------------------------------------------------------------------------
# Determinism: explicitly-keyed RNG streams (spec sec. 12)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SeedBundle:
    world: int
    weather: int
    ecology: int
    policy: int = 0

    def streams(self) -> dict[str, np.random.Generator]:
        # Offsets keep streams independent and reproducible.
        return {
            "world": np.random.default_rng(self.world),
            "weather": np.random.default_rng(self.weather + 10_001),
            "ecology": np.random.default_rng(self.ecology + 20_002),
            "policy": np.random.default_rng(self.policy + 30_003),
        }


# ---------------------------------------------------------------------------
# Step result + opaque world-state snapshot (for matched-pair branching, sec. 11)
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)


# WorldState is intentionally opaque: a deep, EXACT snapshot of everything -
# fields, creatures, weather, and all RNG bit-generator states. Concrete worlds
# define its contents; get_state/set_state must round-trip it bit-for-bit.
WorldState = Any

Policy = Callable[[np.ndarray], np.ndarray]  # obs -> action (frozen at eval)


# ---------------------------------------------------------------------------
# The World protocol (the contract, spec sec. 1)
# ---------------------------------------------------------------------------

@runtime_checkable
class World(Protocol):
    obs_spec: ObsSpec
    action_spec: ActionSpec

    def reset(self, seeds: SeedBundle) -> StepResult: ...
    def step(self, action: np.ndarray) -> StepResult: ...
    def observe(self) -> np.ndarray: ...               # current obs without advancing
    def get_state(self) -> WorldState: ...             # snapshot (incl. RNG bit-states)
    def set_state(self, state: WorldState) -> None: ...  # exact restore


# ---------------------------------------------------------------------------
# Concrete authentic world (skeleton; dynamics stubs cross-ref spec sec. 3-9)
# ---------------------------------------------------------------------------

@dataclass
class WorldParams:
    L: float = 1.0        # arena side (toroidal)
    G: int = 128          # field grid resolution
    dt: float = 0.05
    sea_level: float = 0.35
    gravity: float = 1.0
    k_land: float = 0.20
    k_water: float = 0.60
    # diffusion D_k, decay lambda_k, regrowth rho_j/K_j, weather kappa/sigma,
    # metabolic + thermoregulation coefficients, mutation rate, ... (spec sec. 13)


class PatchOfEarth:
    """Authentic 2.5D world (spec sec. 2-9). float64 throughout; deterministic given seeds.

    The transition is decomposed into stages so surrogate wrappers and the L2
    drift model can intercept cleanly:
        _advance_time -> _integrate_motion -> _update_fields -> _resolve_ecology -> _observe
    The heavy bodies are stubs; the signatures and their ordering are the contract.
    """

    def __init__(
        self,
        params: WorldParams | None = None,
        obs_spec: ObsSpec = DEFAULT_OBS_SPEC,
        action_spec: ActionSpec = DEFAULT_ACTION_SPEC,
    ) -> None:
        self.params = params or WorldParams()
        self.obs_spec = obs_spec
        self.action_spec = action_spec
        self._rng: dict[str, np.random.Generator] = {}
        self._t: float = 0.0
        # L2 hook: a drift model may replace the exact integrator/field-update.
        self.dynamics_override: Callable[..., None] | None = None
        # static + dynamic fields and creature arrays are allocated in reset().

    # --- World protocol -----------------------------------------------------
    def reset(self, seeds: SeedBundle) -> StepResult:
        self._rng = seeds.streams()
        self._t = 0.0
        self._init_static_fields()   # H, sea level, wetness (spec sec. 3)
        self._init_dynamic_fields()  # R_j, C_k, T (spec sec. 4)
        self._init_creatures()       # focal agent + others (spec sec. 6)
        return StepResult(self.observe(), 0.0, False, False, {"t": self._t})

    def step(self, action: np.ndarray) -> StepResult:
        self._advance_time()
        self._integrate_motion(action)        # spec sec. 7
        self._update_fields()                 # spec sec. 4 (explicit-Euler PDE steps)
        info = self._resolve_ecology(action)  # spec sec. 8
        obs = self.observe()                  # spec sec. 9
        reward = self._homeostatic_reward()   # survival/homeostasis ONLY (never detection)
        terminated = self._focal_dead()
        return StepResult(obs, reward, terminated, False, {**info, "t": self._t})

    def observe(self) -> np.ndarray:
        return self._observe()

    def get_state(self) -> WorldState:
        raise NotImplementedError(
            "Return a deep, exact snapshot of fields + creatures + weather + every "
            "RNG bit-generator state (gen.bit_generator.state). Must round-trip with "
            "set_state bit-for-bit (spec sec. 11/12)."
        )

    def set_state(self, state: WorldState) -> None:
        raise NotImplementedError("Restore exactly what get_state captured, including RNG bit-states.")

    # --- transition stages (intervention points; bodies per spec) -----------
    def _advance_time(self) -> None:
        self._t += self.params.dt

    def _integrate_motion(self, action: np.ndarray) -> None:
        raise NotImplementedError("Newtonian + medium drag + slope gravity (spec sec. 7).")

    def _update_fields(self) -> None:
        raise NotImplementedError("Logistic regrowth, scent diffusion-advection-decay, temperature (spec sec. 4).")

    def _resolve_ecology(self, action: np.ndarray) -> dict[str, Any]:
        raise NotImplementedError("Foraging, drinking, predation, carcasses (spec sec. 8).")

    def _observe(self) -> np.ndarray:
        raise NotImplementedError("Raycasts + scent sampling + interoception -> obs_spec.assemble (spec sec. 9).")

    # --- helpers ------------------------------------------------------------
    def _init_static_fields(self) -> None:
        raise NotImplementedError

    def _init_dynamic_fields(self) -> None:
        raise NotImplementedError

    def _init_creatures(self) -> None:
        raise NotImplementedError

    def _homeostatic_reward(self) -> float:
        raise NotImplementedError

    def _focal_dead(self) -> bool:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Surrogate ladder (spec sec. 10) - composable wrappers preserving obs_spec
# ---------------------------------------------------------------------------

class Level(str, Enum):
    L0 = "L0_identical"
    L1 = "L1_discretization"
    L2 = "L2_rollout_drift"
    L3 = "L3_learned_model"
    L4 = "L4_adversarial"


class SurrogateWrapper:
    """Wraps a World and exposes the identical World protocol.

    Invariant: obs_spec / action_spec pass through UNCHANGED, so authentic and
    surrogate are format-identical (the harness asserts equal identity hashes).
    A subclass intervenes at exactly one documented point:
      - obs-level   -> override _obs_transform        (L1)
      - dynamics    -> install base.dynamics_override (L2/L3/L4)
    """

    level: Level = Level.L0

    def __init__(self, base: World) -> None:
        self.base = base
        self.obs_spec = base.obs_spec
        self.action_spec = base.action_spec

    def reset(self, seeds: SeedBundle) -> StepResult:
        return self._post(self.base.reset(seeds))

    def step(self, action: np.ndarray) -> StepResult:
        return self._post(self.base.step(action))

    def observe(self) -> np.ndarray:
        return self._obs_transform(self.base.observe())

    def get_state(self) -> WorldState:
        return self.base.get_state()

    def set_state(self, state: WorldState) -> None:
        self.base.set_state(state)

    def _post(self, r: StepResult) -> StepResult:
        r.obs = self._obs_transform(r.obs)
        return r

    def _obs_transform(self, obs: np.ndarray) -> np.ndarray:
        return obs  # default no-op


class L0Identity(SurrogateWrapper):
    """L0 - mathematically identical. No intervention; detection must be at chance.

    Exists so the harness path is byte-identical to the surrogate path: same
    wrapping, same code, zero behavioral change. The control that proves the
    method does not manufacture signal.
    """
    level = Level.L0


class L1Discretize(SurrogateWrapper):
    """L1 - discretization (spec sec. 10). Quantize the observation to a grid.

    Per-frame tell: obs values land on {k*delta}; inter-frame deltas cluster at
    multiples of delta; affected dims lose entropy. (A stronger *physical* version
    quantizes the world state via the dynamics_override; this obs-level form is the
    minimal, format-preserving intervention and is enough for the first result.)
    """
    level = Level.L1

    def __init__(self, base: World, delta: float = 1.0 / 64) -> None:
        super().__init__(base)
        self.delta = float(delta)

    def _obs_transform(self, obs: np.ndarray) -> np.ndarray:
        return np.round(obs / self.delta) * self.delta


class L2RolloutDrift(SurrogateWrapper):
    """L2 - rollout drift (spec sec. 10). Integrate with accumulating error
    (lower-order integrator, approximate diffusion solve, or a slowly drifting
    constant). Detection is invariant-based (energy / scent-mass / regrowth-
    equilibrium / momentum-decay) for chaos-robustness.

    Intervention point: install a drifting transition on the base via
    base.dynamics_override. Stub until that hook is implemented.
    """
    level = Level.L2

    def __init__(self, base: World, drift_sigma: float = 1e-3) -> None:
        super().__init__(base)
        self.drift_sigma = float(drift_sigma)
        raise NotImplementedError(
            "Install a drifting integrator/field-update via base.dynamics_override "
            "(spec sec. 10/11), then return through super().__init__."
        )


class L3LearnedModel(SurrogateWrapper):
    """L3 - a trained neural simulator of this world replaces the dynamics.
    Tell: smoothing/blur, mode collapse (rare events vanish), missing fine detail.
    """
    level = Level.L3

    def __init__(self, base: World, model: Any = None) -> None:
        super().__init__(base)
        self.model = model
        raise NotImplementedError("Delegate the base transition to a learned model via dynamics_override.")


class L4Adversarial(L3LearnedModel):
    """L4 - the L3 surrogate trained adversarially to match authentic statistics."""
    level = Level.L4


SurrogateFactory = Callable[[World], SurrogateWrapper]


# ---------------------------------------------------------------------------
# Matched-pair shared-world rollout - the keystone confound control (spec sec. 11)
# ---------------------------------------------------------------------------

@dataclass
class PairedEpisode:
    pair_id: int
    level: Level
    prefix_steps: int
    branch_steps: int
    authentic: list[StepResult]
    surrogate: list[StepResult]
    branch_order: tuple[str, str]  # randomized presentation order


def _continue(world: World, policy: Policy, n: int) -> list[StepResult]:
    out: list[StepResult] = []
    obs = world.observe()
    for _ in range(n):
        r = world.step(policy(obs))
        out.append(r)
        obs = r.obs
        if r.terminated:
            break
    return out


def matched_pair_rollout(
    make_world: Callable[[], World],   # () -> a fresh PatchOfEarth
    make_surrogate: SurrogateFactory,  # (World) -> L0..L4 wrapping a restored clone
    seeds: SeedBundle,
    policy: Policy,                    # frozen at eval
    prefix_steps: int,
    branch_steps: int,
    pair_id: int,
    rng: np.random.Generator,
) -> PairedEpisode:
    """Run a shared authentic prefix, snapshot, then branch into authentic and
    surrogate continuations from an IDENTICAL world state, in randomized order.

    L0 branches stay identical (the chance control); L1-L4 branches diverge ONLY
    because of the artifact. This neutralizes ecological, reward, and weather
    confounds at once - and, over a short branch or via invariants, chaos.
    """
    base = make_world()
    assert isinstance(base, World)

    # shared prefix
    r = base.reset(seeds)
    for _ in range(prefix_steps):
        if r.terminated:
            break
        r = base.step(policy(r.obs))
    snapshot = base.get_state()

    # authentic continuation from the snapshot (fresh world, exact restore)
    auth_world = make_world()
    auth_world.reset(seeds)          # allocate structures
    auth_world.set_state(snapshot)   # then overwrite with the exact snapshot
    authentic = _continue(auth_world, policy, branch_steps)

    # surrogate continuation from the SAME snapshot
    surr_base = make_world()
    surr_base.reset(seeds)
    surr_base.set_state(snapshot)
    surrogate_world = make_surrogate(surr_base)
    assert (
        surrogate_world.obs_spec.identity_hash() == auth_world.obs_spec.identity_hash()
    ), "obs_spec format must be identical between authentic and surrogate"
    surrogate = _continue(surrogate_world, policy, branch_steps)

    order: tuple[str, str] = ("authentic", "surrogate")
    if rng.random() < 0.5:
        order = ("surrogate", "authentic")

    return PairedEpisode(
        pair_id=pair_id,
        level=getattr(surrogate_world, "level", Level.L0),
        prefix_steps=prefix_steps,
        branch_steps=branch_steps,
        authentic=authentic,
        surrogate=surrogate,
        branch_order=order,
    )

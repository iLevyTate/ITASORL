"""
ITASORL - PatchOfEarth, recommended first configuration (world_spec sec. 14).

A runnable, deterministic instance of the authentic world for the lean first
experiment: single focal agent, one species, no reproduction; channels =
vision + interoception (smell masked, format preserved). Food is discrete
pellets - a lean stand-in for the sec. 4 biomass field - and the scent PDE
fields are omitted while smell is off. Everything is float64 and reproducible
to the bit, so L0 runs at chance, the matched-pair branch is exact, and the
Experiment A oracle can be built on the resulting logs.

Fills the PatchOfEarth stubs from world.py.
"""

from __future__ import annotations

import copy

import numpy as np

from .world import (
    DEFAULT_ACTION_SPEC,
    DEFAULT_OBS_SPEC,
    ObsSpec,
    PatchOfEarth,
    SeedBundle,
    WorldParams,
)

TWO_PI = 2.0 * np.pi

# Reflectance triples - perceptual signals the agent must learn, NOT labels.
REFL_FOOD = np.array([0.2, 0.8, 0.2], dtype=np.float32)
REFL_WATER = np.array([0.2, 0.4, 0.9], dtype=np.float32)
REFL_WALL = np.array([0.6, 0.5, 0.4], dtype=np.float32)
REFL_BG = np.array([0.0, 0.0, 0.0], dtype=np.float32)


def first_config_obs_spec() -> ObsSpec:
    """vision + interoception; smell masked (zero-filled, length preserved)."""
    return DEFAULT_OBS_SPEC.with_mask({"smell": False})


class PatchOfEarthV0(PatchOfEarth):
    def __init__(self, params: WorldParams | None = None, n_rays: int = 24, n_pellets: int = 12,
                 drift_sigma: float = 0.0, drift_mode: str = "ar1") -> None:
        super().__init__(params or WorldParams(), obs_spec=first_config_obs_spec(), action_spec=DEFAULT_ACTION_SPEC)
        # L2 rollout drift (world_spec sec. 10): if drift_sigma>0, the drag coefficient
        # deviates from the authentic constant. Two modes:
        #   "ar1"    - slow AR(1) random walk within each episode (the B-v2 surrogate; the
        #              worlds differ in drag VOLATILITY at a shared central level).
        #   "regime" - a per-episode CONSTANT drag offset drawn from a band away from zero
        #              (the B-v3 surrogate; identifiable from one trajectory and policy-
        #              relevant, so survival has a reason to encode it).
        # drift_sigma=0 reproduces the exact authentic world in BOTH modes.
        self.drift_sigma = float(drift_sigma)
        self.drift_mode = str(drift_mode)
        # regime band: per-episode offset ~ drift_sigma * U(regime_lo, regime_hi), so it is
        # centered on drift_sigma and bounded away from 0 (a clear, persistent regime shift).
        self.regime_lo, self.regime_hi = 0.5, 1.5
        self._drift_w = 0.0
        # L3 (learned-dynamics surrogate): when drift_mode=="l3" and _g_motion is set, the
        # velocity-update law is replaced by this learned map (vel, a, drag) -> vel_next.
        # The rest of the physics and the REAL observation model are unchanged, so obs stay
        # on the authentic manifold and the sole tell is G's dynamics error.
        self._g_motion = None
        # If set to a list, _integrate_motion appends (vel, a, drag, vel_next) motion
        # transitions - the AUTHENTIC data used to train the L3 surrogate. None = off.
        self._log_motion = None
        # --- tunable constants (world_spec sec. 13) ---
        self.n_rays = n_rays
        self.fov = np.deg2rad(300.0)
        self.ray_steps = 40
        self.max_range = 0.35
        self.n_pellets = n_pellets
        self.pellet_r = 0.02
        self.reach = 0.04
        self.thrust_scale = 0.6
        self.turn_rate = 3.0
        self.terr_basis = 4
        self.water_thresh = 0.5
        self.day_len = 200.0 * self.params.dt
        self.season_len = 20.0 * self.day_len
        # metabolism / homeostasis
        self.E0, self.Emax = 5.0, 10.0
        self.Hyd0, self.Hydmax = 5.0, 10.0
        self.basal_E, self.move_cost, self.thermo_cost = 0.02, 0.05, 0.02
        self.basal_Hyd, self.heat_Hyd, self.drink_rate = 0.01, 0.02, 0.5
        self.food_gain, self.age_max = 2.0, 1e9
        self.Tpref, self.Tmin, self.Tmax, self.Tmid = 0.5, 0.0, 1.0, 0.5
        self.mu_T, self.heat_met, self.lapse = 0.2, 0.05, 0.3
        self.T_base, self.T_amp = 0.45, 0.2
        self._reward = 0.0

    # --- terrain (static smooth field, analytic gradient) -------------------
    def _H_and_grad(self, x: float, y: float):
        arg = self._tf[:, 0] * x + self._tf[:, 1] * y + self._tph
        S = float(np.sum(self._ta * np.sin(arg)))
        th = np.tanh(S)
        sech2 = 1.0 - th * th
        gx = 0.5 * sech2 * float(np.sum(self._ta * self._tf[:, 0] * np.cos(arg)))
        gy = 0.5 * sech2 * float(np.sum(self._ta * self._tf[:, 1] * np.cos(arg)))
        return 0.5 + 0.5 * th, gx, gy

    def _H(self, x: float, y: float) -> float:
        arg = self._tf[:, 0] * x + self._tf[:, 1] * y + self._tph
        return 0.5 + 0.5 * float(np.tanh(np.sum(self._ta * np.sin(arg))))

    def _wetness(self, x: float, y: float) -> float:
        return 1.0 / (1.0 + np.exp(-(self.params.sea_level - self._H(x, y)) / 0.05))

    def _sun_light(self) -> float:
        A = 0.8 + 0.2 * np.sin(TWO_PI * self._t / self.season_len)
        return max(0.0, float(A * np.cos(TWO_PI * self._t / self.day_len)))  # t=0 -> midday

    def _ambient_T(self, x: float, y: float) -> float:
        return self.T_base + self.T_amp * self._sun_light() - self.lapse * self._H(x, y)

    # --- init stages --------------------------------------------------------
    def _init_static_fields(self) -> None:
        rng = self._rng["world"]
        self._tf = rng.uniform(1.5, 3.5, size=(self.terr_basis, 2)) * TWO_PI
        self._tph = rng.uniform(0.0, TWO_PI, size=self.terr_basis)
        self._ta = rng.uniform(0.5, 1.0, size=self.terr_basis) / self.terr_basis

    def _init_dynamic_fields(self) -> None:
        pass  # v0: no PDE fields while smell is off

    def _spawn_pellet(self) -> np.ndarray:
        rng = self._rng["ecology"]
        for _ in range(50):
            p = rng.uniform(0.0, 1.0, size=2)
            if self._wetness(p[0], p[1]) < self.water_thresh:  # on land
                return p
        return rng.uniform(0.0, 1.0, size=2)

    def _init_creatures(self) -> None:
        rng = self._rng["world"]
        self.pos = np.array([0.5, 0.5])
        for _ in range(50):
            if self._wetness(self.pos[0], self.pos[1]) < self.water_thresh:
                break
            self.pos = rng.uniform(0.0, 1.0, size=2)
        self.vel = np.zeros(2)
        self.heading = float(rng.uniform(0.0, TWO_PI))
        self.accel = np.zeros(2)
        self.E, self.Hyd, self.Tb, self.age, self.alive = self.E0, self.Hyd0, self.Tpref, 0.0, True
        self.pellets = np.stack([self._spawn_pellet() for _ in range(self.n_pellets)])
        self.pellet_amt = np.ones(self.n_pellets)
        self._reward = 0.0
        self._drift_w = 0.0
        if self.drift_sigma > 0.0:  # dedicated, deterministic drift stream (L2)
            self._rng["drift"] = np.random.default_rng(int(self._rng["world"].integers(0, 2**31)))
            if self.drift_mode == "regime":  # per-episode CONSTANT offset, fixed for the episode
                self._drift_w = float(np.clip(
                    self.drift_sigma * self._rng["drift"].uniform(self.regime_lo, self.regime_hi),
                    -0.8, 8.0))

    # --- transition stages --------------------------------------------------
    def _integrate_motion(self, action: np.ndarray) -> None:
        thrust = float(np.clip(action[0], 0.0, 1.0))
        turn = float(np.clip(action[1], -1.0, 1.0))
        self.heading = (self.heading + turn * self.turn_rate * self.params.dt) % TWO_PI
        d = np.array([np.cos(self.heading), np.sin(self.heading)])
        _, gx, gy = self._H_and_grad(self.pos[0], self.pos[1])
        a = thrust * self.thrust_scale * d - self.params.gravity * np.array([gx, gy])
        wet = self._wetness(self.pos[0], self.pos[1])
        drag = self.params.k_land * (1.0 - wet) + self.params.k_water * wet
        if self.drift_sigma > 0.0 and self.drift_mode in ("ar1", "regime"):  # L2: perturb drag
            if self.drift_mode == "ar1":  # slow AR(1) wander (regime: _drift_w stays constant)
                self._drift_w = float(np.clip(0.95 * self._drift_w + self._rng["drift"].normal(0.0, self.drift_sigma), -0.8, 8.0))
            drag = drag * (1.0 + self._drift_w)
        vel_before = self.vel
        if self.drift_mode == "l3" and self._g_motion is not None:  # L3: learned velocity law
            vel_next = np.asarray(self._g_motion(vel_before, a, drag), dtype=float)
        else:  # authentic (and L2 via the perturbed drag above): arithmetic unchanged
            vel_next = (1.0 - drag * self.params.dt) * vel_before + a * self.params.dt
        if self._log_motion is not None:  # capture AUTHENTIC transitions to train the L3 surrogate
            self._log_motion.append((np.array(vel_before, float), np.array(a, float),
                                     float(drag), np.array(vel_next, float)))
        self.vel = vel_next
        self.pos = self.pos + self.vel * self.params.dt
        for i in (0, 1):  # walls: clip and stop the normal velocity component
            if self.pos[i] < 0.0:
                self.pos[i], self.vel[i] = 0.0, 0.0
            elif self.pos[i] > 1.0:
                self.pos[i], self.vel[i] = 1.0, 0.0
        self.accel = a

    def _update_fields(self) -> None:
        pass  # v0: no PDE fields

    def _resolve_ecology(self, action: np.ndarray) -> dict:
        eat = float(np.clip(action[2], 0.0, 1.0))
        drink = float(np.clip(action[3], 0.0, 1.0))
        intake, ate = 0.0, False
        if eat > 0.5:
            d2 = np.sum((self.pellets - self.pos) ** 2, axis=1)
            j = int(np.argmin(d2))
            if d2[j] < self.reach ** 2 and self.pellet_amt[j] > 0:
                # Cap by remaining energy capacity too: at E == Emax nothing is
                # actually consumed, so no pellet depletion and no intake reward
                # (reward must stay a homeostatic quantity, not free at the cap).
                gain = min(self.pellet_amt[j], self.food_gain * eat * self.params.dt,
                           self.Emax - self.E)
                self.E = min(self.Emax, self.E + gain)
                self.pellet_amt[j] -= gain
                intake, ate = gain, True
                if self.pellet_amt[j] <= 1e-9:
                    self.pellets[j] = self._spawn_pellet()
                    self.pellet_amt[j] = 1.0
        T = self._ambient_T(self.pos[0], self.pos[1])
        if drink > 0.5 and self._wetness(self.pos[0], self.pos[1]) > self.water_thresh:
            self.Hyd = min(self.Hydmax, self.Hyd + self.drink_rate * drink * self.params.dt)
        amag = float(np.linalg.norm(self.accel))
        cost = self.basal_E + self.move_cost * amag + self.thermo_cost * abs(self.Tb - self.Tpref)
        self.E = max(0.0, self.E - cost * self.params.dt)
        self.Hyd = max(0.0, self.Hyd - (self.basal_Hyd + self.heat_Hyd * max(0.0, T - self.Tmid)) * self.params.dt)
        self.Tb = self.Tb + (self.mu_T * (T - self.Tb) + self.heat_met * amag) * self.params.dt
        self.age += self.params.dt
        self._reward = intake - cost * self.params.dt
        if self.alive and self._focal_dead():
            # Penalize the death TRANSITION once, not the dead state: a caller that
            # steps past termination must not collect -1 on every subsequent step.
            self._reward -= 1.0
            self.alive = False
        return {"ate": ate, "intake": intake}

    def _observe(self) -> np.ndarray:
        light = self._sun_light()
        mr = self.max_range * (0.3 + 0.7 * light)
        ds = mr / self.ray_steps
        vis = np.zeros((self.n_rays, 5), dtype=np.float32)
        start = self.heading - self.fov / 2.0
        for r in range(self.n_rays):
            ang = start + self.fov * (r / (self.n_rays - 1))
            d = np.array([np.cos(ang), np.sin(ang)])
            dist, refl = mr, REFL_BG
            for k in range(1, self.ray_steps + 1):
                t = k * ds
                p = self.pos + d * t
                if p[0] < 0.0 or p[0] > 1.0 or p[1] < 0.0 or p[1] > 1.0:
                    dist, refl = t, REFL_WALL
                    break
                d2 = np.sum((self.pellets - p) ** 2, axis=1)
                jm = int(np.argmin(d2))
                if d2[jm] < self.pellet_r ** 2 and self.pellet_amt[jm] > 0:
                    dist, refl = t, REFL_FOOD
                    break
                if self._wetness(p[0], p[1]) > self.water_thresh:
                    dist, refl = t, REFL_WATER
                    break
            radial = -float(self.vel @ d)
            vis[r] = [dist, refl[0], refl[1], refl[2], radial]
        _, gx, gy = self._H_and_grad(self.pos[0], self.pos[1])
        T = self._ambient_T(self.pos[0], self.pos[1])
        intero = np.array(
            [
                self.vel[0], self.vel[1], np.sin(self.heading), np.cos(self.heading),
                self.E / self.Emax, self.Hyd / self.Hydmax,
                (self.Tb - self.Tmin) / (self.Tmax - self.Tmin),
                self._wetness(self.pos[0], self.pos[1]), gx, gy, T, light,
                self.accel[0], self.accel[1],
            ],
            dtype=np.float32,
        )
        return self.obs_spec.assemble({"vision": vis.reshape(-1), "intero": intero})

    def _homeostatic_reward(self) -> float:
        return float(self._reward)

    def _focal_dead(self) -> bool:
        return (self.E <= 0.0) or (self.Hyd <= 0.0) or (self.Tb < self.Tmin) or (self.Tb > self.Tmax) or (self.age > self.age_max)

    # --- snapshot / restore (exact, incl. RNG bit-states; spec sec. 11/12) --
    def get_state(self):
        return copy.deepcopy(
            {
                "t": self._t, "pos": self.pos, "vel": self.vel, "heading": self.heading, "accel": self.accel,
                "E": self.E, "Hyd": self.Hyd, "Tb": self.Tb, "age": self.age, "alive": self.alive,
                "pellets": self.pellets, "pellet_amt": self.pellet_amt,
                "tf": self._tf, "tph": self._tph, "ta": self._ta, "reward": self._reward,
                "drift_w": self._drift_w,
                "rng": {k: v.bit_generator.state for k, v in self._rng.items()},
            }
        )

    def set_state(self, s) -> None:
        s = copy.deepcopy(s)
        self._t, self.pos, self.vel, self.heading, self.accel = s["t"], s["pos"], s["vel"], s["heading"], s["accel"]
        self.E, self.Hyd, self.Tb, self.age, self.alive = s["E"], s["Hyd"], s["Tb"], s["age"], s["alive"]
        self.pellets, self.pellet_amt = s["pellets"], s["pellet_amt"]
        self._tf, self._tph, self._ta, self._reward = s["tf"], s["tph"], s["ta"], s["reward"]
        self._drift_w = s.get("drift_w", 0.0)
        # Restore only streams this world actually has: a snapshot from a differently
        # configured world (e.g. a drift world's "drift" stream restored into an
        # authentic world) must not KeyError. Streams the snapshot lacks keep their
        # current state - same-config round-trips remain exact.
        for k, st in s["rng"].items():
            if k in self._rng:
                self._rng[k].bit_generator.state = st


def make_v0_world(_seed_offset: int = 0) -> PatchOfEarthV0:
    return PatchOfEarthV0()


if __name__ == "__main__":
    w = PatchOfEarthV0()
    r = w.reset(SeedBundle(world=1, weather=2, ecology=3))
    for _ in range(50):
        r = w.step(np.array([0.5, 0.1, 1.0, 0.0, 0.0], dtype=np.float32))
    print("ran 50 steps; obs dim", r.obs.shape[0], "alive", w.alive, "E", round(w.E, 3))

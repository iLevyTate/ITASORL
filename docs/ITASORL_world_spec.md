# ITASORL World Specification v0 — "A Patch of Earth"

*Companion to the ITASORL design document. This is the concrete base world (step 1 of the Critical Path) for the maximally-rich, Earth-like configuration. The world is authored and exactly computable so it can serve as ground truth; Earth-likeness lives in its physics, ecology, and senses — never in semantics. The visualization may be fully 3D and photoreal regardless; this spec governs only the simulation and the agent's observations.*

## 1. Non-negotiable constraints

Every choice below is bound by five requirements the experiment cannot survive without:

- **Authored & exactly computable.** The authentic world is defined entirely by code and float64 arithmetic, so it *is* the ground truth and an L0 surrogate can be made bit-identical. No real-world data, no uncomputable physics.
- **Deterministic.** Given seeds, a run reproduces to the bit. This is what makes L0 ≈ chance provable (see §12).
- **Dissipative (chaos-tamed).** Movement has drag, ecology has carrying capacity, weather is mean-reverting, fields diffuse and decay. Bounded, stable dynamics keep divergence attributable to artifacts rather than to natural chaos (see §10, §11).
- **Non-semantic.** Observations are physical readings only — no language, labels, or human concepts. Genomes encode physiology, not meaning. Preserves the tabula-rasa invariant.
- **Byte-identical, channel-maskable `obs_spec`.** Authentic and surrogate emit observations of identical format, dtype, and length. The vector is partitioned into channels (vision / smell / interoception) that can each be masked, so the *world* can be maximally rich while a given *experiment* exposes only a clean subset (see §9, §11).

## 2. World representation

- **2.5D.** A continuous 2D plane (x, y) ∈ [0, L)² (toroidal) plus a static terrain-height field H(x, y) and medium layers — giving slopes, water bodies, and land/water/air media without the cost and chaos of full 3D. (3D is a later extension; the awe-layer renderer can already be 3D.)
- **Time** advances in fixed steps dt (float64).
- **Fields** live on a G×G grid, bilinearly interpolated to continuous positions.
- All state in float64.

## 3. Static fields (fixed per world instance, computed at init)

- **Terrain height** H(x, y): a fixed smooth field (a few low-frequency value-noise octaves), normalized to [0, 1].
- **Sea level** s: water exists where H < s.
- **Medium / wetness** w(x, y) = sigmoid((s − H) / ε) ∈ [0, 1] (1 = open water, 0 = dry land). Used to blend drag and buoyancy *continuously*, avoiding hard edges that could leak as a tell.

## 4. Dynamic fields (updated each step; all dissipative)

Each is an explicit-Euler PDE step on the grid:

- **Resource biomass** R_j(x, y, t), j over resource types (plant, fruit, …): logistic regrowth minus foraging.
  ∂R_j/∂t = ρ_j · R_j · (1 − R_j / K_j) · growth_light(L) − forage_j
- **Scent / chemical fields** C_k(x, y, t), k over channels (food-odor, conspecific, predator, decay): diffusion–advection–decay with sources.
  ∂C_k/∂t = D_k ∇²C_k − ∇·(W · C_k) − λ_k C_k + S_k
  (W = wind from weather; sources S_k from food, creatures, carcasses.) Diffusion + decay ⇒ stable.
- **Temperature** T(x, y, t): relaxes toward a target set by sun elevation, altitude lapse (−Γ·H), and medium thermal inertia, plus a weather offset.
  ∂T/∂t = α_med(x, y) · (T_target(x, y, t) − T)

## 5. Environmental forcing

- **Sun (deterministic).** Day/night and seasons as pure trig: sun_elev(t) = sin(2π t / day) modulated by a seasonal amplitude/daylength term. Drives light L (vision range and SNR), the temperature target, and plant growth.
- **Weather (bounded, mean-reverting; stochastic but stationary).** An Ornstein–Uhlenbeck process on a small weather state ω: dω = −κ(ω − ω̄)dt + σ dξ, clipped to a box. From ω derive wind W(t), cloud cover (scales L), and precipitation P(t) (raises hydration/surface water, shifts T). Slow κ ⇒ realistic but non-chaotic; identical between authentic and L0.

## 6. Organism state & genome

Per creature i:

- **Kinematic:** position (x, y), heading φ, velocity (vₓ, v_y).
- **Physiological (homeostasis):** energy E, hydration Hyd, body temperature Tb, age, alive flag.
- **Genome gᵢ (physiology, not semantics):** body size, max thrust, basal metabolic rate, sensory acuity (vision range, scent sensitivity), diet type (which R_j / prey it may consume), thermal preference Tpref. Multi-species = a few genome archetypes (grazer, predator, scavenger). Genomes mutate for Experiment C; fixed for A/B.

## 7. Dynamics (per step dt)

**Movement** (Newtonian, medium-dependent drag, terrain gravity):

- k_drag = lerp(k_land, k_water, w(x, y)); buoyancy/lift from medium.
- slope acceleration g_slope = −g · ∇H(x, y).
- a = R(φ)·thrust(action) + g_slope.
- v ← (1 − k_drag·dt)·v + a·dt;  pos ← wrap(pos + v·dt);  φ ← φ + turn·dt.

**Metabolism / homeostasis:**

- E ← E − (b_basal + b_move·‖a‖ + b_thermo·|Tb − Tpref|)·dt + intake_E
- Hyd ← Hyd − (h_basal + h_heat·max(0, T − Tmid))·dt + intake_water
- Tb ← Tb + (μ·(T − Tb) + heat_met·‖a‖ − evap·P)·dt
- **Death** if E ≤ 0, Hyd ≤ 0, Tb ∉ [Tmin, Tmax], predation, or age > age_max.

## 8. Ecology

- **Foraging:** the eat action draws from R_j at the agent's cell (diet-gated) → intake_E; depletes the field.
- **Drinking:** the drink action in water / under precipitation → intake_water.
- **Predation:** the bite action against a creature within reach (diet-gated) transfers energy; prey dies. Predators emit a predator-scent source.
- **Carcasses & decay:** dead creatures become a decaying biomass + decay-scent source (scavenger food).
- **Reproduction (Experiment C only):** above an energy threshold, spawn offspring with a mutated genome at an energy cost. Disabled for A/B.

## 9. Sensory model = the `obs_spec` (egocentric, fixed-length)

What the **agent** receives each step — Earth-like senses, physical readings only, partitioned into maskable channels.

| Channel | Contents | Dim |
|---|---|---|
| **Vision** (1D retina, N_ray = 24 rays over a 300° FOV) | per ray: hit distance, 3 material/reflectance values (continuous, *not labels*), radial relative velocity | 24 × 5 = 120 |
| **Smell** (K = 4 chemo-channels) | per channel: local concentration + 2D local gradient (chemotaxis) | 4 × 3 = 12 |
| **Interoception** | velocity (2), heading sin/cos (2), energy, hydration, body temp, local wetness, terrain-slope vector (2), ambient temp, light level, acceleration (2) | ~14 |

- **Total ≈ 146-dim** float vector — maximally rich but fully enumerable. Vision range/SNR scales with light L; scent SNR with wind and decay. **Material/reflectance channels are perceptual, never labels** — the agent must *learn* what "edible-looking" reflectance means.
- **`action_spec`** (continuous, fixed): thrust forward + turn (2), bite/eat effort (1), drink (1), emit-scent (1) ≈ 5-dim.
- **Channel mask** (config flag): any channel group may be zeroed/omitted *identically across authentic and surrogate*, so an experiment can expose vision-only, smell-only, etc., without changing the world.

## 10. How the surrogate ladder attaches to this world

- **L0 — identical.** Same code, same seeds → bit-identical logs. Detection must be at chance. (The richness raises the determinism bar; see §12.)
- **L1 — discretization.** Snap to a grid: quantize positions/velocities to Δ_space / Δ_v, and/or quantize field values, and/or coarsen the field grid G, and/or coarsen dt. The raycasts and scent samples inherit the lattice. *Tell:* grid-comb structure / reduced entropy in obs dimensions; per-frame.
- **L2 — rollout drift.** Integrate with accumulating error: a lower-order field/dynamics integrator, an approximate diffusion solve, or slow drift in a physical constant (drag, D_k, g). *Tell (primary, chaos-robust):* violation of conserved / quasi-conserved budgets — energy bookkeeping, scent-field mass conservation, logistic-regrowth equilibrium, momentum decay rate. *Secondary:* short-horizon k-step rollout error vs. the known authentic dynamics. Integrative — needs a window.
- **L3 — learned world model.** Replace the field/dynamics updates with a trained neural simulator of this world. *Tell:* smoothing/blur, mode collapse (rare weather/events vanish), missing fine detail in terrain/scent.
- **L4 — adversarial.** Train the L3 surrogate to minimize the divergence between its observable statistics and the authentic world's.

## 11. Managing the confound surface (the cost of "maximally rich")

Richness multiplies the channels through which an agent could distinguish worlds *for the wrong reason*. The controls:

1. **Matched-pair, shared-world rollout (keystone control).** For each evaluation unit, simulate a shared authentic prefix, then branch into authentic and surrogate continuations from an *identical* full world state (terrain, fields, all creatures, all RNG), in randomized order. L0 branches stay identical; L1–L4 branches diverge *only* because of the artifact. This neutralizes ecological, reward, and weather confounds at once — and, measured over a short branch or via invariants, neutralizes chaos.
2. **Leakage-audit battery at every level.** Simple baselines trained on metadata, reward sequences, episode length, reset counts, timestep indices, wall-clock, action masks, and raw obs-bytes must all *fail* to separate worlds. Pass/fail gate. The richer the obs, the more this matters.
3. **Ecological isolation for the first result.** Single focal agent, with other creatures frozen/scripted (or shared identically via the matched pair), one species, reproduction off — so a detection cannot be socially imported.
4. **Channel-masking staging.** Run the first clean measurement with a minimal channel set (e.g. vision + interoception, smell off, fewer rays), then scale channels up. The *world* stays maximally rich; the *experiment* controls exposure.
5. **Invariant-first L2.** Prefer conserved-quantity violations over pointwise prediction for drift, so the signal is robust to residual chaos.
6. **Chaos budget.** Keep weather mean-reversion κ, ecological carrying capacities, and drag in the dissipative regime; verify empirically that two L0 branches stay within float tolerance over the evaluation horizon — a determinism test that doubles as a chaos test.

## 12. Determinism & RNG plan

Separate, explicitly-keyed PRNG streams (per the design doc's discipline): world-init (terrain, initial fields/positions), weather, ecology / stochastic events, agent policy (if stochastic), and probe. A run is `(world_seed, weather_seed, ecology_seed, …)`. **L0 = the same world run twice under the same seeds → identical logs.** Field updates and event resolution must consume RNG in a fixed, order-stable way — the main determinism hazard the richness introduces.

## 13. Parameterized knobs (tune later; fix before preregistration)

Grid resolution G, ray count N_ray, scent channels K, dt, drag/diffusion/decay constants, carrying capacities, weather κ/σ, metabolic and thermoregulation coefficients, mutation rates, and the ladder strengths (Δ for L1; integrator order or constant-drift σ for L2). The smallest-effect-size-of-interest for the L0 equivalence test is set against the chosen metric once these are fixed.

## 14. Recommended first configuration

To get a clean first signal fast (Experiment A → B at L1, then L2):

- Single focal agent, one species, reproduction off.
- Channels: vision + interoception (smell off); N_ray reduced (e.g. 12); short fixed-horizon episodes.
- Matched-pair shared-world rollout; randomized branch order.
- L0 first (verify chance + determinism), then L1 (per-frame, easiest), then L2 (invariant-based).
- Experiment A oracle computed on the *known* dynamics — no trained agent — to set the detectability ceiling and run the leakage audit.
- Only after this scales: add smell, more rays, multi-species, weather extremes, reproduction (Experiment C), and L3/L4.

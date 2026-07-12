# ItaSoRL

*A tabula-rasa artificial-life system that asks whether a from-scratch digital
organism can tell that its world is a generative **surrogate** rather than the
authentic base dynamics of its computational substrate, using only substrate
seams, with detection **read out, not rewarded**.*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iLevyTate/ITASORL/blob/main/notebooks/colab_gpu.ipynb)

**Google Colab (GPU):** [Open `notebooks/colab_gpu.ipynb` in Colab](https://colab.research.google.com/github/iLevyTate/ITASORL/blob/main/notebooks/colab_gpu.ipynb). Enable a GPU runtime, then run all cells (clones the repo and runs `python scripts/run_e2e.py`).

---

## The theme in one line

Drop a learning creature, built from nothing, into a world that is secretly either
the real thing or a flawed copy of itself. Can the difference be noticed from the
seams alone? And does the creature come to represent that difference *on its own*,
without ever being paid to?

---

## In plain language

*(Read this part if you want the idea without the jargon.)*

Imagine raising a digital creature inside a simulated world. The creature starts
with no built-in knowledge, no labels, no instructions. It just lives there,
senses its surroundings, and learns to predict what happens next.

Now play a trick on it. Sometimes the world it lives in runs on the "true" physics
of the computer underneath. Other times we swap in a slightly *fake* version: a
copy that behaves almost the same, but carries tiny tell-tale flaws, the kind of
seams you would only get from a stand-in rather than the real thing. Think of it
as a very subtle "is this the Matrix?" test for a tiny mind.

We are chasing two separate questions, and keeping them strictly apart matters:

1. **Can the fakeness even be detected?** If an all-seeing outside observer
   inspects the world's behavior, can it spot the copy? (Short answer so far: yes,
   almost perfectly.)
2. **Does the creature notice on its own?** Here is the twist that makes this
   research, not a gadget: we never reward the creature for spotting the fake. We
   never even tell it the fake exists. We only ever *read out* whether the
   knowledge ended up inside its head as a side effect of ordinary learning.

**What we found is the interesting part.** The fakeness is trivially detectable
from the outside, but the creature does not bother to learn it. An outside
observer catches the copy nearly every time; the creature's own internal state
barely registers it at all. Nothing in its day-to-day task *required* it to care
about the difference, so it simply did not represent it.

That gap, between what is *detectable* and what a mind *spontaneously encodes*, is
the real object of study. It turns a vague headline ("can an organism notice its
world is fake?") into a sharper, testable question: *under what conditions does a
mind start to represent something it was never asked to care about?* The next phase
(making survival depend on the difference) is built to answer exactly that.

---

## For researchers

Take an agent with no imported semantics, drop it into a controllable world, and
make that world either an authentic simulation or a subtly-flawed surrogate of
itself. Can the distinction be detected from the substrate's seams alone? And, the
part that matters, does an agent encode that distinction *incidentally*, without
ever being rewarded for it?

The project rests on three invariants:

- **Tabula rasa**: no semantics are imported; the agent starts from nothing.
- **Readout, not reward**: world identity is *probed*, never trained or rewarded.
- **Substrate-grounded**: the only tells are artifacts of the computational substrate.

Detection difficulty is dialed with a **surrogate ladder** (L0 identical control,
L1 discretization, L2 rollout drift, L3 learned-model fingerprint, L4 adversarial).

### Hypotheses

- **H1 (detectability)**: the surrogate is distinguishable from the authentic world by a privileged discriminator.
- **H2 (substrate-grounding)**: the signal lives in substrate artifacts, confirmed via ablations.
- **H3 (emergence under selection)**: representing the distinction emerges when survival depends on it.
- **H4 (legibility / incidental encoding)**: a from-scratch agent encodes the distinction incidentally, without reward.

### Current status

| Component | State |
|-----------|-------|
| World ("A Patch of Earth" v0) | built, verified deterministic, snapshot-exact |
| Ladder L0 / L1 / L2 | implemented and validated |
| Ladder L3 (learned-dynamics surrogate) | implemented + oracle-gated |
| Experiment A (detectability ceiling, agent-free), L1 | **done** |
| Experiment A, L2 | **done** |
| Experiment B (incidental detection), L2 arc | **done (robust negative result)** |
| Experiment B, L3 (learned-dynamics) | **positive at n=10 (reward/survivorship-controlled; partly behavior-mediated)** |
| Experiment C (emergence under selection) / Ladder L4 | not started |

### Key result

*For a hand-tuned dynamics artifact (L2), detectability does not imply incidental
encoding.* An artifact an external oracle detects at AUROC ≈ 0.99 leaves essentially no
decodable trace in a from-scratch agent's representation (≈ 0.5), across three readout
channels and surviving objective-engagement, nonlinear-probe, scale (n=10), and
capacity-ceiling checks.

*But the picture changes at L3.* When the surrogate is a **learned-dynamics fingerprint**
(a small net replacing the velocity law) rather than a hand-tuned knob, the **survival**
agent, and only the survival agent, encodes it. At a difficulty where an untrained net is at
chance (about 0.49, and still only 0.52 under a nonlinear probe) and a prediction-only agent
is near chance (about 0.57), the survival probe reads **0.752** (n = 10, honest 90% CI
**[0.698, 0.807]**, which excludes the pre-registered 0.65 bar; 8 of 10 seeds clear it). The
dissociation is robust: it is not reward-mediated (world is not decodable from summed reward,
AUROC 0.541, clean 10 of 10 seeds), not survivorship-biased (0 early deaths, every pool
110/110), not a linear-probe artifact (the untrained net stays near chance even nonlinearly),
and the L0 authentic-vs-authentic control is at chance (0.517). **Caveat, from a post-hoc
audit:** the signal is partly *behavior-mediated*. The agent moves and forages differently in
the two worlds, so behavior alone (speed, energy, food, drag) already decodes the world at about
0.69; controlling for behavior cleanly (behavior model fit in-fold), the state's
behavior-independent world-signal is about **0.66** (0.676 linear, 0.659 quadratic), still well
above the untrained floor (0.488) but only *at* the 0.65 bar (6 of 10 seeds clear it). So
behavior mediates roughly 0.09 of the 0.752 headline; a real behavior-independent component
survives, but it is weak-to-moderate, not the abstract world-identity direction the raw 0.752
suggests (and this is still a soft upper bound: only per-episode mean behavior was controlled,
so per-timestep behavior could lower it further). The honest statement is: reward- and
survivorship-controlled, robust to nonlinear probing, with a modest (~0.66) behavior-independent
world-signal. This is still the first place
"detectable does not imply learned" reverses: a from-scratch agent, never rewarded for it,
comes to carry world-discriminative state as a byproduct of surviving. Remaining work: a
richer per-timestep behavior control and a second in-band capacity (hidden = 4). See
[`docs/FINDINGS.md`](docs/FINDINGS.md) and [`docs/PREREGISTRATION_L3.md`](docs/PREREGISTRATION_L3.md).

---

## Repository layout

```
.
|-- README.md                   this file: the map
|-- LICENSE
|-- requirements.txt            runtime dependencies
|-- requirements-dev.txt        pytest + dev tooling
|-- itasorl/                    core library (world, agents, experiments)
|   |-- world.py                World protocol, surrogate ladder, matched-pair harness
|   |-- patch_of_earth.py       PatchOfEarthV0 concrete world, incl. L1/L2 hooks
|   |-- agent.py                recurrent world model (RSSM-lite)
|   |-- agent_ac.py             survival actor-critic (Experiment B-v2)
|   |-- experiment_a.py         agent-free L1 detectability oracle
|   |-- experiment_b.py         incidental-detection harness
|   |-- experiment_b2.py        survival-coupled B-v2 pipeline
|   `-- results_io.py           end-to-end run recording
|-- scripts/                    deterministic reproduction runners
|   |-- run_e2e.py              pytest + all experiments (recorded)
|   |-- run_expA.py ...         Experiment A/B runners
|   `-- run_expB2.py            Experiment B-v2 (GPU if available)
|-- docs/
|   |-- ITASORL.md              research plan
|   |-- FINDINGS.md             empirical results
|   |-- PREREGISTRATION.md      B-v2 pre-registration
|   `-- figures/                result figures (.png)
|-- artifacts/expB2/            published B-v2 JSON/PNG (committed)
|-- fullruns/                   e2e run bundles (gitignored; default output)
|-- results/LATEST_RUN.txt      pointer to latest fullruns folder
|-- notebooks/colab_gpu.ipynb   Colab end-to-end runner
|-- tests/                      pytest regression suite
`-- ralph/                      autonomous research + bug-fix loop (Claude Code)
```

### Documents

- [`docs/ITASORL.md`](docs/ITASORL.md): the research plan, core question, literature white-space, hypotheses (H1 to H4), experiments (A/B/C), the surrogate ladder, validity audit, statistics, and engineering architecture.
- [`docs/ITASORL_world_spec.md`](docs/ITASORL_world_spec.md): the world specification, "A Patch of Earth" v0, the 2.5D representation, fields and forcing, dynamics, ecology, the ~146-dim observation, ladder attachment, and confound management.
- [`docs/FINDINGS.md`](docs/FINDINGS.md): empirical results from the first build-and-test cycle.

### Figures

Result figures live in [`docs/figures/`](docs/figures/) and are regenerated by the
run scripts:

- `expA_ceiling.png`: L1 detectability ceiling vs grid spacing.
- `expA_L2_ceiling.png`: L2 detectability ceiling vs drift strength.
- `expB_incidental.png`: recurrent-state probe across the drift sweep (target vs negative control).
- `expB_channels.png`: two incidental-detection channels (recurrent state vs prediction error).
- `expB_kstep.png`: effect of a longer-horizon objective on encoding.

---

## How to run

Dependencies:

```bash
pip install numpy scikit-learn matplotlib torch
```

Reproduce the experiments (each is deterministic given its seeds; run from the repo
root so figures land in `docs/figures/`):

```bash
python scripts/run_e2e.py --quick   # full battery + recorded results (recommended)
python scripts/run_expA.py          # Experiment A, L1
python scripts/run_expA_l2.py       # Experiment A, L2
python scripts/run_expB_full.py     # Experiment B: recurrent-state probe
python scripts/run_expB_surprise.py # Experiment B: prediction-error channel
python scripts/run_expB_kstep.py    # Experiment B: open-loop horizons
python scripts/run_expB_gap.py      # Experiment B: engagement + delta objective
python scripts/run_expB_nonlinear.py# Experiment B: nonlinear probe check
python scripts/run_expB2.py         # Experiment B-v2: survival-coupled
```

A quick smoke test of the Experiment B pipeline:

```bash
python -m itasorl.experiment_b
```

Run the test suite:

```bash
pytest -q
```

**Google Colab (GPU):** [Open in Colab](https://colab.research.google.com/github/iLevyTate/ITASORL/blob/main/notebooks/colab_gpu.ipynb) (same notebook as the badge at the top). Enable a GPU runtime, run all cells.

**Local Jupyter / VS Code:** open [`notebooks/colab_gpu.ipynb`](notebooks/colab_gpu.ipynb) from this repo; it auto-detects local mode (no Drive/download cells).

---

## What to read first

1. `README.md` (this file): the map.
2. [`docs/FINDINGS.md`](docs/FINDINGS.md): what we found and what it means.
3. [`docs/ITASORL.md`](docs/ITASORL.md): the full research plan and rationale.
4. [`docs/ITASORL_world_spec.md`](docs/ITASORL_world_spec.md): the world, in detail.

---

## The one-paragraph takeaway

The hard part of "can an organism notice its world is fake?" turned out not to be
detectability: substrate artifacts are readily detectable by a privileged observer,
with calibrated difficulty and a clean confound audit. The hard part is
**incidental encoding**. A from-scratch agent trained only to predict its senses
does not represent a dynamics artifact it is never asked to care about, even when
that artifact is strongly present and trivially detectable from the outside. That
gap, between what is detectable and what an agent spontaneously encodes, is the real
object of study, and it turns the headline hypothesis into a sharper, conditional
question that the next phase (survival-coupled reward, richer ladder rungs) is
designed to answer.

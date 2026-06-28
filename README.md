# ITASORL

*A tabula-rasa artificial-life system that asks whether a from-scratch digital
organism can tell that its world is a generative **surrogate** rather than the
authentic base dynamics of its computational substrate, using only substrate
seams, with detection **read out, not rewarded**.*

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
| Experiment A (detectability ceiling, agent-free), L1 | **done** |
| Experiment A, L2 | **done** |
| Experiment B (incidental detection), first full arc | **done (robust negative result)** |
| Experiment C (emergence under selection) | not started |
| Ladder L3 / L4 | not started |

### Key result

*Detectability does not imply incidental encoding.* An artifact an external oracle
detects at AUROC ≈ 0.99 leaves essentially no decodable trace in a from-scratch
agent's representation (≈ 0.5), across three readout channels and surviving
objective-engagement and nonlinear-probe checks. See
[`docs/FINDINGS.md`](docs/FINDINGS.md) for the full writeup and figures.

---

## Repository layout

```
.
|-- README.md                   this file: the map
|-- LICENSE
|-- docs/
|   |-- ITASORL.md              research plan: question, hypotheses, experiments, ladder, statistics
|   |-- ITASORL_world_spec.md   world specification: "A Patch of Earth" v0
|   |-- FINDINGS.md             empirical results (where the numbers live)
|   `-- figures/                result figures (.png)
|-- world.py                    World protocol, ObsSpec, surrogate-ladder wrappers, matched-pair harness
|-- patch_of_earth.py           PatchOfEarthV0 concrete world, incl. L1/L2 hooks
|-- logschema.py                step/activation logging schema + writer
|-- agent.py                    recurrent world model (RSSM-lite, PyTorch) + numpy reservoir fallback
|-- experiment_a.py             agent-free L1 detectability oracle + leakage-audit battery
|-- experiment_a_l2.py          L2 detectability oracle (rollout-residual)
|-- experiment_b.py             incidental-detection harness: collect, train, probe
|-- run_expA.py ... run_expB_*.py   deterministic reproduction scripts
|-- tests/                      pytest regression suite
`-- ralph/                      autonomous-loop tooling + journal
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
python run_expA.py          # Experiment A, L1: control / detection / contamination / calibration
python run_expA_l2.py       # Experiment A, L2: same, for rollout drift
python run_expB_full.py     # Experiment B: recurrent-state probe, drift sweep + control + selectivity
python run_expB_surprise.py # Experiment B: prediction-error (surprise) channel
python run_expB_kstep.py    # Experiment B: longer-horizon (open-loop) objective
python run_expB_gap.py      # Experiment B: objective-engagement diagnostic + delta-rollout objective
python run_expB_nonlinear.py# Experiment B: nonlinear (random-forest) probe robustness check
```

A quick smoke test of the Experiment B pipeline:

```bash
python experiment_b.py
```

Run the test suite:

```bash
pytest -q
```

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

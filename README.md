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
| Experiment B, L3 (learned-dynamics) | **positive at n=10, replicated at a second capacity (behavior-independent signal ~0.72 at both; survival-specificity holds at the subtler artifact only); held-out probe: transfers to an unseen fingerprint subtle-to-coarse (0.773) but not coarse-to-subtle (0.638, frozen rule fails) and is reactive, not a persistent representation; cross-recipe probe: also reads a different surrogate family (0.684, rule passes)** |
| Experiment C (emergence under selection) / Ladder L4 | not started |

### Key result

*For a hand-tuned dynamics artifact (L2), detectability does not imply incidental
encoding.* An artifact an external oracle detects at AUROC ≈ 0.99 leaves essentially no
decodable trace in a from-scratch agent's representation (≈ 0.5), across three readout
channels and surviving objective-engagement, nonlinear-probe, scale (n=10), and
capacity-ceiling checks.

*But the picture changes at L3.* When the surrogate is a **learned-dynamics fingerprint**
(a small net replacing the velocity law) rather than a hand-tuned knob, the **survival**
agent, and at this fingerprint only the survival agent, encodes it. At a difficulty where an untrained net is at
chance (about 0.49, and still only 0.52 under a nonlinear probe) and a prediction-only agent
is near chance (about 0.57), the survival probe reads **0.752** (n = 10, honest t-based 90% CI
**[0.698, 0.807]**, which excludes the pre-registered 0.65 bar; 8 of 10 seeds clear it). The
dissociation is robust: it is not reward-mediated (world is not decodable from summed reward,
AUROC 0.541, clean 10 of 10 seeds), not survivorship-biased (0 early deaths, every pool
110/110), not a linear-probe artifact (the untrained net stays near chance even nonlinearly),
and the L0 authentic-vs-authentic control is at chance (0.517). **And the signal is not just
behavior.** The agent does move and forage differently in the two worlds - in fact the full
behavior trace alone decodes the world at **0.803**, better than the state probe itself - so
the obvious deflationary reading was that the probe reads behavior, not a representation. A
pre-registered per-timestep control (dump every step's speed/energy/food/drag, residualize the
recurrent state on the behavior trace in-fold, probe what is left) rejects that reading: the
behavior-independent world-signal is **0.726** (t-based 90% CI **[0.679, 0.772]**, which
excludes the 0.65 bar; the seed-level bootstrap interval [0.685, 0.765] agrees; 9 of 10 seeds
clear it; quadratic variant 0.721). The control is honest on its own
negative controls: the untrained agent's state reads exact chance (0.498) under the same
control even though untrained *behavior* decodes 0.645, and the prediction-only agent stays
near chance (0.574). (An earlier, cruder per-episode-mean control had under-estimated the
signal at ~0.66 by over-removing - the attenuation our synthetic tests predicted.) The honest
statement is: reward- and survivorship-controlled, robust to nonlinear probing, with a
behavior-independent world-signal of about **0.73** that clears the pre-registered bar. This
is the first place "detectable does not imply learned" reverses: a from-scratch agent, never
rewarded for it, comes to carry world-discriminative state as a byproduct of surviving. The
mediation audit is reproducible code (`scripts/audit_behavior_mediation.py`; artifacts in
`artifacts/expB2/`). *A pre-registered replication at a second calibrated capacity sharpens
the claim.* The second in-band fingerprint (hidden = 7, selected by a frozen fallback rule
after hidden = 4 failed its gates) passes every gate and replicates the behavior-independent
world-signal almost exactly: **0.722** (t-based 90% CI [0.672, 0.773]) vs 0.726 at hidden = 8. But
that coarser artifact is one every trained agent picks up (predictor 0.714 vs survival 0.737,
under the pre-registered +0.05 dissociation requirement), so the survival-*only* part of the
claim is conditional on the subtler hidden = 8 artifact. What survives both capacities is a
reward-clean, survivorship-clean, behavior-independent world-signal of about **0.72** in the
survival agent's state. *A held-out probe (n = 10, committed per-seed summary in
`artifacts/expB2/heldout_l3_h8_summary.json`) then sharpens what that signal is.* It splits: the world-identity direction learned against the trained fingerprint
still reads an unseen same-recipe fingerprint the agent never lived with (transfer **0.773** vs
untrained floor 0.569; the pre-registered rule passes), so it is not an overfit to one artifact
instance. A frozen reverse probe (train on the coarser fingerprint, read the subtler one;
per-seed summary in `artifacts/expB2/heldout_l3_h7_reverse_summary.json`) fails its bar at
**0.638**, so this transfer is direction-dependent: it generalizes from subtle training
artifacts, not bidirectionally. And under a common-garden control that runs both groups through an identical tail after
differing prefixes, tail-only state does not carry the prefix world (**0.557**, below the 0.65 bar;
the late tail decays to chance, 0.492). So the L3 world-signal is best read as reactive tracking of
the currently-felt dynamics, not a persistent stored world-identity representation. A cross-recipe
probe (n = 10, `artifacts/l3_crossrecipe/summary.json`) then extends the transfer half across
surrogate *families*: the same direction reads a gate-calibrated random-Fourier-features ridge law
the agent never lived with (**0.684** vs untrained floor 0.548; the pre-registered rule passes,
machine-checked), so the reactive world-signal is recipe-general, not a signature of one function
class. See
[`docs/FINDINGS.md`](docs/FINDINGS.md) and [`docs/PREREGISTRATION_L3.md`](docs/PREREGISTRATION_L3.md).

---

## Repository layout

```
.
|-- README.md                   this file: the map
|-- LICENSE
|-- CITATION.cff                citation metadata (name, ORCID, version)
|-- pyproject.toml              package metadata (pip install -e .)
|-- requirements.txt            runtime dependencies
|-- requirements-dev.txt        pytest + dev tooling
|-- itasorl/                    core library (world, agents, experiments)
|   |-- world.py                World protocol, surrogate ladder, matched-pair harness
|   |-- patch_of_earth.py       PatchOfEarthV0 concrete world, incl. L1/L2 hooks
|   |-- agent.py                recurrent world model (RSSM-lite)
|   |-- agent_ac.py             survival actor-critic (Experiment B-v2)
|   |-- experiment_a.py         agent-free L1 detectability oracle
|   |-- experiment_a_l2.py      agent-free L2 detectability oracle
|   |-- experiment_a_l3.py      agent-free L3 (learned-fingerprint) oracle
|   |-- experiment_b.py         incidental-detection harness
|   |-- experiment_b2.py        survival-coupled B-v2 pipeline
|   |-- surrogate_l3.py         L3 learned-dynamics surrogate (G_motion)
|   |-- behavior_audit.py       behavior-mediation controls (residual probes)
|   |-- stats.py                TOST/ROPE equivalence, bootstrap AUROC CIs
|   `-- results_io.py           end-to-end run recording
|-- scripts/                    deterministic reproduction runners
|   |-- run_e2e.py              pytest + all experiments (recorded)
|   |-- run_expA.py ...         Experiment A/B runners
|   |-- run_expB2.py            Experiment B-v2 / L3 (GPU if available)
|   `-- audit_behavior_mediation.py  behavior-mediation audit on dumped states
|-- docs/
|   |-- ITASORL.md              research plan
|   |-- FINDINGS.md             empirical results
|   |-- PREREGISTRATION.md      B-v2 pre-registration
|   |-- PREREGISTRATION_Bv3.md  B-v3 pre-registration
|   |-- PREREGISTRATION_L3.md   L3 pre-registration + deviation log
|   `-- figures/                result figures (.png) + provenance README
|-- artifacts/expB2/            published B-v2/L3 JSON/PNG (committed)
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

To regenerate all of them in one recorded pass, run `python scripts/run_e2e.py --quick`
from the repo root; each `scripts/run_exp*.py` runner rewrites only its own figure.
Per-figure provenance (which script, which doc section) is tracked in
[`docs/figures/README.md`](docs/figures/README.md).

---

## How to run

Dependencies:

```bash
pip install -r requirements.txt      # runtime deps
pip install -e .                     # or: install itasorl as an editable package
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

## Citing

Citation metadata lives in [`CITATION.cff`](CITATION.cff); GitHub renders a
"Cite this repository" button from it, or run `cffconvert -f bibtex` for BibTeX.

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

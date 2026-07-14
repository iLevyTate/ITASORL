# L3 Held-Out Fingerprint + Common-Garden Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two evaluation channels to the B-v2 pipeline (probe transfer to an unseen L3 fingerprint, and a common-garden shared-tail probe) plus agent persistence, per `docs/superpowers/specs/2026-07-14-l3-heldout-common-garden-probe-design.md`.

**Architecture:** All science code lives in `itasorl/experiment_b2.py` as three new testable units (`transfer_probe` pure probe, `transfer_readout` collection wrapper, `common_garden_rollout` + `cg_probe`), plumbed into `scripts/run_expB2.py` behind strict no-op flags (`--heldout-evals`, `--save-agents`). The config fingerprint stays byte-identical when flags are off so old checkpoints still resume.

**Tech Stack:** Python 3.10-3.12, numpy, torch, scikit-learn, pytest. Repo style: compact, ruff config ignores E401/E701/E702/E731/E741 but F-rules are live (no f-strings without placeholders, F541).

**Key existing code facts (verified 2026-07-14):**
- Linear probe family: `make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))` (itasorl/experiment_a.py:126 `grouped_auroc`).
- `episode_features(H)` for (n, T, hid) arrays: `np.concatenate([H.mean(1), H[:, -1]], axis=-1)` (itasorl/experiment_b.py:109).
- `_episode_feature(H)` for a single (T, hid) episode: `np.concatenate([H.mean(0), H[-1]])` (experiment_b2.py:480).
- `pooled_readout` collects auth pool at fixed `seed_base=800_000`, surr pool at `850_000` (experiment_b2.py:694-699). New seed bases must not collide: fresh-auth transfer pool = 860_000, heldout-surr pool = 870_000, common-garden = 930_000.
- L3 surrogate is module-global `_L3_GMOTION`, installed by `setup_l3_surrogate(**kwargs)` which calls `itasorl.surrogate_l3.train_g_motion` (experiment_b2.py:75-82). `make_world(params, drift_sigma, ray_steps)` attaches it only when `DRIFT_MODE == "l3"` and `drift_sigma > 0`.
- World snapshot/restore: `w.get_state()` / `w.set_state(s)` are exact incl. RNG (patch_of_earth.py:270-290). `set_state` reads `drift_w` with `.get(..., 0.0)`.
- `_run_branch(agent, norm, world, h0, prev_act0, branch_steps, device)` steps a frozen deterministic agent and returns `(H (T,hid), speeds, rewards, alive)` (experiment_b2.py:396).
- Cell checkpoints: `run_cell` returns a picklable dict; `config_fingerprint(base)` hashes all keys except `dump_states` (run_expB2.py:118-123).
- CI runs `ruff check .` and `pytest -q` on Python 3.10/3.11/3.12 (CPU only; CUDA tests use the `DEVICES` list pattern in tests/test_experiment_b2.py:34).
- Tests import pattern: `torch = pytest.importorskip("torch")` then `from itasorl.experiment_b2 import ...` with `# noqa: E402`.
- `auroc_ci` is already imported in experiment_b2.py (line 38, `from .stats import auroc_ci`); `train_g_motion(*, hidden=8, n_eps=250, steps=40, epochs=300, ...)` also accepts `params`, `device`, `seed` (call shape proven at run_expB2.py:212).

---

### Task 1: `transfer_probe` (pure fit-frozen/score-frozen probe)

**Files:**
- Modify: `itasorl/experiment_b2.py` (add function after `leakage_audit_b2`, ~line 517)
- Create: `tests/test_heldout_transfer.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_heldout_transfer.py`:

```python
"""Tests for the held-out fingerprint transfer channel (spec:
docs/superpowers/specs/2026-07-14-l3-heldout-common-garden-probe-design.md).

Synthetic ground truth: a fingerprint-GENERAL signal (same discriminative
direction in train and test pools) must transfer; a fingerprint-SPECIFIC
signal (orthogonal directions) must not."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from itasorl.experiment_b2 import transfer_probe  # noqa: E402


def _pools(rng, n, dim, sig_dim, shift):
    """Two pools of feature vectors separated by `shift` along axis sig_dim."""
    Xa = rng.normal(0, 1, (n, dim))
    Xs = rng.normal(0, 1, (n, dim))
    Xs[:, sig_dim] += shift
    X = np.concatenate([Xa, Xs])
    y = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)
    return X, y


def test_transfer_probe_general_signal_transfers():
    rng = np.random.default_rng(0)
    Xtr, ytr = _pools(rng, 80, 16, sig_dim=3, shift=3.0)
    Xte, yte = _pools(rng, 80, 16, sig_dim=3, shift=3.0)   # SAME direction
    assert transfer_probe(Xtr, ytr, Xte, yte) > 0.9


def test_transfer_probe_specific_signal_does_not_transfer():
    rng = np.random.default_rng(1)
    Xtr, ytr = _pools(rng, 80, 16, sig_dim=3, shift=3.0)
    Xte, yte = _pools(rng, 80, 16, sig_dim=11, shift=3.0)  # ORTHOGONAL direction
    assert abs(transfer_probe(Xtr, ytr, Xte, yte) - 0.5) < 0.15


def test_transfer_probe_degenerate_labels_nan():
    rng = np.random.default_rng(2)
    Xtr, ytr = _pools(rng, 20, 8, 0, 2.0)
    Xte = rng.normal(0, 1, (10, 8))
    yte = np.zeros(10, int)                                 # one class only
    assert np.isnan(transfer_probe(Xtr, ytr, Xte, yte))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_heldout_transfer.py -q`
Expected: 3 failures/errors, `ImportError: cannot import name 'transfer_probe'`

- [ ] **Step 3: Implement `transfer_probe`**

In `itasorl/experiment_b2.py`, after `leakage_audit_b2` (before the "Control agents" banner comment):

```python
def transfer_probe(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray,
                   return_scores: bool = False):
    """Held-out fingerprint transfer: fit the STANDARD linear probe family (same
    scaler+logistic pipeline grouped_auroc uses) once on the training pools, then
    score a FROZEN AUROC on disjoint test pools. No CV: train and test worlds are
    disjoint by construction, and the frozen score is the estimand (does the
    direction learned on the trained fingerprint generalize to an unseen one).
    return_scores=True also returns (yte, p_te) so callers can bootstrap a CI
    with itasorl.stats.auroc_ci (no refit), mirroring _auroc_with_ci."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        nan = float("nan")
        return (nan, yte, np.array([])) if return_scores else nan
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    auc = float(roc_auc_score(yte, p))
    return (auc, yte, p) if return_scores else auc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_heldout_transfer.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add itasorl/experiment_b2.py tests/test_heldout_transfer.py
git commit -m "feat(heldout): transfer_probe - frozen-evaluation linear probe for unseen-fingerprint transfer"
```

---

### Task 2: Held-out surrogate management + `transfer_readout`

**Files:**
- Modify: `itasorl/experiment_b2.py` (globals near `_L3_GMOTION` ~line 72; `pooled_readout` ~line 679; new `transfer_readout` after `pooled_readout`)
- Test: `tests/test_heldout_transfer.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_heldout_transfer.py`:

```python
from itasorl.experiment_b2 import (  # noqa: E402
    pooled_readout, transfer_readout, untrained_agent,
)
import itasorl.experiment_b2 as b2  # noqa: E402
from itasorl.world import WorldParams  # noqa: E402

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
RS = 4
TINY_G = dict(n_eps=6, steps=10, epochs=3, device="cpu", seed=0)


@pytest.fixture()
def l3_tiny():
    """Install tiny trained + held-out surrogates; restore module state after."""
    saved = (b2.DRIFT_MODE, b2._L3_GMOTION, b2._L3_GMOTION_HELDOUT)
    b2.DRIFT_MODE = "l3"
    b2.setup_l3_surrogate(hidden=2, params=P, **TINY_G)
    b2.setup_l3_heldout_surrogate(hidden=3, params=P, **TINY_G)
    yield
    b2.DRIFT_MODE, b2._L3_GMOTION, b2._L3_GMOTION_HELDOUT = saved


def test_pooled_readout_return_pools_shape():
    agent, norm = untrained_agent(P, 0.0, RS, hidden=8, embed=16, world_model=True,
                                  device="cpu", seed=0)
    out, (Ha, Hs) = pooled_readout(agent, norm, P, 0.0, n_eps=6, steps=5, ray_steps=RS,
                                   device="cpu", return_pools=True)
    assert isinstance(out, dict) and Ha.ndim == 3 and Hs.ndim == 3
    # default call still returns the bare dict (schema no-op)
    out2 = pooled_readout(agent, norm, P, 0.0, n_eps=6, steps=5, ray_steps=RS, device="cpu")
    assert isinstance(out2, dict) and set(out2) == set(out)


def test_transfer_readout_runs_and_restores_surrogate(l3_tiny):
    agent, norm = untrained_agent(P, 0.45, RS, hidden=8, embed=16, world_model=True,
                                  device="cpu", seed=0)
    _, (Ha, Hs) = pooled_readout(agent, norm, P, 0.45, n_eps=8, steps=5, ray_steps=RS,
                                 device="cpu", return_pools=True)
    before = b2._L3_GMOTION
    tr = transfer_readout(agent, norm, P, 0.45, Ha, Hs, n_eps=8, steps=5,
                          ray_steps=RS, device="cpu")
    assert b2._L3_GMOTION is before, "trained surrogate must be restored after transfer eval"
    assert set(tr) >= {"transfer_target", "transfer_n_auth", "transfer_n_surr"}
    assert np.isnan(tr["transfer_target"]) or 0.0 <= tr["transfer_target"] <= 1.0


def test_transfer_readout_requires_heldout_installed(l3_tiny):
    b2._L3_GMOTION_HELDOUT = None
    agent, norm = untrained_agent(P, 0.45, RS, hidden=8, embed=16, world_model=True,
                                  device="cpu", seed=0)
    with pytest.raises(RuntimeError):
        transfer_readout(agent, norm, P, 0.45, np.zeros((5, 5, 8), np.float32),
                         np.zeros((5, 5, 8), np.float32), n_eps=5, steps=5,
                         ray_steps=RS, device="cpu")
```

Note: check `itasorl/surrogate_l3.py:67` `train_g_motion` kwargs before finalizing
`TINY_G`; it accepts `hidden, n_eps, steps, epochs, ...` plus `params/device/seed`
(same call shape `run_cell` uses at run_expB2.py:212). Adjust names to match if any differ.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_heldout_transfer.py -q`
Expected: new tests error with `cannot import name 'transfer_readout'`

- [ ] **Step 3: Implement**

In `itasorl/experiment_b2.py`. First, next to `_L3_GMOTION` (~line 72):

```python
# Held-out L3 fingerprint (spec 2026-07-14): a SECOND trained G, never seen by any
# agent during training, used only by transfer_readout. None until setup runs.
_L3_GMOTION_HELDOUT = None


def setup_l3_heldout_surrogate(**train_kwargs) -> None:
    """Train the held-out G (different capacity, same frozen recipe) and store it for
    transfer_readout. Does NOT touch the training surrogate _L3_GMOTION."""
    global _L3_GMOTION_HELDOUT
    from .surrogate_l3 import train_g_motion
    _L3_GMOTION_HELDOUT = train_g_motion(**train_kwargs)
```

Second, change `pooled_readout` signature (~line 679) to
`def pooled_readout(agent, norm, params, drift_sigma, *, n_eps=110, steps=24, ray_steps=5, device=None, seed=0, dump_path=None, leak_margin=0.1, return_pools=False) -> dict:`
and change BOTH return statements:
- the early `too_few_survivors` return (~line 709): wrap as `out = { ... }` then
  `return (out, (Ha, Hs)) if return_pools else out`
- the final return (~line 755): same pattern, `out = { ... }` then
  `return (out, (Ha, Hs)) if return_pools else out`

Third, add after `pooled_readout`:

```python
def transfer_readout(agent, norm, params, drift_sigma, Ha_train, Hs_train, *,
                     n_eps=110, steps=24, ray_steps=5, device=None, seed=0,
                     dump_path=None) -> dict:
    """Unseen-fingerprint transfer channel (spec 2026-07-14). Fits the standard
    linear probe on the TRAINED-fingerprint pools (Ha_train vs Hs_train, i.e. the
    exact pools pooled_readout probed), then scores it FROZEN on a fresh authentic
    pool vs a pool collected under the HELD-OUT surrogate _L3_GMOTION_HELDOUT.
    Fresh authentic pool: the probe must never be tested on authentic episodes it
    trained on. Restores the training surrogate in a finally block so the global
    can never leak into later evals."""
    global _L3_GMOTION
    if _L3_GMOTION_HELDOUT is None:
        raise RuntimeError("transfer_readout: call setup_l3_heldout_surrogate() first")
    device = device or default_device()
    saved = _L3_GMOTION
    try:
        _L3_GMOTION = _L3_GMOTION_HELDOUT
        Ha2, _ = collect_pool(agent, norm, params, 0.0, n_eps, steps, device, 860_000, ray_steps)
        H7, _ = collect_pool(agent, norm, params, drift_sigma, n_eps, steps, device, 870_000, ray_steps)
    finally:
        _L3_GMOTION = saved
    if dump_path is not None:
        d = os.path.dirname(dump_path)
        if d:
            os.makedirs(d, exist_ok=True)
        np.savez_compressed(dump_path, Ha2=Ha2, H7=H7,
                            drift_sigma=np.float64(drift_sigma), steps=np.int64(steps))
    nan = float("nan")
    out = {"transfer_n_auth": int(len(Ha2)), "transfer_n_surr": int(len(H7)),
           "transfer_deaths_auth": int(n_eps - len(Ha2)),
           "transfer_deaths_surr": int(n_eps - len(H7)),
           "transfer_target": nan, "transfer_lo": nan, "transfer_hi": nan}
    if len(Ha_train) < 5 or len(Hs_train) < 5 or len(Ha2) < 5 or len(H7) < 5:
        return out
    Xtr = episode_features(np.concatenate([Ha_train, Hs_train]))
    ytr = np.concatenate([np.zeros(len(Ha_train)), np.ones(len(Hs_train))]).astype(int)
    Xte = episode_features(np.concatenate([Ha2, H7]))
    yte = np.concatenate([np.zeros(len(Ha2)), np.ones(len(H7))]).astype(int)
    auc, yv, pv = transfer_probe(Xtr, ytr, Xte, yte, return_scores=True)
    out["transfer_target"] = auc
    if pv.size:
        out["transfer_lo"], out["transfer_hi"] = auroc_ci(yv, pv, seed=seed)
    return out
```

Note the authentic transfer pool is collected INSIDE the swap block: harmless
(authentic worlds never attach a G, experiment_b2.py:90-91) and keeps the
swap/restore in one place.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_heldout_transfer.py -q`
Expected: all pass (tiny G training takes a few seconds on CPU)

- [ ] **Step 5: Run the full suite (regression) and ruff**

Run: `python -m pytest -q && ruff check .`
Expected: all pass, no lint errors

- [ ] **Step 6: Commit**

```bash
git add itasorl/experiment_b2.py tests/test_heldout_transfer.py
git commit -m "feat(heldout): held-out surrogate slot + transfer_readout with swap/restore"
```

---

### Task 3: `common_garden_rollout` + `cg_probe`

**Files:**
- Modify: `itasorl/experiment_b2.py` (add both after `transfer_readout`)
- Create: `tests/test_common_garden.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_common_garden.py`:

```python
"""Common-garden channel tests (spec 2026-07-14): differing prefix world,
identical authentic tail; the probe reads tail-only states.

Load-bearing guarantees mirror test_experiment_b2.py:
  - L0 (drift off): auth- and surr-prefix tails are BIT-IDENTICAL, so the
    channel manufactures no signal;
  - drift on: tails diverge (prefix history is the only difference);
  - on synthetic tails, a PERSISTENT group signal scores high on both windows
    while a REACTIVE (early-only) signal collapses to chance on the late window."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from itasorl.agent_ac import RecurrentActorCritic  # noqa: E402
from itasorl.experiment_b2 import RunningNorm, cg_probe, common_garden_rollout  # noqa: E402
from itasorl.world import WorldParams  # noqa: E402

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
RS = 4


def _agent_norm():
    from itasorl.patch_of_earth import PatchOfEarthV0
    w = PatchOfEarthV0(P)
    torch.manual_seed(0)
    agent = RecurrentActorCritic(w.obs_spec.size, w.action_spec.size, embed=16, hidden=8).train(False)
    return agent, RunningNorm(w.obs_spec.size).freeze()


def test_cg_L0_tails_bit_identical():
    agent, norm = _agent_norm()
    auth, surr = common_garden_rollout(agent, norm, P, 0.0, n_pairs=3, prefix_steps=4,
                                       tail_steps=6, ray_steps=RS, device="cpu")
    assert len(auth) >= 1
    for a, s in zip(auth, surr):
        assert np.array_equal(a, s), "L0 common-garden tails diverged - channel is not confound-free"


def test_cg_drift_tails_diverge():
    agent, norm = _agent_norm()
    auth, surr = common_garden_rollout(agent, norm, P, 0.5, n_pairs=3, prefix_steps=4,
                                       tail_steps=6, ray_steps=RS, device="cpu")
    assert any(not np.array_equal(a, s) for a, s in zip(auth, surr)), \
        "drift-on prefixes left identical tails - snapshot carry is broken"


def _synthetic_tails(rng, n, T, hid, offset_fn):
    """Tails where group 1 carries offset_fn(t) added to one hidden unit."""
    auth = [rng.normal(0, 1, (T, hid)).astype(np.float32) for _ in range(n)]
    surr = []
    for _ in range(n):
        H = rng.normal(0, 1, (T, hid)).astype(np.float32)
        H[:, 0] += np.array([offset_fn(t) for t in range(T)], np.float32)
        surr.append(H)
    return auth, surr


def test_cg_probe_persistent_vs_reactive():
    rng = np.random.default_rng(0)
    T = 24
    pers_a, pers_s = _synthetic_tails(rng, 60, T, 8, lambda t: 3.0)          # persistent
    reac_a, reac_s = _synthetic_tails(rng, 60, T, 8, lambda t: 3.0 if t < 4 else 0.0)  # reactive
    pers = cg_probe(pers_a, pers_s, late_k=8, seed=0)
    reac = cg_probe(reac_a, reac_s, late_k=8, seed=0)
    assert pers["cg_tail_target"] > 0.9 and pers["cg_latetail_target"] > 0.9
    assert abs(reac["cg_latetail_target"] - 0.5) < 0.15, "late window must not see an early-only signal"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_common_garden.py -q`
Expected: ImportError on `cg_probe` / `common_garden_rollout`

- [ ] **Step 3: Implement**

In `itasorl/experiment_b2.py` after `transfer_readout`:

```python
def common_garden_rollout(agent, norm, params, drift_sigma, *, n_pairs=110,
                          prefix_steps=20, tail_steps=24, ray_steps=5,
                          device=None, seed_base=930_000) -> tuple[list, list]:
    """Common-garden channel (spec 2026-07-14): PAIRED episodes from identical
    seeds run their prefix in the authentic vs the surrogate world, then BOTH
    continue under authentic dynamics (fresh authentic world restored from each
    prefix's final snapshot, drift_w forced to 0). Returns (auth_tails,
    surr_tails): lists of (tail_steps, hidden) arrays, tail-only states. A pair
    is dropped if EITHER member dies in prefix or tail (symmetric, so
    survivorship cannot create asymmetry)."""
    device = device or default_device()
    auth_tails, surr_tails = [], []
    for p in range(n_pairs):
        seeds = _seeds(seed_base + p)
        pair = []
        for dsig in (0.0, drift_sigma):
            w = make_world(params, dsig, ray_steps)
            w.reset(seeds)
            h = agent.initial_state(1, device)
            prev = torch.zeros(1, agent.act_dim, device=device)
            obs = w.observe().astype(np.float64)
            died = False
            for _ in range(prefix_steps):
                obs_t = torch.as_tensor(norm(obs)[None], dtype=torch.float32, device=device)
                _, env_act, _, _, h = agent.act(obs_t, prev, h, deterministic=True)
                r = w.step(env_act[0].detach().cpu().numpy().astype(np.float32))
                obs = r.obs.astype(np.float64)
                prev = env_act
                if r.terminated:
                    died = True
                    break
            if died:
                pair.append(None)
                continue
            tail = make_world(params, 0.0, ray_steps)   # common garden: authentic dynamics
            tail.reset(seeds)
            tail.set_state({**w.get_state(), "drift_w": 0.0})
            Ht, _, _, alive = _run_branch(agent, norm, tail, h, prev, tail_steps, device)
            pair.append(Ht if (alive and len(Ht) == tail_steps) else None)
        if pair[0] is not None and pair[1] is not None:
            auth_tails.append(pair[0])
            surr_tails.append(pair[1])
    return auth_tails, surr_tails


def cg_probe(auth_tails: list, surr_tails: list, *, late_k: int = 8, seed: int = 0) -> dict:
    """Probe tail-only states for the PREFIX world. cg_tail_target uses the full
    tail's [mean h, final h]; cg_latetail_target repeats it on the last late_k
    steps only - the persistence-decay check (a reactive signal washes out along
    the tail; a persistent representation does not)."""
    n = len(auth_tails)
    out = {"cg_n_pairs": n, "cg_tail_target": float("nan"), "cg_tail_lo": float("nan"),
           "cg_tail_hi": float("nan"), "cg_latetail_target": float("nan")}
    if n < 5:
        return out
    y = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)
    X = np.stack([_episode_feature(H) for H in auth_tails + surr_tails])
    k = min(late_k, auth_tails[0].shape[0])
    Xl = np.stack([_episode_feature(H[-k:]) for H in auth_tails + surr_tails])
    tgt, lo, hi = _auroc_with_ci(X, y, seed=seed)
    out.update(cg_tail_target=tgt, cg_tail_lo=lo, cg_tail_hi=hi,
               cg_latetail_target=probe_auroc(Xl, y))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_common_garden.py -q`
Expected: `3 passed` (L0 bit-identity relies on get_state/set_state exactness,
already proven by test_matched_pair_L0_is_bit_identical)

If `test_cg_L0_tails_bit_identical` fails: the likely cause is the prefix world
at dsig=0.0 vs drift_sigma paths differing in RNG draws at reset. Debug against
`matched_pair_recurrent_rollout` (experiment_b2.py:420-474), which solved the
same problem; do NOT weaken the assertion.

- [ ] **Step 5: Full suite + ruff**

Run: `python -m pytest -q && ruff check .`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add itasorl/experiment_b2.py tests/test_common_garden.py
git commit -m "feat(heldout): common_garden_rollout + cg_probe with L0 bit-identity guarantee"
```

---

### Task 4: Agent persistence (`save_agent_bundle` / `load_agent_bundle`)

**Files:**
- Modify: `itasorl/experiment_b2.py` (add after `cg_probe`)
- Create: `tests/test_agent_bundle.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_agent_bundle.py`:

```python
"""save/load round-trip for trained-agent bundles (spec 2026-07-14: agents were
never persisted, which forced a full retrain when new eval channels appeared)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from itasorl.experiment_b2 import (  # noqa: E402
    load_agent_bundle, save_agent_bundle, untrained_agent,
)
from itasorl.world import WorldParams  # noqa: E402

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)


def test_bundle_round_trip(tmp_path):
    agent, norm = untrained_agent(P, 0.45, 4, hidden=8, embed=16, world_model=True,
                                  device="cpu", seed=0)
    path = tmp_path / "agent.pt"
    save_agent_bundle(str(path), agent, norm)
    agent2, norm2 = load_agent_bundle(str(path), device="cpu")
    obs = np.linspace(-1, 1, agent.obs_dim)[None]        # (1, obs_dim)
    x = torch.as_tensor(norm(obs), dtype=torch.float32)
    x2 = torch.as_tensor(norm2(obs), dtype=torch.float32)
    assert torch.allclose(x, x2)
    h = agent.initial_state(1, "cpu")
    prev = torch.zeros(1, agent.act_dim)
    _, a1, _, _, _ = agent.act(x, prev, h, deterministic=True)
    _, a2, _, _, _ = agent2.act(x2, prev, agent2.initial_state(1, "cpu"), deterministic=True)
    assert torch.allclose(a1, a2), "reloaded agent must act identically"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent_bundle.py -q`
Expected: ImportError on `save_agent_bundle`

- [ ] **Step 3: Implement**

In `itasorl/experiment_b2.py` after `cg_probe`:

```python
def save_agent_bundle(path: str, agent: RecurrentActorCritic, norm: RunningNorm) -> None:
    """Persist a frozen agent + its frozen obs normalizer with the constructor args
    needed to rebuild it. A few MB; prevents ever again losing trained agents."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    torch.save({"state_dict": agent.state_dict(),
                "ctor": {"obs_dim": agent.obs_dim, "act_dim": agent.act_dim,
                         "embed": agent.encoder[0].out_features, "hidden": agent.hidden,
                         "world_model": agent.world_model, "sysid_aux": agent.sysid_aux},
                "norm": {"mean": norm.mean, "var": norm.var, "count": norm.count}}, path)


def load_agent_bundle(path: str, device: str = "cpu"):
    """Rebuild (agent, norm) from save_agent_bundle output. Returns them frozen."""
    blob = torch.load(path, map_location=device, weights_only=False)
    agent = RecurrentActorCritic(**blob["ctor"]).to(device)
    agent.load_state_dict(blob["state_dict"])
    norm = RunningNorm(blob["ctor"]["obs_dim"])
    norm.mean, norm.var, norm.count = blob["norm"]["mean"], blob["norm"]["var"], blob["norm"]["count"]
    return agent.train(False), norm.freeze()
```

Attribute facts (verified against itasorl/agent_ac.py:38-63): `obs_dim`,
`act_dim`, `hidden` are stored attributes; `world_model` and `sysid_aux` are
stored BOOLS; `embed` is NOT stored, recover it as
`agent.encoder[0].out_features` (the first Linear's width).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_agent_bundle.py -q`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add itasorl/experiment_b2.py tests/test_agent_bundle.py
git commit -m "feat(heldout): agent bundle save/load round-trip"
```

---

### Task 5: Runner plumbing (flags, fingerprint no-op, cell wiring)

**Files:**
- Modify: `scripts/run_expB2.py` (cfg ~line 61, config_fingerprint ~line 118, evaluate_agent ~line 182, run_cell ~line 193, record_cell ~line 236, print_cell ~line 263, main ~line 304, summary ~line 404)
- Create: `tests/test_heldout_runner.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_heldout_runner.py`:

```python
"""Runner-level guarantees for the heldout flags (spec 2026-07-14):
1) FINGERPRINT NO-OP: with the new flags OFF, config_fingerprint must equal the
   hash of the pre-change key set, so old runs still --resume.
2) With --heldout-evals ON, the fingerprint changes (heldout cells never mix).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _bootstrap  # noqa: F401, E402
from run_expB2 import config_fingerprint  # noqa: E402

OLD_KEYS = ("updates", "n_eps", "max_steps", "hidden", "ray_steps", "shaping_coef",
            "pool_n", "pool_steps", "mp_pairs", "mp_prefix", "mp_branch", "basal_e",
            "n_pellets", "reach", "dump_states", "sysid_aux", "sysid_coef",
            "drift_mode", "l3_hidden")


def _old_base():
    b = {k: None for k in OLD_KEYS}
    b.update(updates=300, n_eps=16, max_steps=80, hidden=96, ray_steps=5,
             shaping_coef=1.0, pool_n=110, pool_steps=24, mp_pairs=60, mp_prefix=20,
             mp_branch=24, sysid_aux=False, sysid_coef=1.0, drift_mode="l3",
             l3_hidden=8, drifts=[0.0, 0.45], device="cuda")
    return b


def test_fingerprint_noop_when_flags_off():
    old = _old_base()
    new = {**old, "out_dir": "somewhere", "save_agents": True}  # IO keys, excluded
    assert config_fingerprint(new) == config_fingerprint(old)


def test_fingerprint_changes_when_heldout_on():
    old = _old_base()
    new = {**old, "heldout_evals": True, "heldout_hidden": 7, "cg_prefix": 20, "cg_steps": 24}
    assert config_fingerprint(new) != config_fingerprint(old)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_heldout_runner.py -q`
Expected: `test_fingerprint_noop_when_flags_off` FAILS (out_dir/save_agents not yet excluded)

- [ ] **Step 3: Implement the runner changes**

All in `scripts/run_expB2.py`.

3a. `cfg()` - add after the `--dump-states` argument (~line 81):

```python
    ap.add_argument("--heldout-evals", action="store_true",
                    help="L3 only: add the unseen-fingerprint transfer + common-garden "
                         "eval channels per cell (spec 2026-07-14)")
    ap.add_argument("--heldout-hidden", type=int, default=7,
                    help="G_motion capacity of the HELD-OUT fingerprint (frozen: 7)")
    ap.add_argument("--cg-prefix", type=int, default=20, help="common-garden prefix steps")
    ap.add_argument("--cg-steps", type=int, default=24, help="common-garden tail steps")
    ap.add_argument("--save-agents", action="store_true",
                    help="persist each trained arm to <out-dir>/agents/ (a few MB each)")
```

And at the end of `cfg()` before `return a`:

```python
    if a.heldout_evals and a.drift_mode != "l3":
        raise SystemExit("--heldout-evals requires --drift-mode l3")
    if a.quick and a.heldout_evals:
        a.cg_prefix, a.cg_steps = 8, 12
    return a
```

3b. `config_fingerprint` (~line 121) - exclude IO keys:

```python
    fp = {k: v for k, v in base.items() if k not in ("dump_states", "save_agents", "out_dir")}
```

3c. `main()` base dict (~line 344) - add IO keys always, science keys only when on
(keeps the flags-off fingerprint identical to old runs):

```python
    base = {k: getattr(a, k) for k in ("updates", "n_eps", "max_steps", "hidden", "ray_steps",
                                       "shaping_coef", "pool_n", "pool_steps", "mp_pairs", "mp_prefix",
                                       "mp_branch", "basal_e", "n_pellets", "reach", "dump_states",
                                       "sysid_aux", "sysid_coef", "drift_mode", "l3_hidden")}
    base.update(drifts=a.drifts, device=dev, out_dir=a.out_dir, save_agents=a.save_agents)
    if a.heldout_evals:
        base.update(heldout_evals=True, heldout_hidden=a.heldout_hidden,
                    cg_prefix=a.cg_prefix, cg_steps=a.cg_steps)
```

Also in `main()`, after the existing `b2.setup_l3_surrogate(...)` line (~line 329):

```python
        if a.heldout_evals:
            print(f"  heldout evals ON: transfer fingerprint G(hidden={a.heldout_hidden}), "
                  f"common garden prefix={a.cg_prefix} tail={a.cg_steps}")
            b2.setup_l3_heldout_surrogate(hidden=a.heldout_hidden, device=dev, seed=0, params=P)
```

(`setup_l3_heldout_surrogate` must be added to the `from itasorl.experiment_b2 import ...`
block, or called as `b2.setup_l3_heldout_surrogate` - use the `b2.` form, matching
`b2.setup_l3_surrogate` at line 329.)

3d. `run_cell` (~line 211) - after the existing per-worker L3 setup:

```python
    if k.get("heldout_evals") and b2._L3_GMOTION_HELDOUT is None:  # once per worker
        b2.setup_l3_heldout_surrogate(hidden=k["heldout_hidden"], device=dev, seed=0, params=P)
```

And after `agents["survival"] = (sa, sn)` (~line 223):

```python
    if k.get("save_agents"):
        from itasorl.experiment_b2 import save_agent_bundle
        for g, (ag, nm) in agents.items():
            save_agent_bundle(os.path.join(k["out_dir"], "agents",
                                           f"agent_d{d:.2f}_s{s}_{g}.pt"), ag, nm)
```

(`import os` is already at module top; `agents` dict holds untrained/predictor
before survival is added, so place this AFTER the survival line.)

3e. `evaluate_agent` (~line 182) - full replacement:

```python
def evaluate_agent(agent, norm, drift, a, dev, seed, agent_name=""):
    dump_path = tdump = cdump = None
    if getattr(a, "dump_states", None):
        stem = os.path.join(a.dump_states, f"states_d{drift:.2f}_s{seed}_{agent_name}")
        dump_path = stem + ".npz"
        tdump, cdump = stem + "_h7transfer.npz", stem + "_cg.npz"
    heldout = getattr(a, "heldout_evals", False)
    pr = pooled_readout(agent, norm, P, drift, n_eps=a.pool_n, steps=a.pool_steps,
                        ray_steps=a.ray_steps, device=dev, seed=seed, dump_path=dump_path,
                        return_pools=heldout)
    pool, pools = (pr if heldout else (pr, None))
    mp = readout(agent, norm, P, drift, n_pairs=a.mp_pairs, prefix_steps=a.mp_prefix,
                 branch_steps=a.mp_branch, ray_steps=a.ray_steps, device=dev, seed=seed)
    ho = None
    if heldout:
        from itasorl.experiment_b2 import cg_probe, common_garden_rollout, transfer_readout
        Ha, Hs = pools
        ho = transfer_readout(agent, norm, P, drift, Ha, Hs, n_eps=a.pool_n,
                              steps=a.pool_steps, ray_steps=a.ray_steps, device=dev,
                              seed=seed, dump_path=tdump)
        at, st = common_garden_rollout(agent, norm, P, drift, n_pairs=a.pool_n,
                                       prefix_steps=a.cg_prefix, tail_steps=a.cg_steps,
                                       ray_steps=a.ray_steps, device=dev)
        ho.update(cg_probe(at, st, seed=seed))
        if cdump:
            np.savez_compressed(cdump, auth=np.stack(at) if at else np.zeros((0, a.cg_steps, 1)),
                                surr=np.stack(st) if st else np.zeros((0, a.cg_steps, 1)))
    return pool, mp, ho
```

3f. `run_cell` loop over arms (~line 230):

```python
    for g in AG:
        pool, mp, ho = evaluate_agent(agents[g][0], agents[g][1], d, a_ns, dev, s, g)
        out["agents"][g] = {"pool": pool, "mp": mp}
        if ho is not None:
            out["agents"][g]["heldout"] = ho
```

3g. `record_cell` - append at the end of the per-arm loop (schema no-op when off):

```python
        ho = cell["agents"][g].get("heldout")
        if ho is not None:
            for kk in ("transfer_target", "cg_tail_target", "cg_latetail_target", "cg_n_pairs"):
                res[d][g].setdefault(kk, []).append(ho.get(kk))
```

(`fresh_results` is deliberately NOT modified: aggregate JSON keys appear only
when heldout cells exist.)

3h. `print_cell` - append inside the per-arm loop:

```python
        ho = cell["agents"][g].get("heldout")
        if ho is not None:
            print(f"   {'':10s} heldout: transfer={ho['transfer_target']:.3f} "
                  f"[{ho.get('transfer_lo', float('nan')):.3f},{ho.get('transfer_hi', float('nan')):.3f}] "
                  f"(n={ho['transfer_n_auth']}+{ho['transfer_n_surr']})  "
                  f"cg_tail={ho['cg_tail_target']:.3f} "
                  f"[{ho['cg_tail_lo']:.3f},{ho['cg_tail_hi']:.3f}]  "
                  f"cg_late={ho['cg_latetail_target']:.3f} (pairs={ho['cg_n_pairs']})", flush=True)
```

3i. Summary - after the volatility check block (~line 521), still inside `main()`:

```python
    if res[dmax]["survival"].get("transfer_target"):
        print("\nHeld-out channels at strongest drift (frozen rules, spec 2026-07-14):")
        for g in AG:
            tt = np.nanmean(np.array(res[dmax][g]["transfer_target"], float))
            ct = np.nanmean(np.array(res[dmax][g]["cg_tail_target"], float))
            cl = np.nanmean(np.array(res[dmax][g]["cg_latetail_target"], float))
            print(f"  {g:10s} transfer={tt:.3f}  cg_tail={ct:.3f}  cg_late={cl:.3f}")
        s_tt = np.nanmean(np.array(res[dmax]["survival"]["transfer_target"], float))
        u_tt = np.nanmean(np.array(res[dmax]["untrained"]["transfer_target"], float))
        s_cg = np.nanmean(np.array(res[dmax]["survival"]["cg_tail_target"], float))
        u_cg = np.nanmean(np.array(res[dmax]["untrained"]["cg_tail_target"], float))
        t_ok = s_tt >= 0.65 and s_tt >= u_tt + 0.05
        c_ok = s_cg >= 0.65 and s_cg >= u_cg + 0.05
        print(f"  transfer rule (surv>=0.65 AND >untrained+0.05): "
              f"{'GENERALIZES beyond the trained fingerprint' if t_ok else 'fingerprint-specific (informative negative)'}")
        print(f"  common-garden rule (surv>=0.65 AND >untrained+0.05): "
              f"{'PERSISTENT world identity' if c_ok else 'reactive tracking (informative negative)'}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_heldout_runner.py -q`
Expected: `2 passed`

- [ ] **Step 5: Full suite + ruff**

Run: `python -m pytest -q && ruff check .`
Expected: all pass. Watch F541 (no placeholder-less f-strings in the new prints).

- [ ] **Step 6: Commit**

```bash
git add scripts/run_expB2.py tests/test_heldout_runner.py
git commit -m "feat(heldout): --heldout-evals/--save-agents plumbing, fingerprint-safe for old resumes"
```

---

### Task 6: End-to-end smoke on CUDA + docs

**Files:**
- Modify: `scripts/README.md` (the "L3 owed runs" block, after the h7 audit command ~line 85)

- [ ] **Step 1: Run the quick smoke with both flags on**

```bash
python scripts/run_expB2.py --quick --drift-mode l3 --l3-hidden 2 \
    --heldout-evals --heldout-hidden 3 --save-agents --device cuda \
    --out-dir results/quick_heldout --dump-states results/quick_heldout/states
```

Expected (numbers are noise at quick scale; check MECHANICS only):
- startup prints the "heldout evals ON" line and trains TWO G nets
- every cell prints the `heldout: transfer=... cg_tail=...` line with finite values
- `results/quick_heldout/agents/` contains `agent_d*_s*_{untrained,predictor,survival}.pt` (12 files)
- `results/quick_heldout/states/` contains `*_h7transfer.npz` and `*_cg.npz` beside the standard dumps
- the summary prints the two frozen-rule verdict lines
- exit code 0

- [ ] **Step 2: Verify the no-op path is untouched**

```bash
python - <<'EOF'
import json
c = json.load(open('fullruns/l3_h7_traces/cells/cell_d0.45_s0.json'))
assert set(c['cell']['agents']['survival']) == {'pool', 'mp'}, 'old cell schema'
print('old cell schema unchanged:', sorted(c['cell']['agents']['survival']))
EOF
```

Expected: prints `old cell schema unchanged: ['mp', 'pool']` (confirms the reader
expectations; the fingerprint no-op test in Task 5 already covers resume safety).

- [ ] **Step 3: Update scripts/README.md**

After the h7 audit command block (~line 85), add:

```
# OWED - held-out fingerprint + common-garden run (last open L3 item; spec
# docs/superpowers/specs/2026-07-14-l3-heldout-common-garden-probe-design.md).
# Trains at hidden=8, holds out hidden=7; adds transfer + common-garden
# channels and persists trained agents.
python scripts/run_expB2.py --drift-mode l3 --l3-hidden 8 \
    --heldout-evals --heldout-hidden 7 --save-agents \
    --seeds 0 1 2 3 4 5 6 7 8 9 --device cuda \
    --out-dir fullruns/l3_h8_heldout \
    --dump-states fullruns/l3_h8_heldout/states
```

- [ ] **Step 4: Final full verification**

Run: `python -m pytest -q && ruff check .`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add scripts/README.md
git commit -m "docs: heldout/common-garden run command (owed L3 item)"
```

---

## Not in this plan (deliberate)

- Launching the overnight run: human-launched or on explicit ask, per the
  standing run-workflow preference (RAM preflight, background + monitor).
- Recording results in PREREGISTRATION_L3.md sec.12: happens after the run,
  applying the frozen decision rules from the spec.
- The reverse transfer direction (train h7, hold out h8): staged follow-up only
  if the transfer result is positive.

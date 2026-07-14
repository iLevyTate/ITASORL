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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

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

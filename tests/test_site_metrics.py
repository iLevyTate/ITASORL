"""Contract pins for scripts/site_metrics.py (the single source of the site's numbers).

derive_metrics() reads the committed artifacts/expB2/ tree and returns the display
strings the static site quotes. The whole point of the module is that these EQUAL the
hand-verified headline numbers, so future runs regenerate the page instead of drifting
against string pins. Reads real committed artifacts (small, deterministic); no mocks."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

site_metrics = importlib.import_module("site_metrics")


def test_derive_metrics_matches_findings_headline_numbers():
    m = site_metrics.derive_metrics(ROOT / "artifacts" / "expB2")
    # FINDINGS sec.10.2 L3 survival headline + hero (2dp) rounding of the same source.
    assert m["l3_survival"] == "0.752"
    assert m["l3_survival_hero"] == "0.75"
    # Decision interval: t-based 90% CI recomputed from the per-seed survival cells.
    assert m["l3_ci_lo"] == "0.698"
    assert m["l3_ci_hi"] == "0.807"
    # sec.10.6-10.7 transfer + re-scored common-garden (the gate-pinned five).
    assert m["transfer_same"] == "0.773"
    assert m["transfer_reverse"] == "0.638"
    assert m["cg_forward"] == "0.666"
    assert m["cg_reverse"] == "0.684"


def test_derive_metrics_values_are_display_strings():
    m = site_metrics.derive_metrics(ROOT / "artifacts" / "expB2")
    assert all(isinstance(v, str) for v in m.values())

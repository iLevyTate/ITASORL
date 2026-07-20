"""Contract pins for scripts/build_index.py (renders index.html from a template + metrics).

The generator's job is single-source regeneration: every headline number on the page comes
from `{{key}}` placeholders filled by site_metrics.derive_metrics(), so a re-run that moves a
number moves the page. The load-bearing contracts:
  - render() fills known placeholders and REFUSES unknown ones (a typo'd anchor must fail loud);
  - the committed index.html is exactly render(index.template.html, derived) - i.e. idempotent,
    so the recheck gate can regenerate-and-diff instead of pinning bare strings;
  - every derived metric is actually used by the template (no dead single-source keys)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

build_index = importlib.import_module("build_index")
site_metrics = importlib.import_module("site_metrics")


def test_render_substitutes_known_placeholders():
    out = build_index.render("survival {{l3_survival}} CI {{l3_ci_lo}}-{{l3_ci_hi}}",
                             {"l3_survival": "0.752", "l3_ci_lo": "0.698", "l3_ci_hi": "0.807"})
    assert out == "survival 0.752 CI 0.698-0.807"
    assert "{{" not in out


def test_render_raises_on_unknown_placeholder():
    with pytest.raises(KeyError):
        build_index.render("mystery {{not_a_metric}}", {"l3_survival": "0.752"})


def test_committed_index_is_regenerable_and_check_passes():
    # The committed page must equal a fresh render; --check must agree (idempotent).
    template = (ROOT / "index.template.html").read_text(encoding="utf-8")
    metrics = site_metrics.derive_metrics(ROOT / "artifacts" / "expB2")
    rendered = build_index.render(template, metrics)
    assert rendered == (ROOT / "index.html").read_text(encoding="utf-8")
    assert build_index.build(ROOT, check=True) == 0


def test_every_metric_is_used_by_the_template():
    template = (ROOT / "index.template.html").read_text(encoding="utf-8")
    metrics = site_metrics.derive_metrics(ROOT / "artifacts" / "expB2")
    for key in metrics:
        assert "{{" + key + "}}" in template, f"metric '{key}' is derived but never placed"

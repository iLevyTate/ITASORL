"""Regenerate index.html from index.template.html + the single-source metrics.

The template is the hand-maintained page with every headline number replaced by a
`{{key}}` placeholder; `site_metrics.derive_metrics()` supplies the values from
artifacts/expB2/. Placeholders are plain text substitution, so they are valid in every
context the numbers live in (HTML text, JS string literals, JS numeric literals, and
`data-target="..."` attributes) - which an HTML-comment anchor could not be.

Usage:
  python scripts/build_index.py           # write index.html from the template
  python scripts/build_index.py --check    # exit 1 if index.html is stale (CI/gate use)

The forward cross-recipe number (transfer_rff_target ~0.684) is intentionally a template
literal, not a placeholder: its run lives in an untracked fullruns/ dir, so it has no
committed artifact to regenerate from. It collides with the (owned) reverse-cg 0.684, so
the string stays covered; promoting the cross-recipe run into artifacts/expB2/ would let a
future revision own it too.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from site_metrics import derive_metrics  # noqa: E402

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


def render(template: str, metrics: dict[str, str]) -> str:
    """Fill `{{key}}` placeholders from `metrics`; raise KeyError on any unknown key."""
    def _sub(m: re.Match) -> str:
        key = m.group(1)
        if key not in metrics:
            raise KeyError(f"template placeholder {{{{{key}}}}} has no metric")
        return metrics[key]
    return _PLACEHOLDER.sub(_sub, template)


def build(root: str | Path = ROOT, check: bool = False) -> int:
    root = Path(root)
    template = (root / "index.template.html").read_text(encoding="utf-8")
    rendered = render(template, derive_metrics(root / "artifacts" / "expB2"))
    target = root / "index.html"
    if check:
        current = target.read_text(encoding="utf-8") if target.exists() else ""
        if current != rendered:
            print("index.html is STALE - run: python scripts/build_index.py")
            return 1
        print("index.html is up to date with artifacts + template.")
        return 0
    target.write_text(rendered, encoding="utf-8", newline="")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify index.html is current; do not write")
    args = ap.parse_args()
    raise SystemExit(build(check=args.check))

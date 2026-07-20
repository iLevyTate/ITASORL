"""Single source of the static site's headline numbers.

`derive_metrics(artifacts_dir)` reads the committed `artifacts/expB2/` tree and returns
the display strings `index.html` quotes. `scripts/build_index.py` injects them into
anchored `<!--metric:key-->...<!--/metric-->` spans, and the recheck gate regenerates
and diffs instead of pinning bare strings. So the numbers on the page trace to an
artifact by construction; a re-run that moves a headline moves the page in the same step.

Scope: the exact, gate-relevant L3 numbers (survival, its t-based decision CI, held-out
transfer both directions, re-scored common-garden both directions). Soft "~" display
values on the page (oracle ~0.99, agent probe ~0.50, ~0.73 behavior-independent) are
intentional loose rounds and stay as static prose - mechanical formatting would fight
them, and the gate never pinned them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from itasorl.stats import t_ci90  # noqa: E402


def _load(artifacts_dir: Path, name: str) -> dict:
    with open(Path(artifacts_dir) / name, encoding="utf-8") as fh:
        return json.load(fh)


def _survival_per_seed(summary: dict, drift: str = "0.45") -> list[float]:
    """Per-seed pooled survival AUROC for the survival arm at the strong drift - the
    cells the t-based decision interval is computed over (not a stored bootstrap band)."""
    rows = [c for c in summary["cells"] if c["drift"] == drift and c["agent"] == "survival"]
    return [float(c["pool_target"]) for c in sorted(rows, key=lambda c: c["seed"])]


def derive_metrics(artifacts_dir: str | Path) -> dict[str, str]:
    artifacts_dir = Path(artifacts_dir)
    h8 = _load(artifacts_dir, "heldout_l3_h8_summary.json")
    rev = _load(artifacts_dir, "heldout_l3_h7_reverse_summary.json")
    cg_fwd = _load(artifacts_dir, "heldout_l3_h8_cg_rescore.json")
    cg_rev = _load(artifacts_dir, "heldout_l3_h7_reverse_cg_rescore.json")

    survival = h8["aggregate"]["d=0.45 survival"]["pool_target"]["mean"]
    ci_lo, ci_hi = t_ci90(_survival_per_seed(h8))

    return {
        "l3_survival": f"{survival:.3f}",
        "l3_survival_hero": f"{survival:.2f}",
        "l3_ci_lo": f"{ci_lo:.3f}",
        "l3_ci_hi": f"{ci_hi:.3f}",
        "transfer_same": f"{h8['aggregate']['d=0.45 survival']['transfer_target']['mean']:.3f}",
        "transfer_reverse": f"{rev['aggregate']['d=0.45 survival']['transfer_target']['mean']:.3f}",
        "cg_forward": f"{cg_fwd['strong_drift']['survival']['cg_tail_mean']:.3f}",
        "cg_reverse": f"{cg_rev['strong_drift']['survival']['cg_tail_mean']:.3f}",
    }


if __name__ == "__main__":
    for k, v in derive_metrics(ROOT / "artifacts" / "expB2").items():
        print(f"{k:20s} {v}")

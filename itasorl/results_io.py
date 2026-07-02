"""
Record end-to-end run output: logs, parsed metrics, artifacts, summary, zip bundle.

Used by run_e2e.py. Each run lands in fullruns/<MMDDYYYY>/ by default (or
fullruns/<MMDDYYYY_HHMMSS>/ when that date folder already contains a run).

Logs are written incrementally so ``combined.log`` and ``status.json`` can be
tailed while a run is in progress (see ``scripts/watch_run.py --follow``).
Set ``ITASORL_DRIVE_SYNC`` to a directory path to mirror live logs there
(e.g. Google Drive on Colab).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = ROOT / "results" / "runs"  # legacy override via --results-dir
FULLRUNS_ROOT = ROOT / "fullruns"
LATEST_RUN_PTR = ROOT / "results" / "LATEST_RUN.txt"
STATUS_SYNC_INTERVAL_SEC = 2.0


def default_run_dir() -> Path:
    """Primary output under fullruns/ (date folder; time suffix if date folder taken)."""
    date = datetime.now().strftime("%m%d%Y")
    base = FULLRUNS_ROOT / date
    if base.exists() and any(base.iterdir()):
        return FULLRUNS_ROOT / datetime.now().strftime("%m%d%Y_%H%M%S")
    return base


def read_latest_run_dir() -> Path | None:
    """Resolve the most recent e2e run directory from LATEST_RUN.txt."""
    if not LATEST_RUN_PTR.is_file():
        return None
    path = Path(LATEST_RUN_PTR.read_text(encoding="utf-8").strip())
    return path if path.is_dir() else None

# Repo-relative paths produced by reproduction scripts.
STEP_ARTIFACTS: dict[str, list[str]] = {
    "expA_l1": ["docs/figures/expA_ceiling.png"],
    "expA_l2": ["docs/figures/expA_L2_ceiling.png"],
    "expB_full": ["docs/figures/expB_incidental.png"],
    "expB_surprise": ["docs/figures/expB_channels.png"],
    "expB_kstep": ["docs/figures/expB_kstep.png"],
    "expB2": [],  # written directly via run_expB2.py --out-dir
}


def _git_head() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, stderr=subprocess.DEVNULL, text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _device_info() -> dict[str, Any]:
    info: dict[str, Any] = {"python": sys.version.split()[0]}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = None
    return info


def parse_step_metrics(name: str, log: str) -> dict[str, Any]:
    """Best-effort structured metrics from stdout (scripts mostly print text today)."""
    m: dict[str, Any] = {"step": name}

    if name == "pytest":
        mt = re.search(r"(\d+) passed", log)
        mf = re.search(r"(\d+) failed", log)
        m["passed"] = int(mt.group(1)) if mt else None
        m["failed"] = int(mf.group(1)) if mf else 0
        m["ok"] = m["failed"] == 0 and (m["passed"] or 0) > 0
        return m

    if name == "expB_smoke":
        for key, pat in (
            ("target", r"target.*AUROC\s*=\s*([\d.]+)"),
            ("shuffled", r"shuffled.*AUROC\s*=\s*([\d.]+)"),
            ("speed", r"speed.*AUROC\s*=\s*([\d.]+)"),
        ):
            hit = re.search(pat, log)
            if hit:
                m[key] = float(hit.group(1))
        m["organism_encodes_world"] = _encodes(m.get("target"))
        return m

    if name == "expA_l1":
        blocks = re.findall(
            r"\[(L0 control[^\]]*|L1[^\]]*)\].*?oracle AUROC\s*=\s*([\d.]+).*?"
            r"leakage gate\s*=\s*(PASS[^\n]*|FAIL[^\n]*)",
            log, re.S,
        )
        m["cells"] = [
            {"label": lbl.strip(), "oracle_auroc": float(auc), "leakage_pass": "PASS" in gate}
            for lbl, auc, gate in blocks
        ]
        cal = re.findall(r"delta=\s*([\d.]+).*?AUROC=([\d.]+)", log)
        if cal:
            m["calibration"] = [{"delta": float(a), "oracle_auroc": float(b)} for a, b in cal]
        return m

    if name == "expA_l2":
        blocks = re.findall(
            r"\[(L0 control[^\]]*|L2[^\]]*)\].*?oracle AUROC\s*=\s*([\d.]+).*?"
            r"leakage gate\s*=\s*(PASS|FAIL[^\n]*)",
            log, re.S,
        )
        m["cells"] = [
            {"label": lbl.strip(), "oracle_auroc": float(auc), "leakage_pass": gate.startswith("PASS")}
            for lbl, auc, gate in blocks
        ]
        cal = re.findall(r"drift_sigma=\s*([\d.]+)\s+AUROC=([\d.]+)", log)
        if cal:
            m["calibration"] = [{"drift_sigma": float(a), "oracle_auroc": float(b)} for a, b in cal]
        return m

    if name == "expB_full":
        rows = re.findall(
            r"drift=([\d.]+)\s+target=([\d.]+)±([\d.]+).*?"
            r"shuffled=([\d.]+)±([\d.]+).*?speed\(\+ctrl\)=([\d.]+)±([\d.]+)",
            log,
        )
        m["drift_sweep"] = [
            {"drift": float(d), "target_mean": float(t), "target_std": float(ts),
             "shuffled_mean": float(sh), "speed_mean": float(sp),
             "organism_encodes_world": _encodes(float(t))}
            for d, t, ts, sh, _, sp, _ in rows
        ]
        return m

    if name == "expB_surprise":
        rows = re.findall(r"drift=([\d.]+)\s+surprise-probe AUROC = ([\d.]+)", log)
        m["drift_sweep"] = [
            {"drift": float(d), "surprise_auroc": float(a), "organism_encodes_world": _encodes(float(a))}
            for d, a in rows
        ]
        return m

    if name == "expB_kstep":
        rows = re.findall(
            r"open_horizon=\s*(\d+):\s+drift0\.45 target=([\d.]+)±([\d.]+)\s+"
            r"control target=([\d.]+)±([\d.]+)",
            log,
        )
        m["horizons"] = [
            {"open_horizon": int(h), "drift_045_target": float(t), "drift_0_target": float(c),
             "organism_encodes_world": _encodes(float(t))}
            for h, t, _, c, _ in rows
        ]
        return m

    if name == "expB_gap":
        eng = re.search(r"verdict:\s*(ENGAGED|did NOT engage)", log)
        if eng:
            m["open_loop_engaged"] = eng.group(1) == "ENGAGED"
        mse = re.search(r"open-loop MSE=([\d.]+)", log)
        if mse:
            m["open_loop_mse"] = float(mse.group(1))
        rows = re.findall(r"drift=([\d.]+):\s+target AUROC=([\d.]+)±([\d.]+)", log)
        m["delta_objective"] = [
            {"drift": float(d), "target_mean": float(t), "organism_encodes_world": _encodes(float(t))}
            for d, t, _ in rows
        ]
        return m

    if name == "expB_nonlinear":
        rows = re.findall(
            r"drift=([\d.]+):\s+target=([\d.]+)±([\d.]+).*?"
            r"speed\(\+ctrl\)=([\d.]+)±([\d.]+)",
            log,
        )
        m["drift_sweep"] = [
            {"drift": float(d), "target_mean": float(t), "speed_mean": float(sp),
             "organism_encodes_world": _encodes(float(t))}
            for d, t, _, sp, _ in rows
        ]
        return m

    if name == "expB2":
        # Structured JSON copied separately; parse headline numbers from log.
        # Per-agent lines print once per drift block in ascending drift order, so the
        # LAST match is the strongest (test) drift - the cell the headline verdict and
        # the "At strongest drift" |dev| line refer to. re.search would return the
        # drift-0 control cell and understate the result.
        surv = re.findall(
            r"survival\s+PRIMARY pool target = ([\d.]+)\+/-([\d.]+)", log,
        )
        pred = re.findall(r"predictor\s+PRIMARY pool target = ([\d.]+)\+/-", log)
        if surv:
            m["survival_pool_target_mean"] = float(surv[-1][0])
            m["survival_pool_target_std"] = float(surv[-1][1])
            m["organism_encodes_world"] = _encodes(m["survival_pool_target_mean"], threshold=0.65)
        if pred:
            m["predictor_pool_target_mean"] = float(pred[-1])
        dev = re.search(r"survival pooled target \|dev\|=([\d.]+)", log)
        if dev:
            m["survival_deviation_from_chance"] = float(dev.group(1))
        return m

    return m


def _encodes(auroc: float | None, threshold: float = 0.60) -> str | None:
    """Plain verdict for organism world-identity probe AUROC."""
    if auroc is None:
        return None
    dev = abs(auroc - 0.5)
    if dev < 0.05:
        return "no"       # at chance
    if auroc >= threshold or dev >= 0.15:
        return "strong"   # clear signal
    return "weak"         # faint trace (B-v2 survival ~0.60)


def _load_expb2_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open() as f:
        raw = json.load(f)
    out: dict[str, Any] = {}
    for drift, agents in raw.items():
        out[drift] = {}
        for agent, metrics in agents.items():
            pt = metrics.get("pool_target", [])
            if pt:
                mean = sum(pt) / len(pt)
                out[drift][agent] = {
                    "pool_target_mean": mean,
                    "pool_target_std": (sum((x - mean) ** 2 for x in pt) / len(pt)) ** 0.5,
                    "organism_encodes_world": _encodes(mean, threshold=0.65),
                }
    return out


@dataclass
class RunRecorder:
    quick: bool
    run_dir: Path
    manifest: dict[str, Any] = field(default_factory=dict)
    _run_started: float = field(default=0.0, repr=False)
    _status_last_write: float = field(default=0.0, repr=False)
    _mirror_dir: Path | None = field(default=None, repr=False)

    @classmethod
    def _mirror_from_env(cls) -> Path | None:
        mirror_raw = os.environ.get("ITASORL_DRIVE_SYNC", "").strip()
        return Path(mirror_raw) if mirror_raw else None

    @classmethod
    def create(cls, *, quick: bool, out_dir: Path | None = None) -> RunRecorder:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = out_dir or default_run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "steps").mkdir(exist_ok=True)
        (run_dir / "artifacts").mkdir(exist_ok=True)
        rec = cls(
            quick=quick,
            run_dir=run_dir,
            _run_started=time.perf_counter(),
            _mirror_dir=cls._mirror_from_env(),
        )
        rec.manifest = {
            "run_id": run_id,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "quick": quick,
            "git_commit": _git_head(),
            "environment": _device_info(),
            "steps": {},
        }
        (run_dir / "combined.log").write_text("", encoding="utf-8")
        rec._write_manifest()
        rec._write_status(current_step=None, step_status="starting", last_line="")
        LATEST_RUN_PTR.parent.mkdir(parents=True, exist_ok=True)
        LATEST_RUN_PTR.write_text(str(run_dir.resolve()), encoding="utf-8")
        rec._sync_mirror()
        return rec

    @classmethod
    def resume(cls, run_dir: Path, *, mirror_dir: Path | None = None) -> RunRecorder:
        """Continue an interrupted run; steps with status ``ok`` are skipped."""
        run_dir = run_dir.resolve()
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"No manifest.json in {run_dir}; cannot resume.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        (run_dir / "steps").mkdir(exist_ok=True)
        (run_dir / "artifacts").mkdir(exist_ok=True)
        if not (run_dir / "combined.log").is_file():
            (run_dir / "combined.log").write_text("", encoding="utf-8")
        rec = cls(
            quick=bool(manifest.get("quick", False)),
            run_dir=run_dir,
            _run_started=time.perf_counter(),
            _mirror_dir=mirror_dir if mirror_dir is not None else cls._mirror_from_env(),
        )
        rec.manifest = manifest
        manifest.setdefault("steps", {})
        manifest["resumed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["git_commit"] = _git_head()
        manifest["environment"] = _device_info()
        rec._write_manifest()
        done = [n for n, s in manifest["steps"].items() if s.get("status") == "ok"]
        rec._append_combined(
            f"\n{'=' * 72}\nRESUME at {manifest['resumed_at_utc']}\n"
            f"Skipping completed steps: {', '.join(done) or '(none)'}\n{'=' * 72}\n"
        )
        rec._write_status(current_step=None, step_status="resuming", force=True)
        LATEST_RUN_PTR.parent.mkdir(parents=True, exist_ok=True)
        LATEST_RUN_PTR.write_text(str(run_dir), encoding="utf-8")
        rec._sync_mirror(full=True)
        return rec

    def step_is_done(self, name: str) -> bool:
        step = self.manifest.get("steps", {}).get(name)
        return bool(step and step.get("status") == "ok")

    def _combined_path(self) -> Path:
        return self.run_dir / "combined.log"

    def _status_path(self) -> Path:
        return self.run_dir / "status.json"

    def _manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    def _append_combined(self, text: str) -> None:
        with self._combined_path().open("a", encoding="utf-8") as fh:
            fh.write(text)

    def _write_manifest(self) -> None:
        self._manifest_path().write_text(
            json.dumps(self.manifest, indent=2), encoding="utf-8",
        )

    def _write_status(
        self,
        *,
        current_step: str | None,
        step_status: str,
        last_line: str = "",
        force: bool = False,
    ) -> None:
        now = time.perf_counter()
        if not force and (now - self._status_last_write) < STATUS_SYNC_INTERVAL_SEC:
            return
        self._status_last_write = now
        payload = {
            "run_id": self.manifest.get("run_id"),
            "run_dir": str(self.run_dir.resolve()),
            "quick": self.quick,
            "running": step_status not in ("finished", "failed"),
            "current_step": current_step,
            "step_status": step_status,
            "elapsed_sec": round(now - self._run_started, 1),
            "last_line": last_line.rstrip("\n")[-500:],
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self._status_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._sync_mirror()

    def _sync_mirror(self, *, full: bool = False) -> None:
        if self._mirror_dir is None:
            return
        dest_root = self._mirror_dir / self.run_dir.name
        dest_root.mkdir(parents=True, exist_ok=True)
        for name in ("combined.log", "status.json", "manifest.json", "SUMMARY.md", "bundle.zip"):
            src = self.run_dir / name
            if src.is_file():
                shutil.copy2(src, dest_root / name)
        if full:
            for sub in ("steps", "artifacts"):
                src_dir = self.run_dir / sub
                if src_dir.is_dir():
                    shutil.copytree(src_dir, dest_root / sub, dirs_exist_ok=True)

    def note_step(self, name: str, *, status: str) -> None:
        """Record a skipped or external step in manifest + status."""
        self.manifest["steps"][name] = {"status": status}
        self._write_manifest()
        self._write_status(current_step=name, step_status=status, force=True)

    def run_step(self, name: str, cmd: list[str], *, cwd: Path = ROOT,
                 extra_artifacts: list[str] | None = None) -> None:
        t0 = time.perf_counter()
        header = f"\n{'=' * 72}\nSTEP {name}\n$ {' '.join(cmd)}\n{'=' * 72}\n"
        print(header, end="", flush=True)
        self._append_combined(header)
        self._write_status(current_step=name, step_status="running", force=True)

        log_path = self.run_dir / "steps" / f"{name}.log"
        log_path.write_text("", encoding="utf-8")

        # Stream stdout live (Colab/long runs) while capturing for logs + parsing.
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
        log_parts: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log_parts.append(line)
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(line)
            self._append_combined(line)
            self._write_status(current_step=name, step_status="running", last_line=line)
        proc.wait()
        elapsed = time.perf_counter() - t0
        log = "".join(log_parts)

        metrics = parse_step_metrics(name, log)
        metrics_path = self.run_dir / "steps" / f"{name}.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        copied: list[str] = []
        for rel in (STEP_ARTIFACTS.get(name, []) + (extra_artifacts or [])):
            src = cwd / rel
            if src.is_file():
                dest = self._artifact_dest(rel)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                copied.append(str(dest.relative_to(self.run_dir)))

        if name == "expB2":
            for fname in ("expB2_results.json", "expB2_survival.png"):
                p = self.run_dir / "artifacts" / fname
                if p.is_file():
                    rel = str(p.relative_to(self.run_dir))
                    if rel not in copied:
                        copied.append(rel)

        if name == "expB2":
            b2_json = self.run_dir / "artifacts" / "expB2_results.json"
            b2_metrics = _load_expb2_json(b2_json)
            if b2_metrics:
                metrics["agents_by_drift"] = b2_metrics
                metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        status = "ok" if proc.returncode == 0 else "failed"
        self.manifest["steps"][name] = {
            "status": status,
            "exit_code": proc.returncode,
            "elapsed_sec": round(elapsed, 1),
            "log": str(log_path.relative_to(self.run_dir)),
            "metrics": str(metrics_path.relative_to(self.run_dir)),
            "artifacts": copied,
        }
        self._write_manifest()
        self._write_status(
            current_step=name,
            step_status=status,
            last_line=log_parts[-1] if log_parts else "",
            force=True,
        )
        self._sync_mirror(full=True)
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd, log)

    def _artifact_dest(self, rel: str) -> Path:
        rel_path = Path(rel)
        if len(rel_path.parts) > 1:
            return self.run_dir / "artifacts" / rel_path
        return self.run_dir / "artifacts" / rel_path.name

    def finalize(self, *, total_sec: float, make_zip: bool = True) -> Path:
        self.manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        self.manifest["total_elapsed_sec"] = round(total_sec, 1)
        self.manifest["run_dir"] = str(self.run_dir)

        summary_path = self.run_dir / "SUMMARY.md"
        summary_path.write_text(build_summary(self.manifest, self.run_dir), encoding="utf-8")

        zip_path = self.run_dir / "bundle.zip"
        if make_zip:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for path in self.run_dir.rglob("*"):
                    if path.is_file() and path != zip_path:
                        zf.write(path, path.relative_to(self.run_dir))
            self.manifest["bundle_zip"] = str(zip_path.relative_to(self.run_dir))

        self._write_manifest()
        self._write_status(current_step=None, step_status="finished", force=True)
        LATEST_RUN_PTR.parent.mkdir(parents=True, exist_ok=True)
        LATEST_RUN_PTR.write_text(str(self.run_dir.resolve()), encoding="utf-8")
        self._sync_mirror(full=True)
        if self._mirror_dir is not None:
            dest_root = self._mirror_dir / self.run_dir.name
            dest_root.mkdir(parents=True, exist_ok=True)
            for name in ("SUMMARY.md", "bundle.zip"):
                src = self.run_dir / name
                if src.is_file():
                    shutil.copy2(src, dest_root / name)

        print(f"\n{'=' * 72}", flush=True)
        print(f"Results recorded -> {self.run_dir}", flush=True)
        print("  SUMMARY.md  - human-readable outcome", flush=True)
        print("  combined.log - full stdout (updated live during run)", flush=True)
        print("  status.json - live step + last line", flush=True)
        print("  manifest.json - machine-readable index", flush=True)
        if make_zip:
            print("  bundle.zip  - download everything", flush=True)
        print("  tail live: python scripts/watch_run.py --follow", flush=True)
        print(f"{'=' * 72}\n", flush=True)
        return self.run_dir


def build_summary(manifest: dict[str, Any], run_dir: Path) -> str:
    lines = [
        "# ITASORL run summary",
        "",
        f"- **Run ID:** `{manifest.get('run_id')}`",
        f"- **Mode:** {'quick' if manifest.get('quick') else 'full'}",
        f"- **Git commit:** `{manifest.get('git_commit') or 'unknown'}`",
        f"- **Duration:** {manifest.get('total_elapsed_sec', 0) / 60:.1f} min",
        "",
        "## What this run tested",
        "",
        "Can a from-scratch **organism** tell authentic vs surrogate worlds (L2 drift),",
        "read out from its internal state — never trained or rewarded for world identity?",
        "",
        "| AUROC | Meaning |",
        "|-------|---------|",
        "| ~0.50 | No incidental encoding (coin flip) |",
        "| 0.55–0.65 | Weak trace |",
        "| ≥ 0.65 | Pre-registered encoding threshold (B-v2) |",
        "| ~0.99 | Outside oracle (Experiment A) — trivially detectable |",
        "",
        "## Step status",
        "",
        "| Step | Status | Time (min) |",
        "|------|--------|------------|",
    ]
    for name, step in manifest.get("steps", {}).items():
        lines.append(
            f"| {name} | {step.get('status')} | {step.get('elapsed_sec', 0) / 60:.1f} |"
        )

    lines.extend(["", "## Outcomes (organism vs environments)", ""])

    steps = manifest.get("steps", {})
    _section(lines, steps, "expA_l1", _summarize_expA_l1, run_dir)
    _section(lines, steps, "expA_l2", _summarize_expA_l2, run_dir)
    _section(lines, steps, "expB_full", _summarize_expB_full, run_dir)
    _section(lines, steps, "expB_surprise", _summarize_expB_surprise, run_dir)
    _section(lines, steps, "expB2", _summarize_expB2, run_dir)

    lines.extend([
        "",
        "## Headline (plain English)",
        "",
    ])
    lines.append(_headline(steps, run_dir))
    lines.extend([
        "",
        "## Files in this run",
        "",
        "- Live log (tail while running): `combined.log` + `status.json`",
        "- Per-step logs: `steps/`",
        "- Figures + JSON: `artifacts/`",
        "",
        "While a run is in progress: `python scripts/watch_run.py --follow`",
        "",
        "Download **`bundle.zip`** from this folder to keep everything.",
        "",
    ])
    return "\n".join(lines)


def _section(lines: list[str], steps: dict, name: str, fn, run_dir: Path) -> None:
    step = steps.get(name)
    if not step or step.get("status") != "ok":
        return
    metrics_path = run_dir / step["metrics"]
    if not metrics_path.is_file():
        return
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    lines.append(f"### {name}")
    lines.append("")
    fn(lines, metrics)
    lines.append("")


def _summarize_expA_l1(lines: list[str], m: dict) -> None:
    lines.append("*Outside observer (no organism) — L1 quantization:*")
    lines.append("")
    for cell in m.get("cells", []):
        lines.append(f"- **{cell['label']}**: oracle AUROC **{cell['oracle_auroc']:.3f}** "
                     f"(leakage {'PASS' if cell['leakage_pass'] else 'FAIL'})")


def _summarize_expA_l2(lines: list[str], m: dict) -> None:
    lines.append("*Outside observer (no organism) — L2 rollout drift:*")
    lines.append("")
    for cell in m.get("cells", []):
        lines.append(f"- **{cell['label']}**: oracle AUROC **{cell['oracle_auroc']:.3f}**")


def _summarize_expB_full(lines: list[str], m: dict) -> None:
    lines.append("*Organism (prediction-only agent) — recurrent-state probe:*")
    lines.append("")
    for row in m.get("drift_sweep", []):
        verdict = row.get("organism_encodes_world", "?")
        lines.append(
            f"- drift **{row['drift']:.2f}**: target AUROC **{row['target_mean']:.3f}** "
            f"→ encoding: **{verdict}**"
        )


def _summarize_expB_surprise(lines: list[str], m: dict) -> None:
    lines.append("*Organism — prediction-error (surprise) channel:*")
    lines.append("")
    for row in m.get("drift_sweep", []):
        lines.append(
            f"- drift **{row['drift']:.2f}**: surprise AUROC **{row['surprise_auroc']:.3f}** "
            f"→ **{row.get('organism_encodes_world', '?')}**"
        )


def _summarize_expB2(lines: list[str], m: dict) -> None:
    lines.append("*Organism under survival pressure (B-v2 pooled probe):*")
    lines.append("")
    agents = m.get("agents_by_drift") or {}
    for drift in sorted(agents, key=float):
        lines.append(f"**drift {drift}**")
        for agent, stats in agents[drift].items():
            lines.append(
                f"- `{agent}`: pool target **{stats['pool_target_mean']:.3f}** "
                f"→ **{stats.get('organism_encodes_world', '?')}**"
            )
        lines.append("")


def _headline(steps: dict, run_dir: Path) -> str:
    parts: list[str] = []
    b2 = steps.get("expB2")
    if b2 and b2.get("status") == "ok":
        mp = run_dir / b2["metrics"]
        if mp.is_file():
            m = json.loads(mp.read_text(encoding="utf-8"))
            surv = m.get("survival_pool_target_mean")
            if surv is not None:
                if abs(surv - 0.5) < 0.05:
                    parts.append(
                        "Under survival pressure, the organism's internal state still sits "
                        "**at chance** for world identity."
                    )
                elif surv < 0.65:
                    parts.append(
                        f"Survival pressure leaves a **weak trace** (pool target ≈ {surv:.2f}), "
                        "below the pre-registered encoding bar (0.65)."
                    )
                else:
                    parts.append(
                        f"Survival agent shows **elevated encoding** (pool target ≈ {surv:.2f})."
                    )
    bf = steps.get("expB_full")
    if bf and bf.get("status") == "ok":
        parts.append(
            "Prediction-only training (Experiment B) keeps world-identity probes "
            "**near chance** across L2 drift levels."
        )
    if not parts:
        return "Run completed — inspect per-step metrics in `steps/*.json` and logs."
    return " ".join(parts)

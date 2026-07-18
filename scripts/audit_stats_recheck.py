"""Recompute every published L3/B-v2 number from committed artifacts.

Publication gate: loads the committed JSONs under artifacts/expB2/ and
recomputes, from the per-seed cell values, every quantitative claim quoted in
README.md, docs/FINDINGS.md sections 9-10, and docs/PAPER_OUTLINE.md. Fails
loudly (non-zero exit) on any mismatch beyond rounding.

Two interval types appear in the docs, both recomputed here:
  boot: seed-level percentile bootstrap of the across-seed mean
        (itasorl.stats.mean_ci, level 0.90, seed 0)
  t:    Student-t 90% CI of the across-seed mean (decision-relevant at the
        0.65 bar per PREREGISTRATION_L3.md and FINDINGS methods note 5)

Usage:
    python scripts/audit_stats_recheck.py
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from itasorl.stats import equivalence_test, mean_ci, rope_test  # noqa: E402

ARTROOT = os.path.join(os.path.dirname(__file__), "..", "artifacts")
ART = os.path.join(ARTROOT, "expB2")
TOL = 5e-4  # docs quote 3 decimals; allow rounding slack
BAR = 0.65

failures: list[str] = []
n_checks = 0


def t_ci(vals: list[float], level: float = 0.90) -> tuple[float, float]:
    from scipy.stats import t as student_t
    x = np.asarray(vals, dtype=float)
    n = x.size
    m = float(x.mean())
    se = float(x.std(ddof=1)) / math.sqrt(n)
    q = float(student_t.ppf(1.0 - (1.0 - level) / 2.0, n - 1))
    return (m - q * se, m + q * se)


def check(label: str, actual: float, expected: float, tol: float = TOL) -> None:
    global n_checks
    n_checks += 1
    ok = math.isfinite(actual) and abs(actual - expected) <= tol
    mark = "ok  " if ok else "FAIL"
    print(f"  [{mark}] {label}: doc {expected:.3f} vs recomputed {actual:.4f}")
    if not ok:
        failures.append(label)


def check_int(label: str, actual: int, expected: int) -> None:
    global n_checks
    n_checks += 1
    ok = actual == expected
    print(f"  [{'ok  ' if ok else 'FAIL'}] {label}: doc {expected} vs recomputed {actual}")
    if not ok:
        failures.append(label)


def check_true(label: str, actual: bool) -> None:
    global n_checks
    n_checks += 1
    print(f"  [{'ok  ' if actual else 'FAIL'}] {label}")
    if not actual:
        failures.append(label)


def load(name: str) -> dict:
    with open(os.path.join(ART, name), encoding="utf-8") as fh:
        return json.load(fh)


def seed_vals(doc: dict, drift: str, agent: str, metric: str) -> list[float]:
    rows = [r for r in doc["cells"]
            if r["drift"] == drift and r["agent"] == agent and metric in r]
    rows.sort(key=lambda r: r["seed"])
    return [float(r[metric]) for r in rows]


def verify_aggregate_consistency(name: str, doc: dict) -> None:
    """The stored `aggregate` block must be exactly reproducible from `cells`."""
    global n_checks
    bad = 0
    for arm, metrics in doc["aggregate"].items():
        d, agent = arm.split(" ", 1)
        drift = d.split("=")[1]
        for metric, stored in metrics.items():
            vals = [v for v in seed_vals(doc, drift, agent, metric)
                    if np.isfinite(v)]
            mean, lo, hi = mean_ci(vals)
            above = int(sum(v >= doc.get("bar", BAR) for v in vals))
            if (abs(mean - stored["mean"]) > 1e-9 or abs(lo - stored["lo"]) > 1e-9
                    or abs(hi - stored["hi"]) > 1e-9
                    or above != stored["n_above_bar"]):
                bad += 1
                failures.append(f"{name} aggregate {arm}/{metric}")
    n_checks += 1
    print(f"  [{'ok  ' if bad == 0 else 'FAIL'}] {name}: stored aggregate == recompute"
          f" from cells ({len(doc['cells'])} cells)")


def main() -> int:
    n10 = load("behavior_audit_l3_n10.json")
    h8 = load("behavior_audit_l3_h8_traces.json")
    h7 = load("behavior_audit_l3_h7_traces.json")
    held = load("heldout_l3_h8_summary.json")
    rev = load("heldout_l3_h7_reverse_summary.json")
    b2 = load("expB2_results.json")
    b2c = load("expB2_results_confirmatory_n3.json")

    print("== internal consistency: stored aggregates reproduce from per-seed cells ==")
    for name, doc in [("n10", n10), ("h8_traces", h8), ("h7_traces", h7),
                      ("heldout_summary", held), ("h7_reverse_summary", rev)]:
        verify_aggregate_consistency(name, doc)

    print("== FINDINGS 10.2 / README / PAPER_OUTLINE: L3 headline (hidden=8) ==")
    surv = seed_vals(h8, "0.45", "survival", "target")
    check("survival target mean (0.752)", float(np.mean(surv)), 0.752)
    lo, hi = mean_ci(surv)[1:]
    check("survival target boot lo (0.704)", lo, 0.704)
    check("survival target boot hi (0.797)", hi, 0.797)
    tlo, thi = t_ci(surv)
    check("survival target t lo (0.698)", tlo, 0.698)
    check("survival target t hi (0.807)", thi, 0.807)
    check_int("survival seeds >= 0.65 (8)", sum(v >= BAR for v in surv), 8)
    doc_seeds = [0.853, 0.636, 0.841, 0.823, 0.830, 0.573, 0.705, 0.782, 0.759, 0.723]
    check_true("per-seed list in FINDINGS 10.2 matches cells",
               all(abs(a - b) <= TOL for a, b in zip(surv, doc_seeds)))
    pred = seed_vals(h8, "0.45", "predictor", "target")
    check("predictor target mean (0.573)", float(np.mean(pred)), 0.573)
    check("predictor boot lo (0.546)", mean_ci(pred)[1], 0.546)
    check("predictor boot hi (0.599)", mean_ci(pred)[2], 0.599)
    untr = seed_vals(h8, "0.45", "untrained", "target")
    check("untrained target mean (0.488)", float(np.mean(untr)), 0.488)
    check("untrained boot lo (0.461)", mean_ci(untr)[1], 0.461)
    check("untrained boot hi (0.514)", mean_ci(untr)[2], 0.514)
    check_true("dissociation: survival - predictor >= 0.05 (h8)",
               float(np.mean(surv)) - float(np.mean(pred)) >= 0.05)

    print("== FINDINGS 10.2/10.3: L0 control and leakage ==")
    l0 = seed_vals(h8, "0.00", "survival", "target")
    check("L0 survival target (0.517)", float(np.mean(l0)), 0.517)
    check_true("L0 TOST equivalent to chance (margin 0.05)",
               equivalence_test(l0, margin=0.05).equivalent)
    check_true("L0 ROPE accepts equivalence", rope_test(l0).accept)
    leak = seed_vals(held, "0.45", "survival", "pool_reward_leak")
    check("reward-leak mean (0.541)", float(np.mean(leak)), 0.541)
    check("reward-leak min (0.517)", min(leak), 0.517)
    check("reward-leak max (0.559)", max(leak), 0.559)
    check_true("reward-leak clean 10/10",
               all(seed_vals(held, "0.45", "survival", "pool_leak_clean")))
    deaths = (seed_vals(held, "0.45", "survival", "deaths_auth")
              + seed_vals(held, "0.45", "survival", "deaths_surr"))
    check_true("0 early deaths in survival pools",
               all(d == 0 for d in deaths))
    check_true("110/110 episodes per pool",
               all(n == 110 for n in seed_vals(held, "0.45", "survival", "n_auth")
                   + seed_vals(held, "0.45", "survival", "n_surr")))

    print("== FINDINGS 10.4: behavior mediation (hidden=8) ==")
    check("behavior_only linear (0.689)",
          float(np.mean(seed_vals(h8, "0.45", "survival", "behavior_only"))), 0.689)
    check("behavior_only nonlinear (0.705)",
          float(np.mean(seed_vals(h8, "0.45", "survival", "behavior_only_nonlinear"))), 0.705)
    bt = seed_vals(h8, "0.45", "survival", "behavior_trace_only")
    check("behavior trace (0.803)", float(np.mean(bt)), 0.803)
    check("behavior trace boot lo (0.763)", mean_ci(bt)[1], 0.763)
    check("behavior trace boot hi (0.840)", mean_ci(bt)[2], 0.840)
    check("resid_epmean (0.676)",
          float(np.mean(seed_vals(h8, "0.45", "survival", "resid_epmean"))), 0.676)
    check("resid_epmean_quad (0.659)",
          float(np.mean(seed_vals(h8, "0.45", "survival", "resid_epmean_quad"))), 0.659)
    rt = seed_vals(h8, "0.45", "survival", "resid_trace")
    check("resid_trace (0.726)", float(np.mean(rt)), 0.726)
    check("resid_trace boot lo (0.685)", mean_ci(rt)[1], 0.685)
    check("resid_trace boot hi (0.765)", mean_ci(rt)[2], 0.765)
    check_int("resid_trace seeds >= 0.65 (9)", sum(v >= BAR for v in rt), 9)
    tlo, thi = t_ci(rt)
    check("resid_trace t lo (0.679)", tlo, 0.679)
    check("resid_trace t hi (0.772)", thi, 0.772)
    check_true("resid_trace t-CI excludes bar", tlo > BAR)
    rtq = seed_vals(h8, "0.45", "survival", "resid_trace_quad")
    check("resid_trace_quad (0.721)", float(np.mean(rtq)), 0.721)
    check("resid_trace_quad boot lo (0.678)", mean_ci(rtq)[1], 0.678)
    check("resid_trace_quad boot hi (0.760)", mean_ci(rtq)[2], 0.760)
    check("untrained resid_trace (0.498)",
          float(np.mean(seed_vals(h8, "0.45", "untrained", "resid_trace"))), 0.498)
    check("untrained behavior trace (0.645)",
          float(np.mean(seed_vals(h8, "0.45", "untrained", "behavior_trace_only"))), 0.645)
    check("predictor resid_trace (0.574)",
          float(np.mean(seed_vals(h8, "0.45", "predictor", "resid_trace"))), 0.574)

    print("== FINDINGS 10.5: second capacity (hidden=7) ==")
    surv7 = seed_vals(h7, "0.45", "survival", "target")
    pred7 = seed_vals(h7, "0.45", "predictor", "target")
    untr7 = seed_vals(h7, "0.45", "untrained", "target")
    check("h7 survival target (0.737)", float(np.mean(surv7)), 0.737)
    check("h7 survival boot lo (0.688)", mean_ci(surv7)[1], 0.688)
    check("h7 survival boot hi (0.780)", mean_ci(surv7)[2], 0.780)
    check_int("h7 survival seeds >= 0.65 (8)", sum(v >= BAR for v in surv7), 8)
    check("h7 predictor target (0.714)", float(np.mean(pred7)), 0.714)
    check("h7 untrained target (0.586)", float(np.mean(untr7)), 0.586)
    check_true("h7 dissociation NOT met (lead < 0.05, artifact-conditional claim)",
               float(np.mean(surv7)) - float(np.mean(pred7)) < 0.05)
    rt7 = seed_vals(h7, "0.45", "survival", "resid_trace")
    check("h7 resid_trace (0.722)", float(np.mean(rt7)), 0.722)
    check("h7 resid_trace boot lo (0.678)", mean_ci(rt7)[1], 0.678)
    check("h7 resid_trace boot hi (0.763)", mean_ci(rt7)[2], 0.763)
    tlo7, thi7 = t_ci(rt7)
    check("h7 resid_trace t lo (0.672)", tlo7, 0.672)
    check("h7 resid_trace t hi (0.773)", thi7, 0.773)
    check_int("h7 resid_trace seeds >= 0.65 (8)", sum(v >= BAR for v in rt7), 8)
    check("h7 resid_trace_quad (0.704)",
          float(np.mean(seed_vals(h7, "0.45", "survival", "resid_trace_quad"))), 0.704)
    check("h7 predictor resid_trace (0.691)",
          float(np.mean(seed_vals(h7, "0.45", "predictor", "resid_trace"))), 0.691)
    check("h7 untrained resid_trace (0.579)",
          float(np.mean(seed_vals(h7, "0.45", "untrained", "resid_trace"))), 0.579)
    for agent in ("survival", "predictor", "untrained"):
        m = float(np.mean(seed_vals(h7, "0.45", agent, "behavior_trace_only")))
        check_true(f"h7 {agent} behavior trace in stated 0.762-0.796 range ({m:.3f})",
                   0.762 - TOL <= m <= 0.796 + TOL)

    print("== FINDINGS 10.6: held-out fingerprint probe ==")
    tr = seed_vals(held, "0.45", "survival", "transfer_target")
    check("transfer survival (0.773)", float(np.mean(tr)), 0.773)
    tlo, thi = t_ci(tr)
    check("transfer t lo (0.722)", tlo, 0.722)
    check("transfer t hi (0.824)", thi, 0.824)
    check_int("transfer seeds >= 0.65 (9)", sum(v >= BAR for v in tr), 9)
    trp = seed_vals(held, "0.45", "predictor", "transfer_target")
    tru = seed_vals(held, "0.45", "untrained", "transfer_target")
    check("transfer predictor (0.633)", float(np.mean(trp)), 0.633)
    check_int("transfer predictor seeds >= 0.65 (3)", sum(v >= BAR for v in trp), 3)
    check("transfer untrained floor (0.569)", float(np.mean(tru)), 0.569)
    check_true("pre-registered transfer rule passes (>=0.65 and >untrained+0.05)",
               float(np.mean(tr)) >= BAR
               and float(np.mean(tr)) > float(np.mean(tru)) + 0.05)
    cg = seed_vals(held, "0.45", "survival", "cg_tail_target")
    check("common-garden survival (0.557)", float(np.mean(cg)), 0.557)
    tlo, thi = t_ci(cg)
    check("common-garden t lo (0.492)", tlo, 0.492)
    check("common-garden t hi (0.622)", thi, 0.622)
    check_int("common-garden seeds >= 0.65 (1)", sum(v >= BAR for v in cg), 1)
    check("common-garden predictor (0.409)",
          float(np.mean(seed_vals(held, "0.45", "predictor", "cg_tail_target"))), 0.409)
    check("common-garden untrained (0.377)",
          float(np.mean(seed_vals(held, "0.45", "untrained", "cg_tail_target"))), 0.377)
    check("late-tail decay (0.492)",
          float(np.mean(seed_vals(held, "0.45", "survival", "cg_latetail_target"))), 0.492)

    print("== FINDINGS 10.6 / PREREG 2026-07-16: reverse transfer (train h7, hold out h8) ==")
    rpool = seed_vals(rev, "0.45", "survival", "pool_target")
    check("reverse standard-half survival (0.737)", float(np.mean(rpool)), 0.737)
    check("reverse standard-half boot lo (0.688)", mean_ci(rpool)[1], 0.688)
    check("reverse standard-half boot hi (0.780)", mean_ci(rpool)[2], 0.780)
    rl0 = seed_vals(rev, "0.00", "survival", "pool_target")
    check("reverse L0 (0.517)", float(np.mean(rl0)), 0.517)
    check_true("reverse L0 TOST equivalent to chance",
               equivalence_test(rl0, margin=0.05).equivalent)
    check_true("reverse L0 ROPE accepts equivalence", rope_test(rl0).accept)
    rleak = seed_vals(rev, "0.45", "survival", "pool_reward_leak")
    check("reverse reward-leak (0.567)", float(np.mean(rleak)), 0.567)
    check_true("reverse reward-leak clean 10/10",
               all(seed_vals(rev, "0.45", "survival", "pool_leak_clean")))
    rtr = seed_vals(rev, "0.45", "survival", "transfer_target")
    check("reverse transfer survival (0.638)", float(np.mean(rtr)), 0.638)
    tlo, thi = t_ci(rtr)
    check("reverse transfer t lo (0.600)", tlo, 0.600)
    check("reverse transfer t hi (0.676)", thi, 0.676)
    check_int("reverse transfer seeds >= 0.65 (4)", sum(v >= BAR for v in rtr), 4)
    rtru = seed_vals(rev, "0.45", "untrained", "transfer_target")
    rtrp = seed_vals(rev, "0.45", "predictor", "transfer_target")
    check("reverse transfer untrained floor (0.525)", float(np.mean(rtru)), 0.525)
    check("reverse transfer predictor (0.603)", float(np.mean(rtrp)), 0.603)
    check_true("reverse frozen rule FAILS the absolute bar (0.638 < 0.65)",
               float(np.mean(rtr)) < BAR)
    check_true("reverse floor-margin clause passes (> untrained + 0.05)",
               float(np.mean(rtr)) > float(np.mean(rtru)) + 0.05)
    rcg = seed_vals(rev, "0.45", "survival", "cg_tail_target")
    check("reverse common-garden survival (0.598)", float(np.mean(rcg)), 0.598)
    tlo, thi = t_ci(rcg)
    check("reverse common-garden t lo (0.547)", tlo, 0.547)
    check("reverse common-garden t hi (0.649)", thi, 0.649)
    check_int("reverse common-garden seeds >= 0.65 (4)", sum(v >= BAR for v in rcg), 4)
    check("reverse common-garden predictor (0.504)",
          float(np.mean(seed_vals(rev, "0.45", "predictor", "cg_tail_target"))), 0.504)
    check("reverse common-garden untrained (0.456)",
          float(np.mean(seed_vals(rev, "0.45", "untrained", "cg_tail_target"))), 0.456)
    check("reverse late-tail decay (0.489)",
          float(np.mean(seed_vals(rev, "0.45", "survival", "cg_latetail_target"))), 0.489)

    print("== FINDINGS 9 / artifacts: B-v2 survival coupling (L2, n=3) ==")
    rep = [float(v) for v in b2["0.45"]["survival"]["pool_target"]]
    check("replication mean (0.523)", float(np.mean(rep)), 0.523)
    check("replication std ddof=0 (0.045)", float(np.std(rep)), 0.045)
    m, lo, hi = mean_ci(rep)
    check("replication boot lo (0.490)", lo, 0.490)
    check("replication boot hi (0.556)", hi, 0.556)
    for got, exp in zip(sorted(rep, reverse=True), [0.586, 0.495, 0.488]):
        check(f"replication per-seed ({exp})", got, exp)
    conf = [float(v) for v in b2c["0.45"]["survival"]["pool_target"]]
    check("confirmatory mean (0.595)", float(np.mean(conf)), 0.595)
    check("confirmatory std ddof=0 (0.014)", float(np.std(conf)), 0.014)

    print("== FINDINGS 2/3: Experiment A/B summaries (committed) ==")
    with open(os.path.join(ARTROOT, "expA", "summary.json"), encoding="utf-8") as fh:
        ea = json.load(fh)
    l1 = {c["label"]: c for c in ea["expA_l1"]["cells"]}
    check("L1 oracle at delta=0.06 (1.000)",
          l1["L1 discretization, delta=0.06"]["oracle_auroc"], 1.000)
    check("L1 L0 control (0.523)",
          l1["L0 control, identical world"]["oracle_auroc"], 0.523)
    check_true("L1 leakage audit passes on clean cells",
               l1["L1 discretization, delta=0.06"]["leakage_pass"]
               and l1["L0 control, identical world"]["leakage_pass"])
    check_true("L1 contaminated-reward negative control CAUGHT (leakage_pass False)",
               not l1["L1 + contaminated reward (+0.02 in surrogate)"]["leakage_pass"])
    l2 = {c["label"]: c for c in ea["expA_l2"]["cells"]}
    check("L2 oracle at drift=0.30 (0.993)",
          l2["L2 rollout drift, drift_sigma=0.30"]["oracle_auroc"], 0.993)
    check("L2 L0 control (0.440)",
          l2["L2 control, drift_sigma=0  -> identical dynamics"]["oracle_auroc"], 0.440)
    with open(os.path.join(ARTROOT, "expB", "summary.json"), encoding="utf-8") as fh:
        eb = json.load(fh)
    sweep = {r["drift"]: r for r in eb["expB_full"]["drift_sweep"]}
    check("expB recurrent-state target @0.45 (0.510)", sweep[0.45]["target_mean"], 0.510)
    check("expB recurrent-state std @0.45 (0.039)", sweep[0.45]["target_std"], 0.039)
    check("expB recurrent-state target @0.20 (0.509)", sweep[0.2]["target_mean"], 0.509)
    surp = {r["drift"]: r for r in eb["expB_surprise"]["drift_sweep_with_std"]}
    check("surprise channel @0.45 (0.596)", surp[0.45]["mean"], 0.596)
    check("surprise channel std @0.45 (0.007)", surp[0.45]["std"], 0.007)
    nl = {r["drift"]: r for r in eb["expB_nonlinear"]["drift_sweep_with_std"]}
    check("nonlinear probe @0.45 (0.482)", nl[0.45]["mean"], 0.482)
    check("nonlinear probe std @0.45 (0.031)", nl[0.45]["std"], 0.031)
    for row in eb["expB_kstep_rerun_20260713"]["horizons"]:
        check_true(f"kstep rerun horizon {row['open_horizon']} at chance "
                   f"({row['drift_045_target_mean']:.3f} in 0.48-0.51)",
                   0.48 - TOL <= row["drift_045_target_mean"] <= 0.51 + TOL)

    print("== FINDINGS 7.1: B-v3 n=10 and capacity ceiling (committed) ==")
    with open(os.path.join(ART, "bv3_n10_summary.json"), encoding="utf-8") as fh:
        bv3 = json.load(fh)["pooled_target_drift045"]["survival"]
    check("B-v3 survival n=10 (0.610)", bv3["mean"], 0.610)
    check("B-v3 boot lo (0.585)", bv3["boot90_lo"], 0.585)
    check("B-v3 boot hi (0.634)", bv3["boot90_hi"], 0.634)
    check_true("B-v3 CI entirely below 0.65 bar", bv3["boot90_hi"] < BAR)
    with open(os.path.join(ART, "sysid_ceiling_n10_summary.json"), encoding="utf-8") as fh:
        ceil = json.load(fh)["pooled_target_drift045"]["survival"]
    check("capacity ceiling n=10 (0.596)", ceil["mean"], 0.596)
    check("ceiling boot lo (0.577)", ceil["boot90_lo"], 0.577)
    check("ceiling boot hi (0.616)", ceil["boot90_hi"], 0.616)
    check_true("ceiling CI entirely below 0.65 bar", ceil["boot90_hi"] < BAR)
    ctlo, cthi = t_ci(ceil["pool_target_per_seed"])
    check("ceiling t lo (0.573)", ctlo, 0.573)
    check("ceiling t hi (0.619)", cthi, 0.619)
    check_true("ceiling t-CI also below 0.65 bar", cthi < BAR)

    print("== FINDINGS 10.7: cross-recipe transfer probe (committed) ==")
    with open(os.path.join(ARTROOT, "l3_crossrecipe", "summary.json"), encoding="utf-8") as fh:
        xr = json.load(fh)
    tr = xr["transfer_rff"]
    sv = tr["survival_per_seed"]
    m, lo, hi = mean_ci(sv, level=0.90, seed=0)
    check("cross-recipe survival mean (0.684)", m, 0.684)
    check("cross-recipe survival boot lo (0.657)", lo, 0.657)
    check("cross-recipe survival boot hi (0.710)", hi, 0.710)
    tlo, thi = t_ci(sv)
    check("cross-recipe survival t lo (0.654)", tlo, 0.654)
    check("cross-recipe survival t hi (0.715)", thi, 0.715)
    check_true("cross-recipe survival t-CI entirely above 0.65 bar", tlo > BAR)
    check_int("cross-recipe survival seeds >= 0.65 (7)",
              sum(1 for v in sv if v >= BAR), 7)
    mu, ulo, uhi = mean_ci(tr["untrained_per_seed"], level=0.90, seed=0)
    check("cross-recipe untrained floor mean (0.548)", mu, 0.548)
    check("cross-recipe untrained boot lo (0.538)", ulo, 0.538)
    check("cross-recipe untrained boot hi (0.557)", uhi, 0.557)
    mp, plo, phi = mean_ci(tr["predictor_per_seed"], level=0.90, seed=0)
    check("cross-recipe predictor mean (0.574)", mp, 0.574)
    check("cross-recipe predictor boot lo (0.554)", plo, 0.554)
    check("cross-recipe predictor boot hi (0.593)", phi, 0.593)
    check_true("cross-recipe frozen rule recomputed (survival >= 0.65 AND "
               "> untrained + 0.05)", m >= BAR and m > mu + 0.05)
    check_true("cross-recipe rule_pass recorded true", bool(tr["rule_pass"]))
    check("cross-recipe rule margin (0.034)", tr["rule_margin"], 0.034)
    check("cross-recipe integrity gate reproduced 0.752",
          xr["integrity_gate"]["drift045_survival_mean_reproduced"], 0.752)
    g0 = xr["gate0"]
    check("cross-recipe gate-0 rff oracle (0.887)", g0["rff"]["selected_oracle_auroc"], 0.887)
    check_true("cross-recipe gate-0 rff oracle in band",
               0.85 <= g0["rff"]["selected_oracle_auroc"] <= 0.95)
    check_true("cross-recipe cd dropped with empty window (floors > 0.6 "
               "wherever oracle in band)",
               g0["cd"]["dropped"] and all(
                   f > 0.6 for o, f in zip(g0["cd"]["sweep_oracle"],
                                           g0["cd"]["sweep_floor"]) if o >= 0.85))

    print("== FINDINGS Exp C: emergence pilot (milestone 3, N=48 G=30 3 seeds) ==")
    with open(os.path.join(ARTROOT, "expC", "emergence_pilot_summary.json"),
              encoding="utf-8") as fh:
        ec = json.load(fh)
    cells = sorted(ec["cells"], key=lambda r: r["seed"])
    gen0 = np.array([c["gen0_auroc"] for c in cells])
    ft = np.array([c["final_treat_auroc"] for c in cells])
    fc = np.array([c["final_ctrl_auroc"] for c in cells])
    d_treat = ft - gen0
    d_ctrl = fc - gen0
    contrast = d_treat - d_ctrl
    est = ec["estimand"]
    # per-seed deltas/contrast reproduce the stored estimand from raw AUROCs
    check_true("Exp C per-seed delta_treat reproduces stored",
               all(abs(a - b) <= 1e-9 for a, b in zip(d_treat, est["delta_treat"])))
    check_true("Exp C per-seed delta_ctrl reproduces stored",
               all(abs(a - b) <= 1e-9 for a, b in zip(d_ctrl, est["delta_ctrl"])))
    check_true("Exp C per-seed contrast reproduces stored",
               all(abs(a - b) <= 1e-9 for a, b in zip(contrast, est["contrast"])))
    for got, exp in zip(contrast, [-0.028, 0.033, 0.055]):
        check(f"Exp C per-seed contrast ({exp:+.3f})", float(got), exp)
    for got, exp in zip(ft, [0.530, 0.570, 0.604]):
        check(f"Exp C per-seed final treat AUROC ({exp:.3f})", float(got), exp)
    # headline contrast + both interval types, recomputed from the per-seed cells
    check("Exp C mean contrast (+0.020)", float(contrast.mean()), 0.020)
    check_true("Exp C mean contrast reproduces stored",
               abs(float(contrast.mean()) - est["mean_contrast"]) <= 1e-9)
    tlo, thi = t_ci(list(contrast))
    check("Exp C contrast t lo (-0.052)", tlo, -0.052)
    check("Exp C contrast t hi (+0.092)", thi, 0.092)
    check_true("Exp C t-CI reproduces stored",
               abs(tlo - est["t_ci90"][0]) <= 1e-9 and abs(thi - est["t_ci90"][1]) <= 1e-9)
    blo, bhi = mean_ci(list(contrast))[1:]
    check("Exp C contrast boot lo (-0.007)", blo, -0.007)
    check("Exp C contrast boot hi (+0.048)", bhi, 0.048)
    check_true("Exp C boot-CI reproduces stored",
               abs(blo - est["boot_ci90"][0]) <= 1e-9 and abs(bhi - est["boot_ci90"][1]) <= 1e-9)
    check("Exp C mean final treat AUROC (0.568)", float(ft.mean()), 0.568)
    # the pre-registered claim is a null: all three sub-conditions fail, CI spans 0
    check_true("Exp C contrast t-CI spans 0 (ci_excludes_zero False)",
               tlo < 0.0 < thi and not est["ci_excludes_zero"])
    check_true("Exp C mean contrast below SESOI 0.05 (meets_sesoi False)",
               float(contrast.mean()) < 0.05 and not est["meets_sesoi"])
    check_true("Exp C mean final treat AUROC below floor 0.65 (meets_auroc_floor False)",
               float(ft.mean()) < BAR and not est["meets_auroc_floor"])
    check_true("Exp C emergence_claim is False", not est["emergence_claim"])
    # mechanism: selection HAD grip (fitness moved in both arms) but did not route
    # it through detection - the separate-survival series is the evidence
    check_true("Exp C gate-2 fitness moved in both arms",
               ec["gates"]["gate2_fitness_moves_treat"]
               and ec["gates"]["gate2_fitness_moves_ctrl"])
    check_true("Exp C seed-0 treatment bit-reproducible",
               ec["gates"]["determinism_bit_reproducible"])
    check_true("Exp C fitness delta positive in every arm-run (selection had grip)",
               all(c["fit_delta_treat"] > 0 and c["fit_delta_ctrl"] > 0 for c in cells))
    check_true("Exp C authentic-world death rate fell under selection (gen0 ~0.58 -> final)",
               all(0.55 <= c["death_rate_auth_gen0"] <= 0.60 for c in cells)
               and all(c["death_rate_auth_final_treat"] < c["death_rate_auth_gen0"]
                       for c in cells))

    # ---- derived-doc qualifier guard ------------------------------------
    # The reactive-vs-persistent reading is PROVISIONAL until the section 10.6
    # re-score is adjudicated (FINDINGS correction banner, 2026-07-18). The
    # public-facing derived docs are hand-maintained and have drifted stale
    # before (audit fault: index.html stated the reading as final), so pin the
    # qualifier here. When the re-score is adjudicated and the banner comes
    # down, update or remove this block in the same change.
    print("\n== derived-doc qualifier guard (provisional reactive reading) ==")
    root = os.path.join(os.path.dirname(__file__), "..")

    def _read(relpath: str) -> str:
        with open(os.path.join(root, relpath), encoding="utf-8") as fh:
            return fh.read()

    for relpath, needle, label in [
        ("index.html", "PROVISIONAL",
         "index.html carries the PROVISIONAL qualifier"),
        ("CITATION.cff", "provisional",
         "CITATION.cff carries the provisional qualifier"),
        ("README.md", "PROVISIONAL",
         "README carries the PROVISIONAL qualifier"),
        ("docs/FINDINGS.md", "PROVISIONAL pending the 10.6 re-score",
         "FINDINGS section-1 ladder row carries the PROVISIONAL qualifier"),
        ("docs/PAPER_OUTLINE.md", "PROVISIONAL",
         "PAPER_OUTLINE carries the PROVISIONAL qualifier"),
    ]:
        check_true(label, needle in _read(relpath))
    # the reactive wording must never appear in index.html without the
    # qualifier attached in the same sentence block
    idx = _read("index.html")
    for phrase in ("not a persistent stored representation",
                   "not a stored representation"):
        for pos in _find_all(idx, phrase):
            window = idx[pos:pos + 260]
            check_true(f"index.html reactive claim at offset {pos} is qualified",
                       "PROVISIONAL" in window)

    print()
    if failures:
        print(f"RESULT: {len(failures)} of {n_checks} checks FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print(f"RESULT: all {n_checks} checks passed.")
    return 0


def _find_all(haystack: str, needle: str) -> list[int]:
    out, start = [], 0
    while (pos := haystack.find(needle, start)) != -1:
        out.append(pos)
        start = pos + 1
    return out


if __name__ == "__main__":
    raise SystemExit(main())

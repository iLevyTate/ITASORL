# Paper outline (arXiv draft skeleton)

*Working scaffold for the writeup. Every claim in the paper must trace to a row in
the claims inventory below; every row traces to a committed artifact. Update this
file as sections 10.5 (second capacity) and 10.6 (held-out probe) of FINDINGS.md
resolve.*

## Candidate titles

- Detectable but not encoded: incidental detection of surrogate worlds by a
  from-scratch digital organism
- When does an agent notice its world is fake? Detectability vs incidental
  encoding on a surrogate ladder
- Incidental encoding of a learned-dynamics fingerprint under survival pressure

## Structure

1. **Introduction** (from ITASORL.md Part I)
   - The reframed question: authentic base dynamics vs generative surrogate,
     detection read out, never rewarded.
   - The white space: social simulacra and evaluation-awareness work is
     semantically contaminated; this is substrate-grounded and tabula rasa.
   - Contribution: (a) the detectability-vs-encoding gap as a measurable object,
     (b) a robust L2 negative under survival pressure at scale, (c) a controlled
     L3 positive with a behavior-independent component, (d) the preregistered,
     gate-driven methodology itself.
2. **The world and the surrogate ladder** (from ITASORL_world_spec.md; FINDINGS 1)
   - "A Patch of Earth" v0; ~146-dim observation; invariants (tabula rasa,
     readout-not-reward, substrate-grounded).
   - Ladder L0-L4; difficulty calibrated by an oracle band, not by fiat.
3. **Methods** (from PREREGISTRATION*.md; FINDINGS 2, 9, 10.1, 11)
   - Experiment A oracles, matched pairs, leakage-audit battery.
   - Experiment B probes; three-agent design (untrained/predictor/survival) on an
     identical trunk; pooled + matched-pair readouts.
   - Gates: oracle band, engagement, L0 equivalence (TOST/ROPE), leakage.
   - Preregistration and deviation logging as first-class method.
4. **Results**
   - 4.1 Detectability ceilings (FINDINGS 2): L1 1.000, L2 0.993, L3 gate 0.928.
   - 4.2 The L2 negative arc (FINDINGS 3, 4, 9): chance across channels, horizons,
     probes; survival coupling 0.610 [0.585, 0.634]; capacity ceiling 0.596.
   - 4.3 The L3 positive (FINDINGS 10): 0.752 [0.698, 0.807] survival-only;
     controls; behavior mediation; 0.726 behavior-independent.
   - 4.4 Replications: second capacity resolved (FINDINGS 10.5): the
     behavior-independent signal replicates (0.722 vs 0.726) but the
     survival-vs-predictor dissociation is artifact-conditional; held-out
     fingerprint resolved (FINDINGS 10.6): transfer generalizes (0.773 vs
     untrained 0.569) but the common-garden control shows the signal is
     reactive, not a persistent representation (0.557, below bar; late tail
     0.492 at chance); cross-recipe resolved (FINDINGS 10.7): the direction
     also reads a different surrogate family (RFF ridge, 0.684 vs untrained
     0.548, rule passes), while the constant-drag family proved uncalibratable
     (empty gate-0 window) - coefficient-bias artifacts cannot be made
     subtle-but-detectable, independently motivating learned-texture surrogates.
5. **Limitations** (FINDINGS 11, verbatim spirit)
6. **Related work** (ITASORL.md Part I literature sections)
7. **Reproducibility statement** (FINDINGS 12; artifacts/expB2/ promotion history;
   CITATION.cff)

## Claims inventory

| # | Claim | Number | Doc section | Artifact | Figure |
|---|-------|--------|-------------|----------|--------|
| 1 | L0 control at chance (oracle) | AUROC 0.523 | FINDINGS 2.1 | artifacts/expA/summary.json | expA_ceiling.png |
| 2 | L1 detectable (oracle) | AUROC 1.000 | FINDINGS 2.1 | artifacts/expA/summary.json | expA_ceiling.png |
| 3 | L2 detectable (oracle) | AUROC 0.993 | FINDINGS 2.2 | artifacts/expA/summary.json | expA_L2_ceiling.png |
| 4 | L2 not encoded (recurrent state) | 0.510 ± 0.039 @ drift 0.45 | FINDINGS 3.1 | artifacts/expB/summary.json | expB_incidental.png |
| 5 | L2 surprise channel weak | 0.596 ± 0.007 @ drift 0.45 | FINDINGS 3.2 | artifacts/expB/summary.json | expB_channels.png |
| 6 | No liftoff with horizon | 0.48-0.51 across 0/8/16 | FINDINGS 3.3 | artifacts/expB/summary.json (recorded rerun; see 3.3 correction) | expB_kstep.png |
| 7 | L2 null robust (nonlinear probe) | 0.482 ± 0.031 | FINDINGS 3.4 | artifacts/expB/summary.json | - |
| 8 | Survival coupling does not rescue L2 | 0.523 ± 0.045 (replication) | FINDINGS 9 | artifacts/expB2/expB2_results.json | expB2_survival.png |
| 9 | B-v3 regime negative at scale | 0.610, 90% CI [0.585, 0.634], n=10 | FINDINGS 7.1 | artifacts/expB2/bv3_n10_summary.json | - |
| 10 | L2 capacity ceiling below bar | 0.596, 90% CI [0.577, 0.616], n=10 | FINDINGS 7.1 | artifacts/expB2/sysid_ceiling_n10_summary.json | - |
| 11 | L3 gate frozen in-band | oracle 0.928, floor 0.483 | FINDINGS 10.1 | PREREGISTRATION_L3.md sec. 12 | - |
| 12 | L3 encoded by survival only | 0.752, t 90% CI [0.698, 0.807], 8/10 seeds | FINDINGS 10.2 | artifacts/expB2/behavior_audit_l3_h8_traces.json | - |
| 13 | L3 predictor baseline near chance | 0.573 [0.546, 0.599] | FINDINGS 10.2 | artifacts/expB2/behavior_audit_l3_h8_traces.json | - |
| 14 | L3 untrained floor at chance | 0.488 [0.461, 0.514] | FINDINGS 10.2 | artifacts/expB2/behavior_audit_l3_h8_traces.json | - |
| 15 | Reward leak clean | 0.541, clean 10/10 | FINDINGS 10.3 | PREREGISTRATION_L3.md sec. 12 (n=10 audited entry) | - |
| 16 | No survivorship asymmetry | 0 early deaths, 110/110 all pools | FINDINGS 10.3 | PREREGISTRATION_L3.md sec. 12 | - |
| 17 | Behavior alone decodes world | trace 0.803 [0.763, 0.840] | FINDINGS 10.4 | artifacts/expB2/behavior_audit_l3_h8_traces.json | - |
| 18 | Behavior-independent component | 0.726 [0.685, 0.765], 9/10; quad 0.721 | FINDINGS 10.4 | artifacts/expB2/behavior_audit_l3_h8_traces.json | - |
| 19 | Control neither manufactures nor spares signal | untrained resid 0.498; predictor 0.574 | FINDINGS 10.4 | artifacts/expB2/behavior_audit_l3_h8_traces.json | - |
| 20 | Second capacity: behavior-independent signal replicates | 0.722 [0.678, 0.763], 8/10; quad 0.704 | FINDINGS 10.5 | artifacts/expB2/behavior_audit_l3_h7_traces.json | - |
| 21 | Second capacity: dissociation NOT met (artifact-conditional) | survival 0.737 [0.688, 0.780] vs predictor 0.714 [0.687, 0.740]; lead +0.023 < +0.05 | FINDINGS 10.5 | artifacts/expB2/behavior_audit_l3_h7_traces.json | - |
| 22 | Gate 0 re-validated per capacity; hidden=7 frozen | oracle 0.922, floor 0.566; hidden=8 regression exact (0.928/0.482); hidden=4 uninformative | FINDINGS 10.5 | PREREGISTRATION_L3.md sec. 12 + scripts/run_expA_l3.py | - |
| 23 | Held-out fingerprint transfer: GENERALIZES | survival 0.773 [0.722, 0.824], 9/10 vs untrained floor 0.569; rule passes | FINDINGS 10.6 | artifacts/expB2/heldout_l3_h8_summary.json | - |
| 24 | Common-garden control: REACTIVE not persistent | survival cg_tail 0.557 [0.492, 0.622], 1/10 (below bar); late tail 0.492 at chance; rule fails | FINDINGS 10.6 | artifacts/expB2/heldout_l3_h8_summary.json | - |

## Known gaps before submission

- Claims 1-7 resolved (2026-07-16): the recorded 06302026 e2e bundle's step
  metrics (plus the 2026-07-13 k-step rerun and the across-seed stds recovered
  from the bundle log) are promoted to `artifacts/expA/summary.json` and
  `artifacts/expB/summary.json` by `scripts/promote_ab_summaries.py`.
- Claims 9-10 resolved (2026-07-16): per-seed pooled targets promoted to
  `artifacts/expB2/bv3_n10_summary.json` and
  `artifacts/expB2/sysid_ceiling_n10_summary.json` by the same script;
  `scripts/audit_stats_recheck.py` re-verifies every inventory number against
  the committed artifacts.
- Rows 23-24 resolved (spec 2026-07-14, per-seed summary committed as
  `artifacts/expB2/heldout_l3_h8_summary.json`): transfer generalizes, the
  common-garden control reads reactive; the paper reports both as generality
  checks, not as a new headline. Rows 20-22 resolved 2026-07-14: the second
  capacity replicates the behavior-independent signal and bounds the
  survival-specific claim to the subtler artifact.

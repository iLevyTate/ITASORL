import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from experiment_a_l2 import generate_l2_pairs, run_experiment_a_l2

SIG, NP, BR = 2e-4, 120, 40
def show(tag, eps, contaminate=0.0):
    r = run_experiment_a_l2(eps, sigma_meas=SIG, contaminate=contaminate)
    lk = r["leakage"]
    print(tag)
    print(f"   oracle AUROC   = {r['oracle_auroc']:.3f}")
    print(f"   leakage reward = {lk['reward']:.3f}  length = {lk['length']:.3f}  metadata = {lk['metadata']:.3f}")
    print(f"   leakage gate   = {'PASS' if r['leakage_pass'] else 'FAIL (contamination!)'}\n")

print("Generating matched-pair drift data (agent-free) ...\n")
show("[L2 control, drift_sigma=0  -> identical dynamics]", generate_l2_pairs(NP, 0.0, branch=BR))
eps_main = generate_l2_pairs(NP, 0.3, branch=BR)
show("[L2 rollout drift, drift_sigma=0.30]", eps_main)
show("[L2 + contaminated reward (+0.02 in surrogate)]", eps_main, contaminate=0.02)

drifts = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
sweep = [run_experiment_a_l2(generate_l2_pairs(NP, d, branch=BR), sigma_meas=SIG)["oracle_auroc"] for d in drifts]
print("Detectability ceiling (oracle AUROC) vs drift strength  (vel meas-noise=%.0e):" % SIG)
for d, a in zip(drifts, sweep):
    print(f"   drift_sigma={d:5.2f}   AUROC={a:.3f}")

plt.figure(figsize=(7, 4.2))
plt.plot(drifts, sweep, "o-", lw=2, color="#2f855a", label="oracle (measured)")
plt.axhline(0.5, ls="--", color="grey", label="chance")
plt.xlabel("L2 drag-drift strength  (drift_sigma)")
plt.ylabel("detectability ceiling (AUROC)")
plt.title("ITASORL Experiment A - detectability ceiling vs rollout drift")
plt.ylim(0.45, 1.02); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("docs/figures/expA_L2_ceiling.png", dpi=130)
print("\nsaved docs/figures/expA_L2_ceiling.png")

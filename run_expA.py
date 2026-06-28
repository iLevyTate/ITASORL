import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiment_a import generate_clean, run_experiment_a

SIGMA = 0.01
print("Generating clean trajectories (agent-free, fixed policy) ...")
clean = generate_clean(n_pairs=100, steps=30, ray_steps=20)
print(f"  {len(clean)} pairs x 30 steps, obs dim {clean[0]['obs'].shape[1]}, sensor sigma={SIGMA}\n")

def show(tag, r):
    lk = r["leakage"]
    print(f"{tag}")
    print(f"   oracle AUROC      = {r['oracle_auroc']:.3f}")
    print(f"   leakage reward    = {lk['reward']:.3f}   length = {lk['length']:.3f}   metadata = {lk['metadata']:.3f}")
    print(f"   leakage gate      = {'PASS (no confound)' if r['leakage_pass'] else 'FAIL (contamination!)'}\n")

# Control: L0 (identical) -> detection must be at chance
show("[L0 control, identical world]", run_experiment_a(clean, "L0", delta=0.06, sigma=SIGMA))
# Main: L1 at a clearly-detectable grid -> oracle high, leakage clean
show("[L1 discretization, delta=0.06]", run_experiment_a(clean, "L1", delta=0.06, sigma=SIGMA))
# Contaminated: surrogate also makes food 'easier' -> the gate must catch it
show("[L1 + contaminated reward (+0.02 in surrogate)]",
     run_experiment_a(clean, "L1", delta=0.06, sigma=SIGMA, contaminate=0.02))

# Difficulty calibration: detectability ceiling vs grid spacing
deltas = [0.12, 0.08, 0.06, 0.04, 0.03, 0.02, 0.015, 0.01]
sweep = [run_experiment_a(clean, "L1", d, SIGMA)["oracle_auroc"] for d in deltas]
print("Detectability ceiling (oracle AUROC) vs grid spacing delta  (sigma=%.3f):" % SIGMA)
for d, a in zip(deltas, sweep):
    print(f"   delta={d:6.3f}  (delta/sigma={d/SIGMA:5.1f})   AUROC={a:.3f}")

plt.figure(figsize=(7, 4.2))
plt.semilogx(deltas, sweep, "o-", lw=2, color="#2b6cb0", label="oracle (measured)")
plt.axhline(0.5, ls="--", color="grey", label="chance")
plt.axvline(SIGMA, ls=":", color="#c05621", label=f"sigma = {SIGMA}")
plt.gca().invert_xaxis()
plt.xlabel("L1 grid spacing  delta  (log)")
plt.ylabel("detectability ceiling (AUROC)")
plt.title("ITASORL Experiment A - detectability ceiling vs discretization")
plt.ylim(0.45, 1.02)
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("docs/figures/expA_ceiling.png", dpi=130)
print("\nsaved docs/figures/expA_ceiling.png")

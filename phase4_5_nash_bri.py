# phase4_5_nash_bri.py — Nash Equilibrium via Best Response Iteration (BRI)
# Method: Top-K candidates + iterative best response until convergence
# Runtime: ~20-30 minutes on Mac (50 candidates x 6 lambda values)

import networkx as nx
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import os
from collections import defaultdict

os.makedirs("output", exist_ok=True)

# ── Load graph + simplicial data ──────────────────────────────────
print("Loading graph and simplicial complex...")
G = nx.read_edgelist("data/facebook_combined.txt",
                     create_using=nx.Graph(), nodetype=int)
N = G.number_of_nodes()

with open("data/simplicial_data.pkl", "rb") as f:
    sc_data = pickle.load(f)
triangle_membership = sc_data['triangle_membership']
print(f"Graph: {N} nodes | Triangles loaded ✓")

# ══════════════════════════════════════════════════════════════════
# TOP-K CANDIDATE POOL
# Only check top 50 nodes by degree — no rational firm picks
# a low-degree node as seed. Reduces 4039x4039 → 50x50 search.
# ══════════════════════════════════════════════════════════════════

K = 50 #25
dc = nx.degree_centrality(G)
TOP_K = [n for n in sorted(dc, key=dc.get, reverse=True)][:K]
print(f"Top-{K} candidates: degree range "
      f"{G.degree(TOP_K[0])} → {G.degree(TOP_K[-1])}")

# ══════════════════════════════════════════════════════════════════
# FAST HO-ICM — self-contained, no imports from other phases
# ══════════════════════════════════════════════════════════════════

def run_ho_icm_fast(G, triangle_membership, seeds_firm1, seeds_firm2,
                    lambda1=0.1, lambda2=0.35, num_runs=8):
    total = defaultdict(float)
    for _ in range(num_runs):
        state = {n: 0 for n in G.nodes()}
        for s in seeds_firm1: state[s] = 1
        for s in seeds_firm2: state[s] = 2

        newly_active1 = set(seeds_firm1)
        newly_active2 = set(seeds_firm2)

        for _ in range(30):
            next1, next2 = set(), set()

            # Pairwise spreading
            for node in newly_active1:
                for nbr in G.neighbors(node):
                    if state[nbr] == 0 and random.random() < lambda1:
                        state[nbr] = 1; next1.add(nbr)
            for node in newly_active2:
                for nbr in G.neighbors(node):
                    if state[nbr] == 0 and random.random() < lambda1:
                        state[nbr] = 2; next2.add(nbr)

            # Group/triangle spreading — only check neighbors of newly active
            candidates = set()
            for node in newly_active1 | newly_active2:
                for nbr in G.neighbors(node):
                    if state[nbr] == 0:
                        candidates.add(nbr)

            for node in candidates:
                if state[node] != 0: continue
                for tri in triangle_membership[node]:
                    others = [n for n in tri if n != node]
                    if all(state[o] == 1 for o in others):
                        if random.random() < lambda2:
                            state[node] = 1; next1.add(node); break
                    elif all(state[o] == 2 for o in others):
                        if random.random() < lambda2:
                            state[node] = 2; next2.add(node); break

            newly_active1, newly_active2 = next1, next2
            if not newly_active1 and not newly_active2: break

        total['s1'] += sum(1 for s in state.values() if s == 1) / N
        total['s2'] += sum(1 for s in state.values() if s == 2) / N

    return {k: v / num_runs for k, v in total.items()}

# ══════════════════════════════════════════════════════════════════
# BEST RESPONSE FUNCTIONS
# Given opponent's fixed seed, find YOUR best seed from TOP_K
# ══════════════════════════════════════════════════════════════════

def best_response_firm1(fixed_seed2, lambda2, num_runs=5):
    best_seed, best_payoff = None, -1
    for s1 in TOP_K:
        if s1 == fixed_seed2: continue
        r = run_ho_icm_fast(G, triangle_membership,
                            [s1], [fixed_seed2],
                            lambda1=0.1, lambda2=lambda2, num_runs=num_runs)
        if r['s1'] > best_payoff:
            best_payoff = r['s1']
            best_seed = s1
    return best_seed, best_payoff

def best_response_firm2(fixed_seed1, lambda2, num_runs=5):
    best_seed, best_payoff = None, -1
    for s2 in TOP_K:
        if s2 == fixed_seed1: continue
        r = run_ho_icm_fast(G, triangle_membership,
                            [fixed_seed1], [s2],
                            lambda1=0.1, lambda2=lambda2, num_runs=num_runs)
        if r['s2'] > best_payoff:
            best_payoff = r['s2']
            best_seed = s2
    return best_seed, best_payoff

# ══════════════════════════════════════════════════════════════════
# BEST RESPONSE ITERATION (BRI)
# Fixes: convergence allows oscillation detection (not just strict equality)
# ══════════════════════════════════════════════════════════════════

def find_nash_equilibrium(lambda2, max_iter=6, num_runs=5): #num_runs=3 #max_iter=4
    print(f"\n  λ₂={lambda2:.1f} — Running BRI...")

    seed1 = TOP_K[0]
    seed2 = TOP_K[1]

    history = []   # track (seed1, seed2) pairs to detect oscillation

    for iteration in range(max_iter):
        prev_seed1, prev_seed2 = seed1, seed2

        seed2, _ = best_response_firm2(seed1, lambda2, num_runs)
        seed1, _ = best_response_firm1(seed2, lambda2, num_runs)

        print(f"    Iter {iteration+1}: "
              f"seed1=node{seed1}(deg={G.degree(seed1)}) | "
              f"seed2=node{seed2}(deg={G.degree(seed2)})")

        # Strict convergence: neither firm changed
        if seed1 == prev_seed1 and seed2 == prev_seed2:
            print(f"    ✓ Converged at iteration {iteration+1}")
            break

        # Oscillation detection: same pair seen before → stable cycle, stop
        if (seed1, seed2) in history:
            print(f"    ✓ Stable cycle detected at iteration {iteration+1} — stopping")
            break

        history.append((prev_seed1, prev_seed2))

    # Final verification with more runs for accurate payoffs
    r = run_ho_icm_fast(G, triangle_membership,
                        [seed1], [seed2],
                        lambda1=0.1, lambda2=lambda2, num_runs=30)

    print(f"    Nash pair: (node{seed1}, node{seed2}) → "
          f"Firm1={r['s1']:.3f}, Firm2={r['s2']:.3f}")

    return seed1, seed2, r['s1'], r['s2']

# ══════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT — compute Nash Eq pair at each λ₂
# ══════════════════════════════════════════════════════════════════

lambda2_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
results = []

print("\n" + "═"*65)
print("PHASE 4.5 — Nash Equilibrium via Best Response Iteration")
print("═"*65)

for lam2 in lambda2_values:
    s1, s2, f1, f2 = find_nash_equilibrium(lam2)
    dom_ratio = f1 / (f1 + f2) if (f1 + f2) > 0 else 0.5
    results.append({
        'lambda2': lam2,
        'seed1': s1, 'seed2': s2,
        'deg1': G.degree(s1), 'deg2': G.degree(s2),
        'firm1': f1, 'firm2': f2,
        'dom_ratio': dom_ratio
    })

# ── Print results table ───────────────────────────────────────────
print("\n" + "═"*75)
print(f"{'λ₂':>5} | {'Seed1(deg)':>12} | {'Seed2(deg)':>12} | "
      f"{'Firm1':>7} | {'Firm2':>7} | {'DomRatio':>9} | {'Deviated?':>10}")
print("─"*75)
for r in results:
    deviated = "YES ✗" if abs(r['dom_ratio'] - 0.5) > 0.05 else "NO ✓"
    print(f"{r['lambda2']:>5.1f} | "
          f"node{r['seed1']:>4}({r['deg1']:>4}d) | "
          f"node{r['seed2']:>4}({r['deg2']:>4}d) | "
          f"{r['firm1']:>7.3f} | {r['firm2']:>7.3f} | "
          f"{r['dom_ratio']:>9.3f} | {deviated:>10}")
print("═"*75)

# ── Key findings ──────────────────────────────────────────────────
dom_vals   = [r['dom_ratio'] for r in results]
lam_vals   = [r['lambda2']   for r in results]
baseline   = dom_vals[0]
peak_idx   = dom_vals.index(max(dom_vals))
any_equal  = any(abs(d - 0.5) < 0.05 for d in dom_vals)

print("\nKEY FINDINGS:")
print(f"  Baseline (λ₂=0.0, pairwise): dom_ratio = {baseline:.3f}")
print(f"  Peak at λ₂={results[peak_idx]['lambda2']:.1f}:  dom_ratio = {dom_vals[peak_idx]:.3f}")
print(f"  At λ₂=0.5:                  dom_ratio = {dom_vals[-1]:.3f}")

if not any_equal:
    print("\n   No Nash pair gives equal competition (≈0.5)")
    print("   All 6 pairs deviate from Paper 1's (0.5, 0.5) baseline")
    if dom_vals[peak_idx] > baseline and dom_vals[-1] < dom_vals[peak_idx]:
        print("   Non-monotonic shift confirmed (rises then falls)")
else:
    print("\n  ⚠ Some pairs near 0.5 — increase K or num_runs for sharper results")

# ── Plot 1: Dominance ratio curve ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

n = len(lam_vals)
axes[0].plot(lam_vals, dom_vals, 'o-', color='crimson',
             linewidth=2.5, markersize=9, label='Nash Dom. Ratio (Firm 1)')
axes[0].axhline(0.5, color='gray', linestyle='--', linewidth=1.5,
                label='Paper 1 baseline (0.5 = equal)')
axes[0].fill_between(lam_vals,
                     [0.45]*n, [0.55]*n,
                     alpha=0.1, color='gray', label='±5% equal band')
axes[0].set_xlabel("λ₂ (Group Spreading Probability)", fontsize=12)
axes[0].set_ylabel("Firm 1 Dominance Ratio  [s1/(s1+s2)]", fontsize=12)
axes[0].set_title("Nash Equilibrium Shift under HO-ICM\n"
                  "(Best Response Iteration, Top-50 candidates)", fontsize=11)
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)
axes[0].set_ylim(0.3, 1.0)

# ── Plot 2: Firm captures bar chart ───────────────────────────────
f1_vals = [r['firm1'] for r in results]
f2_vals = [r['firm2'] for r in results]
x = np.arange(n)
w = 0.35
axes[1].bar(x - w/2, f1_vals, w, label='Firm 1', color='crimson', alpha=0.8)
axes[1].bar(x + w/2, f2_vals, w, label='Firm 2', color='steelblue', alpha=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels([f"λ₂={v}" for v in lam_vals], fontsize=9)
axes[1].set_ylabel("Fraction of Network Captured", fontsize=12)
axes[1].set_title("Firm Captures at Nash Equilibrium\n"
                  "(per λ₂ value)", fontsize=11)
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3, axis='y')
axes[1].set_ylim(0, 1.0)

plt.tight_layout()
plt.savefig("output/phase4_5_nash_bri.png", dpi=150)
plt.show()
print("\n✓ Phase 4.5 complete! Plot saved → output/phase4_5_nash_bri.png")

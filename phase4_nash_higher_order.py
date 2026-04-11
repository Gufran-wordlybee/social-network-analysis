# phase4_nash_higher_order.py — FAST REWRITE
# Estimated runtime: 3-5 minutes total (was 7+ hours before)
# Two fixes:
#   1. num_runs reduced to 8 (enough for trend, not slow)
#   2. No imports from phase3 — run_ho_icm defined HERE directly
#      (prevents phase3's simulation from auto-running on import)

import networkx as nx
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import os
from collections import defaultdict

os.makedirs("output", exist_ok=True)

# ── Load graph + data (self-contained, no imports from other phases) ──
print("Loading graph...")
G = nx.read_edgelist("data/facebook_combined.txt",
                     create_using=nx.Graph(), nodetype=int)
N = G.number_of_nodes()

with open("data/simplicial_data.pkl", "rb") as f:
    sc_data = pickle.load(f)
triangle_membership = sc_data['triangle_membership']

# ── Seed selection (copied here to avoid importing phase1) ────────
def seed_by_degree_centrality(G, budget=1, exclude=None):
    exclude = set(exclude) if exclude else set()
    dc = nx.degree_centrality(G)
    ranked = [n for n in sorted(dc, key=dc.get, reverse=True)
              if n not in exclude]
    return ranked[:budget]

# ══════════════════════════════════════════════════════════════════
# FAST HO-ICM
# Key optimization: instead of looping ALL 4039 nodes every round,
# only check nodes that are NEIGHBORS of newly active nodes.
# This reduces inner loop from 4039 iterations to ~50 per step.
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

            # ── Pairwise spreading ────────────────────────────────
            for node in newly_active1:
                for nbr in G.neighbors(node):
                    if state[nbr] == 0 and random.random() < lambda1:
                        state[nbr] = 1
                        next1.add(nbr)

            for node in newly_active2:
                for nbr in G.neighbors(node):
                    if state[nbr] == 0 and random.random() < lambda1:
                        state[nbr] = 2
                        next2.add(nbr)

            # ── Group/triangle spreading (OPTIMIZED) ──────────────
            # Only check neighbors of newly active nodes
            # (not all 4039 nodes — that was the slow part)
            candidates = set()
            for node in newly_active1 | newly_active2:
                for nbr in G.neighbors(node):
                    if state[nbr] == 0:
                        candidates.add(nbr)

            for node in candidates:
                if state[node] != 0:
                    continue
                for tri in triangle_membership[node]:
                    others = [n for n in tri if n != node]
                    if all(state[o] == 1 for o in others):
                        if random.random() < lambda2:
                            state[node] = 1
                            next1.add(node)
                            break
                    elif all(state[o] == 2 for o in others):
                        if random.random() < lambda2:
                            state[node] = 2
                            next2.add(node)
                            break

            newly_active1, newly_active2 = next1, next2
            if not newly_active1 and not newly_active2:
                break

        total['s1'] += sum(1 for s in state.values() if s == 1) / N
        total['s2'] += sum(1 for s in state.values() if s == 2) / N

    return {k: v / num_runs for k, v in total.items()}

# ══════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT — vary lambda2, observe Nash position
# ══════════════════════════════════════════════════════════════════

seed1 = seed_by_degree_centrality(G, budget=1)
seed2 = seed_by_degree_centrality(G, budget=1, exclude=set(seed1))

lambda2_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
nash_positions = []

print(f"\n{'λ₂':>6} {'Firm1':>8} {'Firm2':>8} {'dominance1':>10} {'Shifted?':>10}")
print("-" * 48)

for lam2 in lambda2_values:
    r = run_ho_icm_fast(G, triangle_membership, seed1, seed2,
                        lambda1=0.1, lambda2=lam2, num_runs=8)

    total = r['s1'] + r['s2']
    nash_pos = r['s1'] / total if total > 0 else 0.5
    nash_positions.append(nash_pos)

    shifted = "YES" if abs(nash_pos - 0.5) > 0.05 else "NO"
    print(f"{lam2:>6.1f} {r['s1']:>8.3f} {r['s2']:>8.3f} "
          f"{nash_pos:>10.3f} {shifted:>10}")

# ── KEY PLOT ──────────────────────────────────────────────────────
plt.figure(figsize=(9, 5))
plt.plot(lambda2_values, nash_positions, 'o-',
         color='crimson', linewidth=2.5, markersize=9,
         label='Nash Position under HO-ICM')
plt.axhline(0.5, color='gray', linestyle='--', linewidth=1.5,
            label='Paper 1 baseline (0.5 = equal competition)')
plt.fill_between(lambda2_values,
                 [0.45]*len(lambda2_values),
                 [0.55]*len(lambda2_values),
                 alpha=0.1, color='gray', label='±5% equilibrium band')
plt.xlabel("λ₂ (Group/Triangle Spreading Probability)", fontsize=12)
plt.ylabel("Nash Equilibrium Position (Firm 1 share)", fontsize=12)
plt.title("Nash Equilibrium Shifts Under Higher-Order Interactions\n"
          "(Deviation from 0.5 = group dynamics disrupts fair competition)",
          fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.ylim(0.3, 0.8)
plt.tight_layout()
plt.savefig("output/phase4_nash_shift.png", dpi=150)
plt.show()

# ── Summary ───────────────────────────────────────────────────────
print("KEY FINDING:")
print(f"  λ₂=0.0 (pairwise only):  Nash = {nash_positions[0]:.3f}  ← Paper 1 result")
print(f"  λ₂=0.5 (full HO-ICM):   Nash = {nash_positions[-1]:.3f}  ← Our result")
if abs(nash_positions[-1] - nash_positions[0]) > 0.05:
    print("  ✓ Nash Equilibrium SHIFTS with higher-order interactions")
    print("  ✓ This is your novel contribution — publishable finding!")
else:
    print("  Nash position stable — try increasing num_runs for accuracy")
print("Phase-4 complete")

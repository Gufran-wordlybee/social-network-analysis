# phase3_higher_order_icm.py — FIXED VERSION
# Higher-Order ICM: group (triangle) pressure added on top of pairwise

import networkx as nx
import numpy as np
import random
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import os

# ── import fixed seed functions from phase1 ───────────────────────
from phase1_baseline import (G, N,
                              seed_by_degree_centrality,
                              run_icm_pairwise)

os.makedirs("output", exist_ok=True)

with open("data/simplicial_data.pkl", "rb") as f:
    sc_data = pickle.load(f)
triangle_membership = sc_data['triangle_membership']

# ══════════════════════════════════════════════════════════════════
# HIGHER-ORDER ICM
# New mechanic: if ALL other nodes in a triangle are spreading
# firm X, the uninformed node gets an extra group-pressure push
# ══════════════════════════════════════════════════════════════════

def run_ho_icm(G, triangle_membership, seeds_firm1, seeds_firm2,
               lambda1=0.1, lambda2=0.35, num_runs=50):
    """
    lambda1 = pairwise spreading probability  (same as Paper 1)
    lambda2 = triangle/group spreading probability (NEW)
    """
    total = defaultdict(float)

    for _ in range(num_runs):
        state = {n: 0 for n in G.nodes()}
        for s in seeds_firm1: state[s] = 1
        for s in seeds_firm2: state[s] = 2

        newly_active1 = set(seeds_firm1)
        newly_active2 = set(seeds_firm2)

        for _ in range(30):
            next1, next2 = set(), set()

            # Pairwise spreading (original Paper 1 mechanism)
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

            # Group/triangle spreading (NEW higher-order mechanism)
            for node in list(G.nodes()):
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

# ── Compare pairwise vs higher-order ─────────────────────────────
seed1 = seed_by_degree_centrality(G, budget=1)
seed2 = seed_by_degree_centrality(G, budget=1, exclude=set(seed1))

print("Running comparison: Pairwise vs Higher-Order ICM...")
r_pair = run_icm_pairwise(G, seed1, seed2, num_runs=30)
r_ho   = run_ho_icm(G, triangle_membership, seed1, seed2,
                    lambda1=0.1, lambda2=0.35, num_runs=30)

print(f"\n{'Method':<25} {'Firm1':>8} {'Firm2':>8} {'|F1-F2|':>10}")
print("-" * 54)
print(f"{'Pairwise (Paper 1)':<25} {r_pair['s1']:>8.3f} "
      f"{r_pair['s2']:>8.3f} {abs(r_pair['s1']-r_pair['s2']):>10.3f}")
print(f"{'HO-ICM λ₂=0.35 (Ours)':<25} {r_ho['s1']:>8.3f} "
      f"{r_ho['s2']:>8.3f} {abs(r_ho['s1']-r_ho['s2']):>10.3f}")

print("\n✓ Phase 3 complete!")

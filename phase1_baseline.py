# phase1_baseline.py — FIXED VERSION
# Replicate Paper 1: Two firms compete via ICM, Nash Equilibrium at ~0.5

import networkx as nx
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import os

os.makedirs("output", exist_ok=True)

G = nx.read_edgelist("data/facebook_combined.txt",
                     create_using=nx.Graph(), nodetype=int)
N = G.number_of_nodes()
print(f"Graph loaded: {N} nodes, {G.number_of_edges()} edges")

# ══════════════════════════════════════════════════════════════════
# SEED SELECTION — KEY FIX: firms get DIFFERENT seeds
# exclude param ensures Firm 2 never picks Firm 1's seed
# ══════════════════════════════════════════════════════════════════

def seed_by_degree_centrality(G, budget=1, exclude=None):
    exclude = set(exclude) if exclude else set()
    dc = nx.degree_centrality(G)
    ranked = [n for n in sorted(dc, key=dc.get, reverse=True)
              if n not in exclude]
    return ranked[:budget]

def seed_by_eigenvector_centrality(G, budget=1, exclude=None):
    exclude = set(exclude) if exclude else set()
    try:
        ec = nx.eigenvector_centrality(G, max_iter=1000)
    except:
        ec = nx.degree_centrality(G)
    ranked = [n for n in sorted(ec, key=ec.get, reverse=True)
              if n not in exclude]
    return ranked[:budget]

def seed_by_rank_degree(G, budget=1, exclude=None, sample_size=200):
    exclude = set(exclude) if exclude else set()
    sample = [n for n in random.sample(list(G.nodes()),
              min(sample_size, N)) if n not in exclude]
    degrees = {n: G.degree(n) for n in sample}
    return sorted(degrees, key=degrees.get, reverse=True)[:budget]

# ══════════════════════════════════════════════════════════════════
# PAIRWISE ICM — Paper 1 original spreading model
# ══════════════════════════════════════════════════════════════════

def run_icm_pairwise(G, seeds_firm1, seeds_firm2,
                     lambda1=0.1, lambda2=0.1, num_runs=50):
    total = defaultdict(float)
    for _ in range(num_runs):
        state = {n: 0 for n in G.nodes()}
        for s in seeds_firm1: state[s] = 1
        for s in seeds_firm2: state[s] = 2

        active1 = list(seeds_firm1)
        active2 = list(seeds_firm2)

        for _ in range(30):
            new1, new2 = [], []
            for node in active1:
                for nbr in G.neighbors(node):
                    if state[nbr] == 0 and random.random() < lambda1:
                        state[nbr] = 1
                        new1.append(nbr)
            for node in active2:
                for nbr in G.neighbors(node):
                    if state[nbr] == 0 and random.random() < lambda2:
                        state[nbr] = 2
                        new2.append(nbr)
            active1, active2 = new1, new2
            if not active1 and not active2:
                break

        total['s1'] += sum(1 for s in state.values() if s == 1) / N
        total['s2'] += sum(1 for s in state.values() if s == 2) / N

    return {k: v / num_runs for k, v in total.items()}

# ══════════════════════════════════════════════════════════════════
# RUN EXPERIMENT
# ══════════════════════════════════════════════════════════════════

methods = {
    'DC': seed_by_degree_centrality,
    'EC': seed_by_eigenvector_centrality,
    'RD': seed_by_rank_degree
}

MARGIN = 0.05
results = {}


for name, fn in methods.items():
    seed1 = fn(G, budget=1)
    seed2 = fn(G, budget=1, exclude=set(seed1))   # ← THE FIX

    r = run_icm_pairwise(G, seed1, seed2, num_runs=30)
    diff = abs(r['s1'] - r['s2'])
    eq = "YES ✓" if diff < MARGIN else "NO ✗"
    results[name] = r


print("\nPaper 1 prediction: both firms ~equal supporters (Nash Eq at 0.5)")

# ── Plot ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(results))
w = 0.35
names = list(results.keys())

ax.bar(x - w/2, [results[n]['s1'] for n in names],
       w, label='Firm 1', color='crimson', alpha=0.8)
ax.bar(x + w/2, [results[n]['s2'] for n in names],
       w, label='Firm 2', color='steelblue', alpha=0.8)
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5,
           label='Nash Eq target (0.5)')
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=12)
ax.set_ylabel("Fraction of Supporters", fontsize=12)
ax.set_title("Phase 1 — Pairwise ICM Baseline (Paper 1 Replica)", fontsize=13)
ax.legend()
ax.set_ylim(0, 0.8)
plt.tight_layout()
plt.savefig("output/phase1_baseline.png", dpi=150)
plt.show()

"""
just simulation of sir's baseline model
we are not proving nash equillibrium here
we are just checking, showing if its exist or not
sir was able to prove 0.5 in his paper, may be we have done the coding wrongly  
basically my code is saying that if firm1 have selected x than firm 2 will not select x
and x is lets say 1000 connec and firm2 selected with 700.. huge gap.. thats reflecting in the code also
"""
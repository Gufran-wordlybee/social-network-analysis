# phase1_baseline.py
# Goal: Replicate Paper 1 — ICM + 3 seed methods + Nash Equilibrium

import networkx as nx
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt

# ── Load Graph ─────────────────────────────────────────────────────
G = nx.read_edgelist("data/facebook_combined.txt",
                     create_using=nx.Graph(), nodetype=int)
N = G.number_of_nodes()

# ══════════════════════════════════════════════════════════════════
# STEP 1: Three Seed Selection Methods (exactly as in Paper 1)
# ══════════════════════════════════════════════════════════════════

"""DC — pick node with highest degree"""
"""EC — pick node most connected to other high-degree nodes"""
"""RD — sample nodes, rank by degree within sample"""
def seed_by_degree_centrality(G, budget=1, offset=0):
    dc = nx.degree_centrality(G)
    ranked = sorted(dc, key=dc.get, reverse=True)
    return ranked[offset : offset + budget]

def seed_by_eigenvector_centrality(G, budget=1, offset=0):
    try:
        ec = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        ec = nx.degree_centrality(G)
    ranked = sorted(ec, key=ec.get, reverse=True)
    return ranked[offset : offset + budget]

def seed_by_rank_degree(G, budget=1, offset=0, sample_size=200):
    sample = random.sample(list(G.nodes()), min(sample_size, N))
    degrees = {n: G.degree(n) for n in sample}
    ranked = sorted(degrees, key=degrees.get, reverse=True)
    return ranked[offset : offset + budget]
# ══════════════════════════════════════════════════════════════════
# STEP 2: Independent Cascade Model — Pairwise (Paper 1 original)
# ══════════════════════════════════════════════════════════════════

def run_icm_pairwise(G, seeds_firm1, seeds_firm2,
                     lambda1=0.1, lambda2=0.1, num_runs=50):
    """
    Two firms spread simultaneously via ICM.
    Returns: (fraction_informed_1, fraction_informed_2,
               fraction_supporter_1, fraction_supporter_2)
    """
    total = defaultdict(float)

    for _ in range(num_runs):
        # States: 0=uninformed, 1=firm1, 2=firm2, 3=both
        state = {n: 0 for n in G.nodes()}
        for s in seeds_firm1: state[s] = 1
        for s in seeds_firm2: state[s] = 2

        active1 = list(seeds_firm1)
        active2 = list(seeds_firm2)

        for level in range(20):
            new_active1, new_active2 = [], []

            for node in active1:
                for neighbor in G.neighbors(node):
                    if state[neighbor] == 0:
                        if random.random() < lambda1:
                            state[neighbor] = 1
                            new_active1.append(neighbor)

            for node in active2:
                for neighbor in G.neighbors(node):
                    if state[neighbor] == 0:
                        if random.random() < lambda2:
                            state[neighbor] = 2
                            new_active2.append(neighbor)

            active1, active2 = new_active1, new_active2
            if not active1 and not active2:
                break

        # Count results
        informed1 = sum(1 for s in state.values() if s in [1, 3]) / N
        informed2 = sum(1 for s in state.values() if s in [2, 3]) / N

        # Supporter = node informed by ONLY one firm (higher influence wins)
        support1 = sum(1 for s in state.values() if s == 1) / N
        support2 = sum(1 for s in state.values() if s == 2) / N

        total['i1'] += informed1
        total['i2'] += informed2
        total['s1'] += support1
        total['s2'] += support2

    return {k: v/num_runs for k, v in total.items()}

# ══════════════════════════════════════════════════════════════════
# STEP 3: Game Theory — Compute Nash Equilibrium
# ══════════════════════════════════════════════════════════════════

def compute_position(supporter_fraction):
    """Convert supporter fraction to Hotelling position (0 to 1)"""
    return supporter_fraction / 2.0

def best_response(x_opponent):
    """
    Hotelling best response:
    If opponent is at x, best response is 1 - x (mirror).
    Nash Equilibrium = both at 0.5
    """
    if x_opponent > 0.5:
        return 1 - x_opponent
    elif x_opponent < 0.5:
        return 1 - x_opponent
    else:
        return 0.5

# ══════════════════════════════════════════════════════════════════
# STEP 4: Run Full Experiment (Paper 1 replica)
# ══════════════════════════════════════════════════════════════════

methods = {
    'DC': seed_by_degree_centrality,
    'EC': seed_by_eigenvector_centrality,
    'RD': seed_by_rank_degree
}

results = {}
print("Running baseline experiment (Paper 1 replica)...")
def seed_by_degree_centrality(G, budget=1, offset=0): #give firm 2 the 2nd-ranked node
    dc = nx.degree_centrality(G)
    ranked = sorted(dc, key=dc.get, reverse=True)
    return ranked[offset : offset + budget]

for method_name, method_fn in methods.items():
    seed1 = method_fn(G, budget=1)
    seed2 = method_fn(G, budget=1, offset=1)      # Both firms use same method
    
    res = run_icm_pairwise(G, seed1, seed2, num_runs=30)
    
    pos1 = compute_position(res['s1'])
    pos2 = compute_position(res['s2'])
    nash = best_response(pos2)
    
    results[method_name] = {
        'supporter_1': res['s1'],
        'supporter_2': res['s2'],
        'position_1': pos1,
        'position_2': pos2,
        'nash_equilibrium': nash
    }
    print(f"\n{method_name}:")
    print(f"  Supporter Firm 1: {res['s1']:.3f}")
    print(f"  Supporter Firm 2: {res['s2']:.3f}")
    print(f"  Nash Equilibrium: {nash:.3f}  (Paper 1 predicts: 0.5)")

# ── Plot supporters over levels (like Paper 1's Fig 7) ──────────
plt.figure(figsize=(10, 4))
for i, (mname, r) in enumerate(results.items()):
    plt.bar(i*3,   r['supporter_1'], color='red',  alpha=0.7, label='Firm 1' if i==0 else '')
    plt.bar(i*3+1, r['supporter_2'], color='blue', alpha=0.7, label='Firm 2' if i==0 else '')
    plt.text(i*3+0.5, max(r['supporter_1'], r['supporter_2'])+0.01, 
             mname, ha='center', fontsize=10)

plt.xlabel("Seed Selection Method")
plt.ylabel("Fraction of Supporters")
plt.title("Baseline Replication of Paper 1 (Pairwise ICM)")
plt.legend()
plt.tight_layout()
plt.savefig("output/phase1_baseline_results.png", dpi=150)
plt.show()
print("\n✓ Phase 1 complete — baseline replicated!")
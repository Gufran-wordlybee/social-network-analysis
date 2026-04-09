# phase2_simplicial_complex.py — FAST VERSION
# Replace the clique-finding section with this

import networkx as nx
import numpy as np
import pickle

G = nx.read_edgelist("data/facebook_combined.txt",
                     create_using=nx.Graph(), nodetype=int)

# ── Fast triangle finding (seconds, not minutes) ──────────────────
print("Finding triangles (fast method)...")

triangles = []
for node in G.nodes():
    neighbors = set(G.neighbors(node))
    for nbr in neighbors:
        if nbr > node:                          # avoid duplicates
            common = neighbors & set(G.neighbors(nbr))
            for common_node in common:
                if common_node > nbr:           # avoid duplicates
                    triangles.append((node, nbr, common_node))

print(f"Nodes:     {G.number_of_nodes()}")
print(f"Edges:     {G.number_of_edges()}")
print(f"Triangles: {len(triangles)}")           # expect ~1.6 million

# ── Triangle membership per node ──────────────────────────────────
triangle_membership = {node: [] for node in G.nodes()}
for tri in triangles:
    for node in tri:
        triangle_membership[node].append(tri)

avg = np.mean([len(v) for v in triangle_membership.values()])
max_t = max(len(v) for v in triangle_membership.values())
print(f"Avg triangles per node: {avg:.1f}")
print(f"Max triangles per node: {max_t}")

# ── Save ──────────────────────────────────────────────────────────
with open("data/simplicial_data.pkl", "wb") as f:
    pickle.dump({
        'triangles': triangles,
        'triangle_membership': triangle_membership
    }, f)

print("\n✓ Phase 2 complete — done in seconds!")
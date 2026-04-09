# phase2_simplicial_complex.py
# Previously: G was just a graph with edges
# Now: G becomes a simplicial complex with 0-simplices (nodes),
#      1-simplices (edges), 2-simplices (triangles), 3-simplices (4-cliques)

import networkx as nx
import xgi
import numpy as np
from itertools import combinations

G = nx.read_edgelist("data/facebook_combined.txt",
                     create_using=nx.Graph(), nodetype=int)

# ── Step 1: Find all cliques ───────────────────────────────────────
print("Finding cliques (takes ~30 seconds for Facebook)...")
all_cliques = list(nx.find_cliques(G))   # maximal cliques only

edges      = [tuple(sorted(c)) for c in all_cliques if len(c) == 2]
triangles  = [tuple(sorted(c)) for c in all_cliques if len(c) == 3]
tetrahedra = [tuple(sorted(c)) for c in all_cliques if len(c) == 4]

# Also extract all 2-subsets of larger cliques (closure property)
for clique in all_cliques:
    if len(clique) >= 3:
        for pair in combinations(clique, 2):
            edges.append(tuple(sorted(pair)))
    if len(clique) >= 4:
        for tri in combinations(clique, 3):
            triangles.append(tuple(sorted(tri)))

edges     = list(set(edges))
triangles = list(set(triangles))
tetrahedra= list(set(tetrahedra))

print(f"0-simplices (nodes):      {G.number_of_nodes()}")
print(f"1-simplices (edges):      {len(edges)}")
print(f"2-simplices (triangles):  {len(triangles)}")
print(f"3-simplices (tetrahedra): {len(tetrahedra)}")

# ── Step 2: Build XGI SimplicialComplex ───────────────────────────
SC = xgi.SimplicialComplex()
SC.add_nodes_from(G.nodes())
SC.add_simplices_from(triangles)   # XGI auto-adds closure (edges too)
SC.add_simplices_from(tetrahedra)
                        #thankyou bhaiya!! 💓
print(f"\nSimplicial Complex built:") 
print(f"  Nodes: {SC.num_nodes}")
print(f"  Edges in SC: {SC.num_edges}")

# ── Step 3: Triangle membership (needed for Phase 3 HO-ICM) ───────
triangle_membership = {node: [] for node in G.nodes()}
for tri in triangles:
    for node in tri:
        triangle_membership[node].append(tri)

max_tri = max(len(v) for v in triangle_membership.values())
avg_tri = np.mean([len(v) for v in triangle_membership.values()])
print(f"\nTriangle membership stats:")
print(f"  Max triangles per node: {max_tri}")
print(f"  Avg triangles per node: {avg_tri:.2f}")

# Save for later phases
import pickle
with open("data/simplicial_data.pkl", "wb") as f:
    pickle.dump({
        'triangles': triangles,
        'tetrahedra': tetrahedra,
        'triangle_membership': triangle_membership
    }, f)

print("\n✓ Phase 2 complete — simplicial complex built and saved!")
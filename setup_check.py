# setup_check.py — run this to confirm everything works

import networkx as nx
import xgi
import numpy as np
import matplotlib.pyplot as plt

# ── Load Facebook dataset ─────────────────────────────────────────
# After downloading facebook_combined.txt.gz from SNAP, place it in data/
G = nx.read_edgelist("data/facebook_combined.txt", 
                      create_using=nx.Graph(), nodetype=int)

print(f"Nodes: {G.number_of_nodes()}")       # should be 4039
print(f"Edges: {G.number_of_edges()}")       # should be 88234
print(f"Triangles: {sum(nx.triangles(G).values()) // 3}")  # ~1.6M triangles

# ── Quick visualization (small subgraph) ──────────────────────────
sub = G.subgraph(list(G.nodes())[:100])
nx.draw(sub, node_size=20, edge_color='gray', alpha=0.5)
plt.title("Facebook subgraph (100 nodes)")
plt.savefig("output/facebook_subgraph.png", dpi=150)
plt.show()
print("Setup complete ✓")

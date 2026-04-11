

KEY CONCEPTS:

1. ICM (Independent Cascade Model):
   Chain reaction like forwarding a WhatsApp message.
   Each node gets one chance to spread to each neighbor with probability λ.
   Once tried, never tries again. Cascade stops when nobody new is convinced.
   F1 + F2 supporters DO NOT sum to 1 — unreached nodes make up the rest.

2. Seed Node:
   Starting point of information spread.
   Core question = which node to pick as seed → game theory.
   IMPORTANT: Firm 1 and Firm 2 must pick DIFFERENT seed nodes.
   Bug we fixed: both firms were picking the same node (highest degree).
   Fix: seed2 = method(G, budget=1, exclude=set(seed1))

3. Hotelling's Model:
   Two ice cream vendors on a beach → both converge to the middle (0.5).
   Paper 1 maps to information spreading:
   Nash Equilibrium = neither firm can do better by switching seeds.

4. DC / EC / RD — Three Seed Selection Methods:
   DC (Degree Centrality) = pick node with most direct connections
                             → most popular person
   EC (Eigenvector Centrality) = pick node whose friends are also popular
                                  → person who knows influencers
   RD (Rank Degree) = sample 200 random nodes, pick best among them
                       → realistic limited-information scenario

5. Lambda (λ):
   Probability that information successfully spreads from one node to another.
   λ₁ = 0.1 → pairwise spreading probability (Paper 1 original)
   λ₂ = group/triangle spreading probability (our new addition)
   λ₂ > λ₁ because group pressure is stronger than individual pressure.

6. Higher-Order Interaction:
   Pairwise = one person influences one other (edge between 2 nodes)
   Higher-order = GROUP collectively influences a person
   Example: WhatsApp group where ALL members discuss → stronger influence
   Math:
   - 1-simplex = edge (2 nodes) — pairwise
   - 2-simplex = triangle (3 nodes) — 3-body group interaction
   - 3-simplex = tetrahedron (4 nodes) — 4-body group interaction

7. Simplicial Complex:
   If triangle {A,B,C} exists, then edges A-B, B-C, A-C must ALL exist
   (closure property). Richer math tools than hypergraph.
   Built from Facebook graph using triangle-finding algorithm.

8. Mean Field ODEs:
   Instead of simulating every node, track FRACTIONS of population.
   6 states: S, A, B, AB, a, b (always sum to 1)
   Paper 1 ODE: dA/dt = λ₁·S·A (pairwise term)
   Our extension: dA/dt = λ₁·S·A + M₂·S·A² (adds group pressure term)
   Purpose: proves finding is mathematically general, not just on Facebook.

OUR MODEL — Higher-Order ICM:
Two mechanisms work simultaneously:
  Step 1 (Pairwise — Paper 1): single neighbor nudges node → prob λ₁=0.1
  Step 2 (Group — OUR ADD): ALL members of triangle spreading →
                              node gets group pressure → prob λ₂=0.35

DATASET:
facebook_combined.txt (SNAP Stanford)
- 4039 nodes (people), 88234 edges (friendships)
- ~1.6 million triangles → rich in higher-order structure
- Avg clustering coefficient: 0.6055 (very high — friends know each other)
- Download: https://snap.stanford.edu/data/ego-Facebook.html
- After download: gunzip facebook_combined.txt.gz

KEY RESEARCH FINDING (for paper abstract):
"Higher-order interactions have a non-monotonic effect on competitive
equilibrium. Moderate group pressure (λ₂≈0.2) amplifies the dominant
firm's advantage, but strong group pressure (λ₂≥0.4) partially
democratizes competition by disproportionately amplifying the
weaker firm's spread."


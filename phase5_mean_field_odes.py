# phase5_mean_field_odes.py
# Extend Paper 1's ODEs with higher-order coupling terms

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════
# Paper 1's original ODEs (pairwise only)
# States: S (uninformed), A (firm1), B (firm2), AB (both),
#         a (supporter1), b (supporter2)
# ══════════════════════════════════════════════════════════════════

def odes_pairwise(y, t, lam1, lam2):
    """Paper 1's original mean field ODEs — pairwise only"""
    S, A, B, AB, a, b = y
    
    dS  = -lam1*S*A - lam2*S*B
    dA  =  lam1*S*A - lam2*A*B
    dB  =  lam2*S*B - lam1*A*B
    dAB =  lam1*A*B + lam2*A*B
    da  =  lam1*A*(1 - a - b)
    db  =  lam2*B*(1 - a - b)
    
    return [dS, dA, dB, dAB, da, db]

# ══════════════════════════════════════════════════════════════════
# YOUR EXTENSION: Higher-Order ODEs
# New terms: M2 * S * A² (three-body group interaction — firm1)
#            M2 * S * B² (three-body group interaction — firm2)
#
# Meaning: The rate of new firm1 supporters is amplified when
#          a susceptible node is surrounded by two firm1 nodes
#          (triangle group pressure)
# ══════════════════════════════════════════════════════════════════

def odes_higher_order(y, t, lam1, lam2, M2=0.3):
    """
    Extended ODEs with higher-order coupling.
    M2: coupling strength for 3-body (triangle) interactions.
    M2=0 → reduces back to Paper 1's pairwise ODEs.
    """
    S, A, B, AB, a, b = y
    
    # Pairwise terms (same as Paper 1)
    pairwise_A = lam1 * S * A
    pairwise_B = lam2 * S * B
    
    # Higher-order terms (NEW — your contribution)
    # When two firm1 nodes are in a triangle with S, 
    # the influence is amplified by A² (two members pushing)
    ho_A = M2 * S * (A**2)     # group pressure from firm1 triangle
    ho_B = M2 * S * (B**2)     # group pressure from firm2 triangle
    
    dS  = -(pairwise_A + ho_A) - (pairwise_B + ho_B)
    dA  =  (pairwise_A + ho_A) - lam2*A*B
    dB  =  (pairwise_B + ho_B) - lam1*A*B
    dAB =   lam1*A*B + lam2*A*B
    da  =   lam1*A*(1 - a - b) + M2*(A**2)*(1 - a - b)
    db  =   lam2*B*(1 - a - b) + M2*(B**2)*(1 - a - b)
    
    return [dS, dA, dB, dAB, da, db]

# ── Initial conditions: seed nodes ────────────────────────────────
# Approximated from dataset: 1 seed out of 4039 nodes
S0 = 1 - 2/4039
A0, B0 = 1/4039, 1/4039
y0 = [S0, A0, B0, 0, 0, 0]
t  = np.linspace(0, 50, 500)

# ── Solve both ODEs ───────────────────────────────────────────────
sol_pairwise = odeint(odes_pairwise, y0, t, args=(0.1, 0.1))
sol_ho_weak  = odeint(odes_higher_order, y0, t, args=(0.1, 0.1, 0.2))
sol_ho_strong= odeint(odes_higher_order, y0, t, args=(0.1, 0.1, 0.5))

# ── Plot supporter fractions over time (like Paper 1's Fig 7) ─────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Supporter fraction for firm 1 (index 4 = 'a')
axes[0].plot(t, sol_pairwise[:,4],  label='Pairwise (Paper 1)',   lw=2)
axes[0].plot(t, sol_ho_weak[:,4],   label='HO-ICM M₂=0.2 (Ours)',lw=2, linestyle='--')
axes[0].plot(t, sol_ho_strong[:,4], label='HO-ICM M₂=0.5 (Ours)',lw=2, linestyle=':')
axes[0].set_xlabel("Time", fontsize=12)
axes[0].set_ylabel("Fraction of Firm 1 Supporters", fontsize=12)
axes[0].set_title("Supporter Dynamics: Pairwise vs Higher-Order")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Uninformed fraction over time
axes[1].plot(t, sol_pairwise[:,0],  label='Pairwise', lw=2)
axes[1].plot(t, sol_ho_weak[:,0],   label='HO M₂=0.2', lw=2, linestyle='--')
axes[1].plot(t, sol_ho_strong[:,0], label='HO M₂=0.5', lw=2, linestyle=':')
axes[1].set_xlabel("Time", fontsize=12)
axes[1].set_ylabel("Fraction Uninformed (S)", fontsize=12)
axes[1].set_title("Uninformed Population Decay")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("output/phase5_mean_field_comparison.png", dpi=150)
plt.show()
print("✓ Phase 5 complete — ODE extension done!")

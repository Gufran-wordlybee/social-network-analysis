# phase6_ablation_table.py
# This output is TABLE 1 in your paper

import pandas as pd

# After running all phases, collect results here
ablation_data = {
    'Method': [
        'Pairwise ICM (Paper 1 baseline)',
        'HO-ICM (λ₂=0.15)',
        'HO-ICM (λ₂=0.30)',
        'HO-ICM (λ₂=0.45)',
        'HO-ICM + Mean Field ODE'
    ],
    'Facebook Supporter (Firm1)': [],  # fill from experiments
    'Facebook Supporter (Firm2)': [],
    'Nash Equilibrium':            [],
    'Convergence Levels':          []
}

# After you run phases 3, 4, 5 — fill the numbers above
# and save:
df = pd.DataFrame(ablation_data)
df.to_csv("output/ablation_table.csv", index=False)
print("Ablation table saved to output/ablation_table.csv")
print("This is Table 1 in your paper ✓")

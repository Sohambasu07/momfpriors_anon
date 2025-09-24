import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import math

# Define NEGATED park1 and park2 functions (to maximize)

def park2(x1, x2, x3, x4, s):
    x1 = 1 - 2 * (x1 - 0.6) ** 2
    x2 = x2
    x3 = 1 - 3 * (x3 - 0.5) ** 2
    x4 = 1 - (x4 - 0.8) ** 2
    A = 0.9 + 0.1 * s
    B = 0.1 * (1 - s)
    return (
        A * (5 - 2 / 3 * np.exp(x1 + x2) + x4 * np.sin(x3) * A - x3 + B) / 4
        - 0.7
    )

def park1(x1, x2, x3, x4, s):
    x1 = 1 - 2 * (x1 - 0.6) ** 2
    x2 = x2
    x3 = 1 - 3 * (x3 - 0.5) ** 2
    x4 = 1 - (x4 - 0.8) ** 2
    T1 = (
        (x1 + 1e-3 * (1 - s))
        / 2
        * np.sqrt(1 + (x2 + x3**2) * x4 / (x1**2 + 1e-4))
    )
    T2 = (x1 + 3 * x4) * np.exp(1 + np.sin(x3))
    A = 0.9 + 0.1 * s
    B = 0.1 * (1 - s)
    return A * (T1 + T2 - B) / 22 - 0.8

def is_pareto_efficient(costs):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] > c, axis=1) | np.all(costs[is_efficient] == c, axis=1)
            is_efficient[i] = True
    return is_efficient

# Grid
grid_size = 20
x1 = np.linspace(0.01, 1, grid_size)
x2 = np.linspace(0.01, 1, grid_size)
x3 = np.linspace(0.01, 1, grid_size)
x4 = np.linspace(0.01, 1, grid_size)
X1, X2, X3, X4 = np.meshgrid(x1, x2, x3, x4)
flat_X1 = X1.ravel()
flat_X2 = X2.ravel()
flat_X3 = X3.ravel()
flat_X4 = X4.ravel()

# Objective evaluations
park1_vals = park1(flat_X1, flat_X2, flat_X3, flat_X4, 1.0)
park2_vals = park2(flat_X1, flat_X2, flat_X3, flat_X4, 1.0)
obj_values = np.vstack([park2_vals, park1_vals]).T

# Pareto front at fidelity=1
mask = is_pareto_efficient(obj_values)
pareto_front = obj_values[mask]
pareto_inputs = np.vstack([flat_X1, flat_X2, flat_X3, flat_X4]).T[mask]

# Select 2 random points
np.random.seed(42)
# indices = np.random.choice(len(pareto_inputs), size=2, replace=False)
samples = np.random.uniform(0, 1, (20, 4))  # Randomly sample 20 points in [0, 1]^4
# samples = pareto_inputs[indices]

# Fidelity levels
fidelity_levels = [0.0, 0.1, 0.2, 0.5, 0.7, 0.9, 1.0]
cmap = get_cmap("viridis", len(fidelity_levels))

# Plot
plt.figure(figsize=(9, 6))
plt.scatter(pareto_front[:, 0], pareto_front[:, 1], color='red', s=20, label="True Pareto front (s=1.0)")

# Ablation across fidelity
for i, (x1_s, x2_s, x3_s, x4_s) in enumerate(samples):
    for j, s in enumerate(fidelity_levels):
        p1 = park1(x1_s, x2_s, x3_s, x4_s, s)
        p2 = park2(x1_s, x2_s, x3_s, x4_s, s)
        plt.scatter(p2, p1, color=cmap(j), label=f"Fidelity {s}" if i == 0 else None, edgecolor='black')

# Labels and legend
plt.xlabel("park2 (maximize)")
plt.ylabel("park1 (maximize)")
plt.title("Maximization: Two Pareto Configs Across Fidelity Levels")
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()

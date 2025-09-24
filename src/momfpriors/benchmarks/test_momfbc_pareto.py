import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import math

# Define NEGATED Branin and Currin functions (to maximize)

def mf_branin(x1, x2, s):
    x11 = 15 * x1 - 5
    x22 = 15 * x2
    b = 5.1 / (4 * math.pi**2) - 0.01 * (1 - s)
    c = 5 / math.pi - 0.1 * (1 - s)
    r = 6
    t = 1 / (8 * math.pi) + 0.05 * (1 - s)
    y = (x22 - b * x11**2 + c * x11 - r) ** 2 + 10 * (1 - t) * np.cos(x11) + 10
    B = 21 - y
    return B / 22

def mf_currin(x1, x2, s):
    A = 2300 * x1**3 + 1900 * x1**2 + 2092 * x1 + 60
    B = 100 * x1**3 + 500 * x1**2 + 4 * x1 + 20
    y = (1 - 0.1 * (1 - s) * np.exp(-1 / (2 * x2))) * A / B
    C = -y + 14
    return C / 15

def is_pareto_efficient(costs):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] > c, axis=1) | np.all(costs[is_efficient] == c, axis=1)
            is_efficient[i] = True
    return is_efficient

# Grid
grid_size = 200
x1 = np.linspace(0.01, 1, grid_size)
x2 = np.linspace(0.01, 1, grid_size)
X1, X2 = np.meshgrid(x1, x2)
flat_X1 = X1.ravel()
flat_X2 = X2.ravel()

# Objective evaluations
branin_vals = mf_branin(flat_X1, flat_X2, 1.0)
currin_vals = mf_currin(flat_X1, flat_X2, 1.0)
obj_values = np.vstack([branin_vals, currin_vals]).T

# Pareto front at fidelity=1
mask = is_pareto_efficient(obj_values)
pareto_front = obj_values[mask]
pareto_inputs = np.vstack([flat_X1, flat_X2]).T[mask]

# Select 2 random points
np.random.seed(42)
indices = np.random.choice(len(pareto_inputs), size=2, replace=False)
samples = pareto_inputs[indices]

# Fidelity levels
fidelity_levels = [0.0, 0.1, 0.2, 0.5, 0.7, 0.9, 1.0]
cmap = get_cmap("viridis", len(fidelity_levels))

# Plot
plt.figure(figsize=(9, 6))
plt.scatter(pareto_front[:, 0], pareto_front[:, 1], color='red', s=20, label="True Pareto front (s=1.0)")

# Ablation across fidelity
for i, (x1_s, x2_s) in enumerate(samples):
    for j, s in enumerate(fidelity_levels):
        b = mf_branin(x1_s, x2_s, s)
        c = mf_currin(x1_s, x2_s, s)
        plt.scatter(b, c, color=cmap(j), label=f"Fidelity {s}" if i == 0 else None, edgecolor='black')

# Labels and legend
plt.xlabel("Branin (maximize)")
plt.ylabel("Currin (maximize)")
plt.title("Maximization: Two Pareto Configs Across Fidelity Levels")
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()

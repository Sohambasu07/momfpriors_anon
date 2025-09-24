from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from momfpriors.benchmarks.mf_zdt import mf_zdt1_fn, mf_zdt6_fn

map_str_to_fn = {
    "MF-ZDT1": (mf_zdt1_fn, 30),
    "MF-ZDT6": (mf_zdt6_fn, 10),
}


def plot_function(
    fn: str,
    fidelities: list[int],
    save_file_type: str = "pdf",
    plot_suffix: str = "",
) -> None:
    """Plot the specified multi-fidelity function with given fidelities.

    Args:
        fn: The name of the function to plot
        fidelities: List of fidelities to plot
        save_file_type: The file type to save the plot
        plot_suffix: Suffix to append to the saved file name.
    """
    fn, num_rest = map_str_to_fn[fn]
    x0 = 0.5
    x_rest = [0.5] * (num_rest - 2)

    # Grid resolution
    n = 100
    x1_vals = np.linspace(0, 1, n)
    x2_vals = np.linspace(0, 1, n)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    s_values = fidelities[:4]  # Limit to 4 fidelities for plotting

    fig = plt.figure(figsize=(16, 12))
    for idx, s in enumerate(s_values, 1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        F2 = np.zeros_like(X1)

        for i in range(n):
            for j in range(n):
                x = [x0, X1[i, j], X2[i, j], *x_rest]
                F1, F2[i, j] = fn(x, s / 100.0)

        ax.plot_surface(X1, X2, F2, cmap="viridis")
        ax.set_title(f"f2 Surface for s = {s}")
        xlabel = ax.set_xlabel("x[1]")
        ylabel = ax.set_ylabel("x[2]")
        zlabel = ax.set_zlabel("f2")
        xlabel.set_position((0.5, -0.1, 0))
        ylabel.set_position((-0.1, 0.5, 0))
        zlabel.set_position((0, 0, 1.05))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--function", "-fn",
        type=str,
        choices=["MF-ZDT1", "MF-ZDT6"],
        default="MF-ZDT1",
        help="Select the function to plot: MF-ZDT1 or MF-ZDT6"
    )
    parser.add_argument(
        "--fidelities", "-fids",
        nargs="+",
        type=int,
        default=[10, 50, 75, 100],
        help="List of fidelities to plot (max 4 values)"
    )
    parser.add_argument(
        "--save_file_type", "-file",
        type=str,
        choices=["png", "pdf", "svg"],
        default="pdf",
        help="File type to save the plot"
    )
    parser.add_argument(
        "--plot_suffix", "-suffix",
        type=str,
        default="",
        help="Suffix to append to the saved file name"
    )
    args = parser.parse_args()

    plot_function(
        fn=args.function,
        fidelities=args.fidelities,
        save_file_type=args.save_file_type,
        plot_suffix=args.plot_suffix
    )


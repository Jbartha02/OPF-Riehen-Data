from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

import utils

def _plot_FFOR(output_folder: str, points: list[tuple[float, float]], SHOW_PLOTS: bool, SAVE_PLOTS: bool = True) -> None:
    """Plot the convex hull of the given points and save the figure to the output folder."""
    points = np.asarray(points, dtype=float)
    hull = ConvexHull(points)

    # Plot the convex hull
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111)

    # plot points
    ax.scatter(points[:, 0], points[:, 1], c="blue", s=50, zorder=5)
    # plot lines between points in convex hull
    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1], "gray", linewidth=2)

    ax.set_xlabel("P_flex [kW]", fontsize=16)
    ax.set_ylabel("Q_flex [kVar]", fontsize=16)
    ax.set_title("FFOR", fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    if SAVE_PLOTS:
        fig.savefig(
            f"{output_folder}/plot_FFOR_{len(points)}.png",
            bbox_inches="tight",
            dpi=400,
        )
    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    results = utils.extract_results_parameters_from_scenario(r"2703_23_homogen/results")
    filtered_results = [result for result in results if result["analysis_year"] == 2030]

    for result in filtered_results:
        print(f"Processing {result['output_folder']}...")

        points_file = Path(result["output_folder"]) / "results_pq_flex_points.csv"
        if not points_file.is_file():
            print(f"Warning: {points_file} does not exist, skipping.")
            continue

        points = np.loadtxt(points_file, delimiter=",", skiprows=1)
        _plot_FFOR(result["output_folder"], points, SHOW_PLOTS=True, SAVE_PLOTS=False)
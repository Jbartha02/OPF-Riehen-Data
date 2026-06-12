from pathlib import Path

import numpy as np
import pandas as pd
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

def plot_seasonal_p_flex(df: pd.DataFrame):
    """
    Plot the seasonal variation of the min and max P_flex values.
    Input df should have a datetime index and columns "min_p_flex" and "max_p_flex".
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.index, df["min_p_flex"], label="Min P_flex", marker="o")
    ax.plot(df.index, df["max_p_flex"], label="Max P_flex", marker="o")
    ax.set_xlabel("Date")
    ax.set_ylabel("P_flex [kW]")
    ax.set_title("Seasonal Variation of P_flex")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
def plot_pv_weather_seasonality_p_flex(df: dict[str, pd.DataFrame], year: int):
    """
    Plot for the report of the seasonal variation of the min and max P_flex values for differend PV weather conditions.
    Input df should have a datetime index and columns "min_p_flex" and "max_p_flex".
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for pv_weather, df_pv in df.items():
        color = {
            "pvcld": "deepskyblue",
            "pvsun": "blue",
        }
        legend_label = {
            "pvcld": "Cloudy Day 12:00",
            "pvavg": "Average Day 12:00",
            "pvsun": "Sunny Day 12:00",
            "midnight": "No PV 00:00"
        }
        if pv_weather == "pvavg":
            ax.stackplot(df_pv.index, df_pv["min_p_flex"], labels=[legend_label.get(pv_weather)], color='paleturquoise', alpha=0.3)
            ax.stackplot(df_pv.index, df_pv["max_p_flex"], color='darkturquoise', alpha=0.3)
            ax.plot(df_pv.index, df_pv["min_p_flex"], color='paleturquoise', alpha=0.6, linewidth=4, marker="o")
            ax.plot(df_pv.index, df_pv["max_p_flex"], color='darkturquoise', alpha=0.4, linewidth=4, marker="o")
        elif pv_weather == "midnight":
            ax.plot(df_pv.index, df_pv["min_p_flex"], label=legend_label.get(pv_weather), linewidth=3, linestyle='--', color='gray', marker="x")
            ax.plot(df_pv.index, df_pv["max_p_flex"], label=None, linewidth=3, linestyle='--', color='gray', marker="x")
        else:
            ax.plot(df_pv.index, df_pv["min_p_flex"], label=legend_label.get(pv_weather), color=color.get(pv_weather, None), linewidth=4, marker="o")
            #ax.plot(df_pv.index, df_pv["max_p_flex"], label=legend_label.get(pv_weather), color=color.get(pv_weather, None), linewidth=2, marker="o")
    ax.set_xlabel("Date", fontsize=16)
    ax.set_ylabel("P_flex [kW]", fontsize=16)
    ax.set_title(f"Seasonal variation of P_flex for different PV scenarios in {year}, T=2h", fontsize=20, pad=5)
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
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
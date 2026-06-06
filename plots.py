import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

import config


def _plot_FFOR(conf: config.Config, points: list[tuple[float, float]], SHOW_PLOTS: bool) -> None:
    """Plot the convex hull of the given points and save the figure to the output folder."""
    points = np.asarray(points, dtype=float)
    hull = ConvexHull(points)
    
    # Plot the convex hull
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111)
    
    # plot points
    ax.scatter(points[:, 0], points[:, 1], c='blue', s=50, zorder=5)
    # plot lines between points in convex hull
    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1], 'gray', linewidth=2)
    
    ax.set_xlabel('P_flex [kW]', fontsize=16)
    ax.set_ylabel('Q_flex [kVar]', fontsize=16)
    ax.set_title('FFOR', fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    fig.savefig(f"{conf.output_folder}/plot_FFOR_{len(points)}.png", bbox_inches='tight', dpi=400)
    if SHOW_PLOTS:
        plt.show()
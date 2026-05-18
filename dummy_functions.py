# ==============================================================================
# this contains some unimportant functions
# ==============================================================================

import config
import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D





# TODO: remove this function again (just to check implementation)
def plot_first_10_nodes(conf: config.Config, p_pv: gp.tupledict, p_pv_flex: gp.tupledict, p_hp: gp.tupledict, p_hp_flex: gp.tupledict, p_bess_pos: gp.tupledict, p_bess_neg: gp.tupledict) -> None:
    """Plot p_pv and p_hp trajectories for the first 10 node indices."""
    
    nodes = list(conf.node_metadata_df.index[:10])
    time_points = conf.time_index_list

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 9))

    for node in nodes:
        pv_values = [p_pv[node, t].X if (node, t) in p_pv else 0.0 for t in time_points]
        pv_flex_values = [p_pv_flex[node, t].X if (node, t) in p_pv_flex else 0.0 for t in time_points]
        hp_values = [p_hp[node, t].X if (node, t) in p_hp else 0.0 for t in time_points]
        hp_flex_values = [p_hp_flex[node, t].X if (node, t) in p_hp_flex else 0.0 for t in time_points]
        bess_values = [p_bess_pos[node, t].X + p_bess_neg[node, t].X if (node, t) in p_bess_pos and (node, t) in p_bess_neg else 0.0 for t in time_points]
        axes[0].plot(time_points, pv_values, label=f"Node {node}")
        axes[0].plot(time_points, pv_flex_values, linestyle='--', label=f"Node {node} Flex")
        axes[1].plot(time_points, hp_values, label=f"Node {node}")
        axes[1].plot(time_points, hp_flex_values, linestyle='--', label=f"Node {node} Flex")
        axes[2].plot(time_points, bess_values, label=f"Node {node}")

    axes[0].set_title("p_pv (first 10 nodes)")
    axes[1].set_title("p_hp (first 10 nodes)")
    axes[2].set_title("p_bess (first 10 nodes)")
    axes[1].set_xlabel("Time index")
    axes[0].set_ylabel("kW")
    axes[1].set_ylabel("kW")
    axes[2].set_ylabel("kW")
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[2].grid(True, alpha=0.3)
    axes[0].legend(ncol=2, fontsize=8)

    plt.tight_layout()
    plt.show()




# TODO: remove this function again (just to check implementation)
def plot_octagon_and_circle():

    # 1. Setup Parameters
    P = 4.0                  # Radius of the circle
    alpha = np.sqrt(2) - 1   # Tangent of 22.5 degrees

    # Create a dense grid of points to evaluate the inequalities
    x = np.linspace(-P * 1.5, P * 1.5, 400)
    y = np.linspace(-P * 1.5, P * 1.5, 400)
    X, Y = np.meshgrid(x, y)

    # 2. Evaluate the 8 Inequalities
    # Condensed into absolute value logic for the interior area
    cond1 = np.abs(X) + alpha * np.abs(Y) <= P
    cond2 = alpha * np.abs(X) + np.abs(Y) <= P
    octagon_interior = cond1 & cond2

    # 3. Create the Plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Shade the interior of the octagon
    ax.contourf(X, Y, octagon_interior, levels=[0.5, 1], colors=['#A7C7E7'], alpha=0.7)

    # Draw the bounding circle
    circle = plt.Circle((0, 0), P, color='navy', fill=False, linestyle='--', linewidth=2)
    ax.add_patch(circle)

    # Draw the 8 infinite boundary lines
    for sign_x in [-1, 1]:
        for sign_y in [-1, 1]:
            # Group 1: +/- x +/- alpha*y = P
            y1 = (P - sign_x * x) / (sign_y * alpha)
            ax.plot(x, y1, color='tomato', linewidth=1.5, alpha=0.8)
            
            # Group 2: +/- alpha*x +/- y = P
            y2 = (P - sign_x * alpha * x) / sign_y
            ax.plot(x, y2, color='mediumseagreen', linewidth=1.5, alpha=0.8)

    # 4. Formatting and Legend
    ax.set_xlim(-P * 1.25, P * 1.25)
    ax.set_ylim(-P * 1.25, P * 1.25)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_title('8 Affine Equations Defining a Rotated Octagon', fontsize=14, pad=15)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    # Custom Legend
    legend_elements = [
        Line2D([0], [0], color='navy', linestyle='--', lw=2, label=f'Circle (Radius P={P})'),
        Line2D([0], [0], color='tomato', lw=2, label=r'$\pm x \pm (\sqrt{2}-1)y = P$'),
        Line2D([0], [0], color='mediumseagreen', lw=2, label=r'$\pm (\sqrt{2}-1)x \pm y = P$'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#A7C7E7', markersize=12, label='Octagon Interior')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.35, 1))

    plt.show()
    
if __name__ == "__main__":
    plot_octagon_and_circle()
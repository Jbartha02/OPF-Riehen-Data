# ==============================================================================
# this contains some unimportant functions
# ==============================================================================

import config
import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull


#### ----- Dummy functions for testing the FFOR algorithm logic ---- ####
def empty_iteration_function():
    """The logic of this dummy function can be used to iterate over multiple optimization directions a,b in the FFOR algorithm."""
    conf = config.Config()
    
    # run optimizations for initial directions of a,b
    pq_points = []
    for a,b in conf.optimization_dirs_init:
        p, q = _dummy_minimizer(a, b)
        pq_points.append((p, q))
    
    # compute initial convex hull after initial optimization directions
    hull = ConvexHull(pq_points)
    print(f"Initial convex hull area: {hull.volume}")
    _plot_convex_hull(pq_points)
    
    
    # optimize every direction defined by the equations of the convex hull (one for each initial direction) and iterate until convergence of the hull area
    for a,b,c in hull.equations:
        print(f"Equation: {a}*P + {b}*Q + {c} = 0")
        pq_points.extend(dummy_iterator(conf, a, b, c, pq_points))
        
    # compute final convex hull after iterating over all directions and adding new points
    hull_final = ConvexHull(pq_points)
    print(f"Final convex hull area: {hull_final.volume}")
    _plot_convex_hull(pq_points)

def _dummy_minimizer(a,b) -> tuple[float, float]:
    # this function is a placeholder
    # should contain:
    # - model definition (variables, constraints) # NOTE: maybe it can also be the same model every time, so it could be passed in the function arguments instead
    # - objective definition with the given a,b
    # - optimization
    # - result saving
    # - return the optimized p and q value
    return np.random.rand()*5, np.random.rand()*5

def dummy_iterator(conf, a,b,c, pq_points):
    """This recursive function calculates new points in the (P,Q) space by optimizing in the direction defined by a,b. For every point that increases the convex hull area by more than eta, it iterates again in the two directions adjacent to the new point, until convergence."""
    print(f"New objective: {a}*P + {b}*Q") # TODO: check signs (maybe revert to -a and -b)
    # minimize in direction a,b and get new p,q point
    p,q = _dummy_minimizer(a, b)
    new_points = [(p, q)]

    # check if the new point increases the area of the convex hull by more than eta
    hull_old = ConvexHull(pq_points)
    hull_new = ConvexHull(pq_points + [(p,q)])
    area_old = hull_old.volume
    area_new = hull_new.volume
    if (area_new - area_old) / area_old >= conf.eta_polygon_area:
        # minimize the two new directions adjacent to the new point and iterate again
        new_equations = hull_new.equations[~np.isin(hull_new.equations, hull_old.equations).all(axis=1)]
        
        for a,b,c in new_equations:
            new_points.extend(dummy_iterator(conf, a,b,c, pq_points + [(p, q)]))
    else:
        # direction is sufficiently converged
        print(f"Direction {a},{b} converged with area improvement {(area_new - area_old) / area_old:.4f} < {conf.eta_polygon_area}")

    return new_points
    
#### ----- END Dummy functions for testing the FFOR algorithm logic ---- ####




def test_convex_hull():
    # just to see what ConvexHull can do
    coordinates = [(0, 0), (0, 4), (5, 5), (6, 0)]
    
    hull1 = ConvexHull(coordinates)
    print(f"Convex hull area: {hull1.volume}")
    print(f"Convex hull vertices: {hull1.vertices}")
    print(f"Convex hull equations: {hull1.equations}")
    
    hull2 = ConvexHull(coordinates + [(-1, 2)])
    print(f"Convex hull area after adding point: {hull2.volume}")
    print(f"Convex hull vertices after adding point: {hull2.vertices}")
    
    print(f"Convex hull simplices: {hull2.simplices}")
    print(f"Convex hull equations: {hull2.equations}")
    print(f"Convex hull neighbors: {hull2.neighbors}")
    print(f"Convex hull coplanar: {hull2.coplanar}")
    
    print(f"Convex hull equations 2 not 1: {hull2.equations[~np.isin(hull2.equations, hull1.equations).all(axis=1)]}")
    
    _plot_convex_hull(coordinates + [(-1, 2)])

def _plot_convex_hull(points):
    points = np.asarray(points, dtype=float)
    hull = ConvexHull(points)
    
    # Plot the convex hull
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    
    ax.scatter(points[:, 0], points[:, 1], c='red', s=50, zorder=5)
    
    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1], 'b-', linewidth=2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Convex Hull')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.show()


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
    empty_iteration_function()
    test_convex_hull()
    plot_octagon_and_circle()
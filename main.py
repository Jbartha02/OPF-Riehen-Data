import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

import config
import functions as funcs
import dummy_functions as dummyfuncs


def main():
    """This is the main function where it all starts."""
    conf_list: list[config.Config] = [
        lambda: config.Config(
            year=2050,
            month=8,
            day=19,
            start_hour=9,
            n_timesteps=1,
            delta_t=0.5,
        ),
        #lambda: config.Config(
        #    year=2050,
        #    month=8,
        #    day=19,
        #    start_hour=9,
        #    n_timesteps=1
        #),
    ]
    
    # run each scenario in conf_list
    for conf in conf_list:
        _run_analyis(conf=conf())

    
def _run_analyis(conf: config.Config):
    """This function orchestrates the optimization of the FFOR for a single configuration conf."""
    
    # run optimizations for initial directions of a,b
    pq_flex_points = []
    for a,b in conf.optimization_dirs_init:
        p_flex, q_flex = _setup_and_minimize_model(conf=conf, a=a, b=b)
        pq_flex_points.append((p_flex, q_flex))

    # compute initial convex hull after initial optimization directions
    hull = ConvexHull(pq_flex_points)
    print(f"Initial convex hull area: {hull.volume}")
    #dummyfuncs._plot_convex_hull(pq_flex_points)
    
    # optimize every edge-normal direction, defined by the equations of the convex hull (one for each initial direction) and iterate until convergence of the hull area
    for a,b,c in hull.equations:
        print(f"Equation: {a}*P_flex + {b}*Q_flex + {c} = 0")
        pq_flex_points.extend(_maximise_edge_normal(conf=conf, a=a, b=b, c=c, pq_flex_points=pq_flex_points))
        
    # compute final convex hull after iterating over all directions and adding new points
    hull_final = ConvexHull(pq_flex_points)
    print(f"Final convex hull area: {hull_final.volume}")
    #dummyfuncs._plot_convex_hull(pq_flex_points)
    
    # export final points to csv
    df_pq_flex = pd.DataFrame(pq_flex_points, columns=["P_flex", "Q_flex"])
    df_pq_flex.to_csv(f"{conf.output_folder}/results_pq_flex_points.csv", index=False)
    
    # print final points
    print(f"Done maximizing FFOR. Found {len(pq_flex_points)} points on the FFOR:")
    for p_flex, q_flex in pq_flex_points:
        print(f"P_flex: {p_flex}, Q_flex: {q_flex}")


def _setup_and_minimize_model(conf: config.Config, a: float, b: float) -> tuple[float, float]:
    """Set up the model variables and boundary conditions. Minimize the objective given by -a*P_flex - b*Q_flex and return the optimal P_flex and Q_flex."""
    # TODO: move this function to functions.py?
    # TODO: finish implementation of result saving
    
    model = gp.Model("OPF")
    model.Params.MIPGap = 0.01  # accept 1% gap - default is 0.01%
    #model.Params.MIPFocus = 1  # focus on finding feasible solutions quickly
    model.Params.NoRelHeurWork = 10  # spend first 10s on heuristics before LP relaxation
    #model.Params.Heuristics = 0.15  # default is 0.05; Relative time spent in feasibility heuristics
    #model.Params.Presolve = 2  # aggressive presolve
    
    ### ---- Define node power variables and constraints for each time step ---- ###
    # PV
    p_pv, p_pv_flex, q_pv, q_pv_flex = funcs.define_pv_vars_and_bcs(model, conf)

    # HP
    p_hp, p_hp_flex, q_hp, q_hp_flex, t_hp = funcs.define_hp_vars_and_bcs(model, conf)

    # BESS
    p_bess_pos, p_bess_neg, p_bess_flex, q_bess, q_bess_flex, soc_bess, b_bess_charge = funcs.define_bess_vars_and_bcs(model, conf)
    
    # EV
    # TODO: add
    
    
    ### ---- Define OPV variables and constraints that connect node powers for each time step ---- ###
    # TODO: implement




    p_node = model.addVars(conf.node_metadata_df.index, conf.time_index_list, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="p_node")
    '''for t in conf.time_index_list:
        for node in conf.node_metadata_df.index:
            model.addConstr(
                p_node[node, t] == conf.p_load[node, t] + p_pv.get((node, t), 0)  + p_hp.get((node, t), 0) + conf.p_bess_base_pos[node, t] + conf.p_bess_base_neg[node, t] + p_bess_flex[node, t], name=f"p_node_balance_n{node}_t{t}"
            )'''



    #TODO: remove, just to check implementation
    p_flex = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="p_flex")
    q_flex = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="q_flex")
    for t in conf.time_index_list:
        model.addConstr(
            p_flex
            == gp.quicksum(p_pv_flex[node, t] for node in conf.node_group_dict["PV"])
            + gp.quicksum(p_hp_flex[node, t] for node in conf.node_group_dict["HP"])
            + gp.quicksum(p_bess_flex[node, t] for node in conf.node_group_dict["BESS"]),
            name=f"p_flex_balance_t{t}",
        )
        model.addConstr(
            q_flex
            == gp.quicksum(q_pv_flex[node, t] for node in conf.node_group_dict["PV"])
            + gp.quicksum(q_hp_flex[node, t] for node in conf.node_group_dict["HP"])
            + gp.quicksum(q_bess_flex[node, t] for node in conf.node_group_dict["BESS"]),
            name=f"q_flex_balance_t{t}",
        )

    obj = -a*p_flex + -b*q_flex
    model.setObjective(obj, sense=GRB.MINIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        #dummyfuncs.plot_first_10_nodes(conf, p_pv, p_pv_flex, p_hp, p_hp_flex, p_bess_pos, p_bess_neg)

        print("Optimal value:", model.objVal)
        print("Optimal p_pv:")
        for node in conf.node_group_dict["PV"][0:3]:
            for t in conf.time_index_list:
                print(f"  Node {node}, Time {t}: {p_pv[node, t].X}")
    elif model.status == GRB.INF_OR_UNBD:
        print("Model is INFEASIBLE OR UNBOUNDED — computing IIS...")
        model.computeIIS()
        # list offending constraints and variables
        for c in model.getConstrs():
            if c.IISConstr:
                print("IIS constr:", c.constrName)
        for v in model.getVars():
            if v.IISLB or v.IISUB:
                print("IIS var:", v.varName, "IISLB:", v.IISLB, "IISUB:", v.IISUB)


    print(conf.node_metadata_df)
    print(conf.p_load)
    print(conf.p_pv_base)
    print(conf.p_pv_base[2,11])
    
    #df = pd.DataFrame(conf.cop_hp)
    #df.to_csv("cop_hp.csv", index=False)
    # TODO: delete until here
    
    
    
    ### --- RESULT OUTPUT --- ###
    # define variables to output
    results_n_t_dict = {
        "p_pv": p_pv,
        "p_pv_flex": p_pv_flex,
        "q_pv": q_pv,
        "q_pv_flex": q_pv_flex,
        "p_hp": p_hp,
        "p_hp_flex": p_hp_flex,
        "q_hp": q_hp,
        "q_hp_flex": q_hp_flex,
        "t_hp": t_hp,
        "p_bess_pos": p_bess_pos,
        "p_bess_neg": p_bess_neg,
        "p_bess_flex": p_bess_flex,
        "q_bess": q_bess,
        "q_bess_flex": q_bess_flex,
        "soc_bess": soc_bess,
        "b_bess_charge": b_bess_charge,
        #"Voltage": V,
        # TODO: add EV variables
    }

    # extract and save nodal results to longtable
    df = pd.concat([funcs._extract_nodal_results_to_df(conf, var, varname) for varname, var in results_n_t_dict.items()])
    df.to_csv(f"{conf.output_folder}/results_n_t_a{a}_b{b}.csv", index=False)
    
    return p_flex.X, q_flex.X # TODO: check if correct
    
    
def _maximise_edge_normal(conf: config.Config, a: float, b: float, c: float, pq_flex_points: list[float]):
    """This recursive function calculates new points in the (P_flex,Q_flex) space by optimizing in the edge normal-direction defined by a,b. For every point that increases the convex hull area by more than eta, it iterates again in the two directions adjacent to the new point, until convergence."""
    print(f"New minimization objective: -{a}*P_flex - {b}*Q_flex") # TODO: check signs (maybe revert to -a and -b)
    # minimize in direction -a,-b and get new optimal p,q point
    p_flex,q_flex = _setup_and_minimize_model(conf=conf, a=a, b=b)
    new_points_list = [(p_flex, q_flex)]

    # check if the new point increases the area of the convex hull by more than eta
    hull_old = ConvexHull(pq_flex_points)
    hull_new = ConvexHull(pq_flex_points + [(p_flex,q_flex)])
    area_old = hull_old.volume # this is the area spanned by the points in pq_flex_points
    area_new = hull_new.volume
    if (area_new - area_old) / area_old >= conf.eta_polygon_area:
        # minimize the two new directions adjacent to the new point and iterate again
        new_equations = hull_new.equations[~np.isin(hull_new.equations, hull_old.equations).all(axis=1)]
        
        for a,b,c in new_equations:
            new_points_list.extend(_maximise_edge_normal(conf=conf, a=a, b=b, c=c, pq_flex_points=pq_flex_points+[(p_flex, q_flex)]))
    else:
        # direction is sufficiently converged
        print(f"Direction {a},{b} converged with area improvement {(area_new - area_old) / area_old:.4f} < {conf.eta_polygon_area}")

    return new_points_list


if __name__ == "__main__":
    main() 
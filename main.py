import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

import config
import functions as funcs


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
        _run_analysis(conf=conf())

    
def _run_analysis(conf: config.Config):
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
    model.Params.NoRelHeurWork = 5  # spend first xx sec on heuristics before LP relaxation
    #model.Params.Heuristics = 0.15  # default is 0.05; Relative time spent in feasibility heuristics
    #model.Params.Presolve = 2  # aggressive presolve
    
    ### ---- Define node power variables and constraints for each time step ---- ###
    p_pv, p_pv_flex, q_pv, q_pv_flex = funcs.define_pv_vars_and_bcs(model, conf)
    p_hp, p_hp_flex, q_hp, q_hp_flex, t_hp = funcs.define_hp_vars_and_bcs(model, conf)
    p_bess_pos, p_bess_neg, p_bess_flex, q_bess, q_bess_flex, soc_bess, b_bess_charge = funcs.define_bess_vars_and_bcs(model, conf)
    p_ev, p_ev_flex = funcs.define_ev_vars_and_bcs(model, conf)

    # 1) Per-unit edges
    edges_df_pu = funcs.per_unit_edges(
        conf.edges_metadata_df.copy(),
        V_base_kV=conf.V_base,
        S_base_MVA=conf.S_base
    )

    # 2) Root index (PCC = transformer node) # TODO: move this to config
    root_osmid = 97
    matches = conf.node_metadata_df.index[conf.node_metadata_df["LV_osmid"] == root_osmid].tolist()
    if not matches:
        raise ValueError(f"Root osmid {root_osmid} not found in node_metadata_df. Check LV_osmid column.")
    root_idx = matches[0]
    print(f"Root node: osmid={root_osmid}, matrix_idx={root_idx}")


    # 3) Radial tree + LinDistFlow data
    N = conf.node_metadata_df.shape[0]
    T = len(conf.time_index_list)
    tree = funcs.build_radial_tree_from_edges(n_nodes=N, edges_df_pu=edges_df_pu, root_idx=root_idx)
    ldf_data = funcs.assemble_lindistflow_data(tree, T=T, V_min=0.95, V_max=1.05)

    # 4) Nodal injection expressions (p.u.) — base + flex variables
    Sbase_kW = conf.S_base * 1000.0
    P_inj_expr = {}
    Q_inj_expr = {}

    for i in conf.node_metadata_df.index:
        for tcol in enumerate(conf.time_index_list):

            # --- Active power (kW) ---
            expr_P = gp.LinExpr(float(conf.p_load[i, tcol]))   # fixed load (negative)

            pv = p_pv.get((i, tcol), None)
            hp = p_hp.get((i, tcol), None)
            bp = p_bess_pos.get((i, tcol), None)
            bn = p_bess_neg.get((i, tcol), None)
            ev = p_ev.get((i, tcol), None)

            if pv is not None: expr_P += pv          # base + p_pv_flex
            if hp is not None: expr_P += hp          # base + p_hp_flex (<=0)
            if bp is not None: expr_P += bp          # charging (>=0, withdrawal)
            if bn is not None: expr_P += bn          # discharging (<=0, injection)
            if ev is not None: expr_P += ev          # EV charging (positive profile → subtract = withdrawal)

            # --- Reactive power (kVAr) ---
            expr_Q = gp.LinExpr(0.0)

            qpv = q_pv.get((i, tcol), None)
            qhp = q_hp.get((i, tcol), None)
            qbe = q_bess.get((i, tcol), None)

            if qpv is not None: expr_Q += qpv
            if qhp is not None: expr_Q += qhp
            if qbe is not None: expr_Q += qbe
            
            # TODO: simpler:
            #expr_P = conf.p_load[i, tcol] + p_pv.get((i, tcol), 0) + p_hp.get((i, tcol), 0) + p_bess_pos.get((i, tcol), 0) + p_bess_neg.get((i, tcol), 0) + p_ev.get((i, tcol), 0)
            #expr_Q = q_pv.get((i, tcol), 0) + q_hp.get((i, tcol), 0) + q_bess.get((i, tcol), 0)

            # Scale to p.u.
            P_inj_expr[(i, tcol)] = (1.0 / Sbase_kW) * expr_P # TODO pu base not clear of kW or MW
            Q_inj_expr[(i, tcol)] = (1.0 / Sbase_kW) * expr_Q

    # 5) LinDistFlow grid constraints
    V, Pf, Qf = funcs.add_lindistflow_to_model(
        model, ldf_data, P_inj_expr, Q_inj_expr,
        fix_root_voltage=1.0,
        use_soc_lines=True
    )

    # 6) Objective: maximize total flexibility at PCC
    # (analogous to dummy, but now grid-constrained)
    root = ldf_data["root"]
    children_root = ldf_data["incidence_out"][root]

    # Total flexibility variables
    p_flex_total = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="p_flex_total")
    q_flex_total = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="q_flex_total")

    for t in conf.time_index_list:
        model.addConstr(
            p_flex_total
            == gp.quicksum(
                p_pv_flex.get((n, t), 0)
                + p_hp_flex.get((n, t), 0)
                + p_bess_flex.get((n, t), 0)
                + p_ev_flex.get((n,t), 0)
                for n in conf.node_group_dict["ALL NODES"]
            ),
            name=f"p_flex_total[{t}]"
        )
        model.addConstr(
            q_flex_total
            == gp.quicksum(
                q_pv_flex.get((n, t), 0)
                + q_hp_flex.get((n, t), 0)
                + q_bess_flex.get((n, t), 0)
                for n in conf.node_group_dict["ALL NODES"]),
            name=f"q_flex_total[{t}]"
        )

    # Maximize total flexibility (minimize negative flex)
    # Scale to p.u. to keep objective dimensionally consistent with grid vars
    model.setObjective(
        -a * p_flex_total / Sbase_kW
        -b * q_flex_total / Sbase_kW,
        GRB.MINIMIZE
    )

    # 7) Solve
    model.Params.OutputFlag = 1
    model.optimize()

    # 8) Results
    if model.status == GRB.OPTIMAL:
        print(f"\nOptimal flex (p.u.): {-model.objVal:.4f}")
        print(f"Optimal flex (kW):   {-model.objVal * Sbase_kW:.2f}")

        # Per-time flex summary
        print("\n  t  | p_flex_total (kW) | q_flex_total (kVAr) | Ppcc (kW)  | Qpcc (kVAr)")
        print("  " + "-"*75)
        for idx_t in range(T):
            pft = p_flex_total.X
            qft = q_flex_total.X
            ppcc = sum(Pf[root, j, idx_t].X for j in children_root) * Sbase_kW
            qpcc = sum(Qf[root, j, idx_t].X for j in children_root) * Sbase_kW
            print(f"  {conf.time_index_list[idx_t]:2d} | {pft:17.2f} | {qft:19.2f} | {ppcc:10.2f} | {qpcc:10.2f}")

        # Voltage summary: min/max per time step
        V_arr = np.array([[V[i, tt].X for tt in range(T)] for i in range(N)])
        print("\nVoltage min/max per time step (p.u.):")
        for tt in range(T):
            print(f"  t={conf.time_index_list[tt]:2d}: min={V_arr[:,tt].min():.4f}  max={V_arr[:,tt].max():.4f}")

    elif model.status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE):
        print("Infeasible/unbounded — computing IIS...")
        model.computeIIS()
        for c in model.getConstrs():
            if c.IISConstr:
                print("IIS constr:", c.constrName)
        for v in model.getVars():
            if v.IISLB or v.IISUB:
                print("IIS var:", v.varName, "IISLB:", v.IISLB, "IISUB:", v.IISUB)


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
        "p_ev": p_ev,
        "p_ev_flex": p_ev_flex,
        "Voltage": V,
    }
    results_edge_t_dict = {
        "P_edge": Pf, # TODO: currently false time indexes for export. adapt to conf.time_index_list in var definition
        "Q_edge": Qf,
    }

    # extract and save nodal results to longtable
    df = pd.concat([funcs._extract_nodal_results_to_df(conf, var, varname) for varname, var in results_n_t_dict.items()])
    df.to_csv(f"{conf.output_folder}/results_node_t_a{a}_b{b}.csv", index=False)

    # extract and save edge results to longtable
    df = pd.concat([funcs._extract_edge_results_to_df(conf, var, varname) for varname, var in results_edge_t_dict.items()])
    df.to_csv(f"{conf.output_folder}/results_edge_t_a{a}_b{b}.csv", index=False)


    return p_flex_total.X, q_flex_total.X
    
    
def _maximise_edge_normal(conf: config.Config, a: float, b: float, c: float, pq_flex_points: list[tuple[float, float]]) -> list[tuple[float, float]]:
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

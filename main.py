import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import functions as funcs
import dummy_functions as dummyfuncs


def main():
    conf = config.Config()
    model = gp.Model("OPF_with_PF")

    # Device variables and local constraints
    p_pv, p_pv_flex, q_pv, q_pv_flex = funcs.define_pv_vars_and_bcs(model, conf)
    p_hp, p_hp_flex, q_hp, q_hp_flex, t_hp = funcs.define_hp_vars_and_bcs(model, conf)
    p_bess_pos, p_bess_neg, p_bess_flex, q_bess, q_bess_flex, soc_bess, b_bess_charge = funcs.define_bess_vars_and_bcs(model, conf)
    # EV: TODO

    # 1) Per-unit edges
    edges_df_pu = funcs.per_unit_edges(
        conf.edges_metadata_df.copy(),
        V_base_kV=conf.V_base,
        S_base_MVA=conf.S_base
    )

    # 2) Root index (PCC = transformer node)
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
    # TODO: EV needs to be added
    Sbase_kW = conf.S_base * 1000.0
    P_inj_expr = {}
    Q_inj_expr = {}

    for i in conf.node_metadata_df.index:
        for idx_t, tcol in enumerate(conf.time_index_list):

            # --- Active power (kW) ---
            expr_P = gp.LinExpr(float(conf.p_load[i, tcol]))   # fixed load (negative)

            pv = p_pv.get((i, tcol), None)
            hp = p_hp.get((i, tcol), None)
            bp = p_bess_pos.get((i, tcol), None)
            bn = p_bess_neg.get((i, tcol), None)

            if pv is not None: expr_P += pv          # base + p_pv_flex
            if hp is not None: expr_P += hp          # base + p_hp_flex (<=0)
            if bp is not None: expr_P += bp          # charging (>=0, withdrawal)
            if bn is not None: expr_P += bn          # discharging (<=0, injection)

            # --- Reactive power (kVAr) ---
            expr_Q = gp.LinExpr(0.0)

            qpv = q_pv.get((i, tcol), None)
            qhp = q_hp.get((i, tcol), None)
            qbe = q_bess.get((i, tcol), None)

            if qpv is not None: expr_Q += qpv
            if qhp is not None: expr_Q += qhp
            if qbe is not None: expr_Q += qbe

            # Scale to p.u.
            P_inj_expr[(i, idx_t)] = (1.0 / Sbase_kW) * expr_P # TODO pu base not clear of kW or MW
            Q_inj_expr[(i, idx_t)] = (1.0 / Sbase_kW) * expr_Q

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

    # Total flex per time: sum of all device flex variables (kW, kVAr)
    # same quantities as in the dummy, but now constrained by the grid.
    p_flex_total = model.addVars(range(T), lb=-GRB.INFINITY, name="p_flex_total")
    q_flex_total = model.addVars(range(T), lb=-GRB.INFINITY, name="q_flex_total")

    for idx_t, tcol in enumerate(conf.time_index_list):
        model.addConstr(
            p_flex_total[idx_t]
            == gp.quicksum(p_pv_flex[n, tcol]  for n in conf.node_group_dict["PV"])
            +  gp.quicksum(p_hp_flex[n, tcol]  for n in conf.node_group_dict["HP"])
            +  gp.quicksum(p_bess_flex[n, tcol] for n in conf.node_group_dict["BESS"]),
            name=f"p_flex_total[{idx_t}]"
        )
        model.addConstr(
            q_flex_total[idx_t]
            == gp.quicksum(q_pv_flex[n, tcol]  for n in conf.node_group_dict["PV"])
            +  gp.quicksum(q_hp_flex[n, tcol]  for n in conf.node_group_dict["HP"])
            +  gp.quicksum(q_bess_flex[n, tcol] for n in conf.node_group_dict["BESS"]),
            name=f"q_flex_total[{idx_t}]"
        )

    alpha = 1.0   # weight on active flex
    beta  = 1.0   # weight on reactive flex

    # Maximize total flexibility (minimize negative flex)
    # Scale to p.u. to keep objective dimensionally consistent with grid vars
    model.setObjective(
        -alpha * gp.quicksum(p_flex_total[tt] for tt in range(T)) / Sbase_kW
        - beta  * gp.quicksum(q_flex_total[tt] for tt in range(T)) / Sbase_kW,
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
            pft = p_flex_total[idx_t].X
            qft = q_flex_total[idx_t].X
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
                print("IIS var:", v.varName)


if __name__ == "__main__":
    main()

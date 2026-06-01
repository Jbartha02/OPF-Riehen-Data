import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
from collections import deque, defaultdict
from typing import Dict, Any, List, Tuple, Optional

import config



def define_pv_vars_and_bcs(model: gp.Model, conf: config.Config) -> tuple[gp.tupledict, gp.tupledict]:
    """Define all pv variables and their relationships (bounds and balance constraints) in the optimization model."""
    # Initialize variables
    p_pv = model.addVars(conf.node_group_dict["PV"], conf.time_index_list, lb=0, ub=np.nanmax(conf.p_pv_ub), vtype=GRB.CONTINUOUS, name="p_pv") # TODO: update bounds
    p_pv_flex = model.addVars(conf.node_group_dict["PV"], conf.time_index_list, lb=-np.nanmax(conf.p_pv_ub), ub=np.nanmax(conf.p_pv_ub), vtype=GRB.CONTINUOUS, name="p_pv_flex") #TODO: update bounds
    q_pv = model.addVars(conf.node_group_dict["PV"], conf.time_index_list, lb=-conf.pv_max_q_p_ratio*np.nanmax(conf.p_pv_ub), ub=conf.pv_max_q_p_ratio*np.nanmax(conf.p_pv_ub), vtype=GRB.CONTINUOUS, name="q_pv") #TODO: set this to flex
    q_pv_flex = q_pv # only because q_pv_base is zero!
    
    # Define constraints
    for node in conf.node_group_dict["PV"]:
        for t in conf.time_index_list:
            model.addConstr(
                p_pv[node, t] == conf.p_pv_base[node, t] + p_pv_flex[node, t],
                name=f"pv_balance_n{node}_t{t}",
            )
            model.addConstr(
                p_pv[node, t] <= conf.p_pv_ub[node, t], name=f"pv_ub_n{node}_t{t}"
            )
            model.addConstr(
                p_pv[node, t] >= conf.p_pv_lb[node, t], name=f"pv_lb_n{node}_t{t}"
            )
            model.addConstr(
                q_pv[node, t] <= conf.pv_max_q_p_ratio * p_pv[node, t], name=f"q_pv_ub_n{node}_t{t}"
            )
            model.addConstr(
                q_pv[node, t] >= -conf.pv_max_q_p_ratio * p_pv[node, t], name=f"q_pv_lb_n{node}_t{t}"
            )
    return p_pv, p_pv_flex, q_pv, q_pv_flex


def define_hp_vars_and_bcs(model: gp.Model, conf: config.Config) -> tuple[gp.tupledict, gp.tupledict]:
    """Define all hp variables and their relationships (bounds and balance constraints) in the optimization model."""
    # Initialize variables
    p_hp = model.addVars(conf.node_group_dict["HP"], conf.time_index_list, lb=np.nanmin(conf.p_hp_base), ub=0, vtype=GRB.CONTINUOUS, name="p_hp") # TODO: update bounds
    p_hp_flex = model.addVars(conf.node_group_dict["HP"], conf.time_index_list, lb=np.nanmin(conf.p_hp_base), ub=-np.nanmin(conf.p_hp_base), vtype=GRB.CONTINUOUS, name="p_hp_flex") #TODO: update bounds
    q_hp = model.addVars(conf.node_group_dict["HP"], conf.time_index_list, lb=conf.hp_q_p_ratio*np.nanmin(conf.p_hp_base), ub=0, vtype=GRB.CONTINUOUS, name="q_hp") #TODO: remove this (can be replaced in equations)
    q_hp_flex = model.addVars(conf.node_group_dict["HP"], conf.time_index_list, lb=conf.hp_q_p_ratio*np.nanmin(conf.p_hp_base), ub=-conf.hp_q_p_ratio*np.nanmin(conf.p_hp_base), vtype=GRB.CONTINUOUS, name="q_hp_flex") #TODO: update bounds
    t_hp = model.addVars(conf.node_group_dict["HP"], conf.time_index_list, lb=np.nanmin(conf.t_hp_lb), ub=np.nanmax(conf.t_hp_ub), vtype=GRB.CONTINUOUS, name="t_hp") #TODO: update bounds
    
    # Define constraints
    for node in conf.node_group_dict["HP"]:
        for t in conf.time_index_list:
            if t == conf.time_index_list[0]:
                model.addConstr(
                    -p_hp[node, t] * conf.cop_hp[node, t]
                    == conf.node_metadata_df.loc[node, "HP_Thermal_capacitance_KWh/K"]
                    * (t_hp[node, t] - conf.t_hp_base[node, t+23])
                    + conf.node_metadata_df.loc[node, "HP_Thermal_conductivity_kW/K"]
                    * (t_hp[node, t] - conf.t_outdoor[t]),
                    name=f"hp_temp_balance_n{node}_t{t}",
                )
            else:
                model.addConstr(
                    -p_hp[node, t] * conf.cop_hp[node, t]
                    == conf.node_metadata_df.loc[node, "HP_Thermal_capacitance_KWh/K"]
                    * (t_hp[node, t] - t_hp[node, t-1])
                    + conf.node_metadata_df.loc[node, "HP_Thermal_conductivity_kW/K"]
                    * (t_hp[node, t] - conf.t_outdoor[t]),
                    name=f"hp_temp_balance_n{node}_t{t}",
                )
            model.addConstr(
                p_hp[node, t] == conf.p_hp_base[node, t] + p_hp_flex[node, t],
                name=f"hp_balance_n{node}_t{t}",
            )
            model.addConstr(
                p_hp[node, t] >= -conf.node_metadata_df.loc[node, "HP_Nominal_power_kW"],
                name=f"hp_min_n{node}_t{t}",
            )
            model.addConstr(p_hp[node, t] <= 0, name=f"hp_max_n{node}_t{t}")
            model.addConstr(
                q_hp[node, t] == conf.hp_q_p_ratio * p_hp[node, t], name=f"q_hp_ratio_n{node}_t{t}" # TODO: merge with q_hp_base constraint and remove q_hp
            )
            model.addConstr(
                q_hp[node, t] == conf.q_hp_base[node, t] + q_hp_flex[node, t], name=f"q_hp_base_n{node}_t{t}"
            )
            model.addConstr(
                t_hp[node, t] <= conf.t_hp_ub[node, t], name=f"t_hp_ub_n{node}_t{t}"
            )
            model.addConstr(
                t_hp[node, t] >= conf.t_hp_lb[node, t], name=f"t_hp_lb_n{node}_t{t}"
            )
    return p_hp, p_hp_flex, q_hp, q_hp_flex, t_hp


def define_bess_vars_and_bcs(model: gp.Model, conf: config.Config) -> tuple[gp.tupledict, gp.tupledict]:
    """Define all bess variables and their relationships (bounds and balance constraints) in the optimization model."""
    # Initialize variables
    p_bess_pos = model.addVars(conf.node_group_dict["BESS"], conf.time_index_list, lb=0, ub=np.nanmax(conf.node_metadata_df["BESS_Nominal_power_kW"].to_numpy()), vtype=GRB.CONTINUOUS, name="p_bess_pos") # TODO: update bounds
    p_bess_neg = model.addVars(conf.node_group_dict["BESS"], conf.time_index_list, lb=-np.nanmax(conf.node_metadata_df["BESS_Nominal_power_kW"].to_numpy()), ub=0, vtype=GRB.CONTINUOUS, name="p_bess_neg") # TODO: update bounds
    p_bess_flex = model.addVars(conf.node_group_dict["BESS"], conf.time_index_list, lb=-np.nanmax(conf.node_metadata_df["BESS_Nominal_power_kW"].to_numpy()), ub=np.nanmax(conf.node_metadata_df["BESS_Nominal_power_kW"].to_numpy()), vtype=GRB.CONTINUOUS, name="p_bess_flex") #TODO: update bounds
    q_bess = model.addVars(conf.node_group_dict["BESS"], conf.time_index_list, lb=-np.nanmax(conf.node_metadata_df["BESS_Nominal_power_kW"].to_numpy()), ub=np.nanmax(conf.node_metadata_df["BESS_Nominal_power_kW"].to_numpy()), vtype=GRB.CONTINUOUS, name="q_bess") #TODO: update bounds
    q_bess_flex = q_bess # only because q_bess_base is zero!
    soc_bess = model.addVars(conf.node_group_dict["BESS"], conf.time_index_list, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="soc_bess") # TODO: update bounds
    b_bess_charge = model.addVars(conf.node_group_dict["BESS"], conf.time_index_list, vtype=GRB.BINARY, name="b_bess_charge")
    
    # Define constraints
    for node in conf.node_group_dict["BESS"]:
        for t in conf.time_index_list:
            if t == conf.time_index_list[0]:
                model.addConstr(
                    soc_bess[node, t] == conf.soc_bess_base[node, t+23] - (conf.p_bess_base_pos[node, t+23] * conf.node_metadata_df.loc[node, "BESS_Charging_efficiency"] + conf.p_bess_base_neg[node, t-1] / conf.node_metadata_df.loc[node, "BESS_Discharging_efficiency"]) / conf.node_metadata_df.loc[node, "BESS_Battery_capacity_kWh"], # TODO: implement delta_t?
                    name=f"soc_bess_balance_n{node}_t{t}",
                )
            else:
                model.addConstr(
                    soc_bess[node, t] == soc_bess[node, t-1] - (p_bess_pos[node, t-1] * conf.node_metadata_df.loc[node, "BESS_Charging_efficiency"] + p_bess_neg[node, t-1] / conf.node_metadata_df.loc[node, "BESS_Discharging_efficiency"]) / conf.node_metadata_df.loc[node, "BESS_Battery_capacity_kWh"], # TODO: implement delta_t?
                    name=f"soc_bess_balance_n{node}_t{t}",
                )
            model.addConstr(
                p_bess_pos[node, t] + p_bess_neg[node, t] == conf.p_bess_base_pos[node, t] + conf.p_bess_base_neg[node, t] + p_bess_flex[node, t], name=f"p_bess_balance_n{node}_t{t}"
            )
            model.addConstr(
                soc_bess[node, t] <= conf.soc_bess_ub[node, t], name=f"soc_bess_ub_n{node}_t{t}"
            )
            model.addConstr(
                soc_bess[node, t] >= conf.soc_bess_lb[node, t], name=f"soc_bess_lb_n{node}_t{t}"
            )
            model.addConstr(
                p_bess_pos[node, t] <= conf.node_metadata_df.loc[node, "BESS_Nominal_power_kW"] * b_bess_charge[node, t], name=f"p_bess_pos_charge_n{node}_t{t}"
            )
            model.addConstr(
                p_bess_neg[node, t] >= -conf.node_metadata_df.loc[node, "BESS_Nominal_power_kW"] * (1 - b_bess_charge[node, t]), name=f"p_bess_neg_discharge_n{node}_t{t}"
            )
            model.addConstr(
                p_bess_pos[node, t] >= 0, name=f"p_bess_pos_lb_n{node}_t{t}"
            )
            model.addConstr(
                p_bess_neg[node, t] <= 0, name=f"p_bess_neg_ub_n{node}_t{t}"
            )
            for a, b in conf.bess_power_octagon_approximation:
                model.addConstr(
                    a * (p_bess_pos[node, t] + p_bess_neg[node, t]) + b * q_bess[node, t] <= conf.node_metadata_df.loc[node, "BESS_Nominal_power_kW"], name=f"bess_power_octagon_n{node}_t{t}_a{a}_b{b}"
                )
                
    return p_bess_pos, p_bess_neg, p_bess_flex, q_bess, q_bess_flex, soc_bess, b_bess_charge

def per_unit_edges(edges_df: pd.DataFrame, V_base_kV: float, S_base_MVA: float) -> pd.DataFrame:
    """
    Convert line parameters to per-unit based on V_base (kV, line-to-line) and S_base (MVA):
      r_ohm, x_ohm, b_Siemens, s_nom_MVA -> r_pu, x_pu, b_pu, s_nom_pu
    Expected columns in edges_df:
      - 'u_idx', 'v_idx' (int indices of nodes)
      - 'r' (Ohm), 'x' (Ohm), 'b' (Siemens), 's_nom' (MVA)
      - optional: other metadata retained
    Returns a copy with added columns:
      - 'r_pu', 'x_pu', 'b_pu', 's_nom_pu'
      - 'Z_base_ohm', 'I_base_kA' (same for all rows, kept for convenience)
    """
    required = {'r','x','b','s_nom'}
    missing = required - set(edges_df.columns)
    if missing:
        raise ValueError(f"edges_df is missing column {missing}")
    
    Z_base_kohm = (V_base_kV**2) / S_base_MVA               # kOhm
    Z_base_ohm = Z_base_kohm * 1e3                          # Ohm
    I_base_kA = S_base_MVA / (np.sqrt(3) * V_base_kV)       # kA
    out = edges_df.copy()

    out['r_pu'] = out['r'] / Z_base_ohm
    out['x_pu'] = out['x'] / Z_base_ohm

    out['b_pu'] = out['b'] * Z_base_ohm

    out['s_nom_pu'] = out['s_nom'] / S_base_MVA

    # helper
    out['Z_base_ohm'] = Z_base_ohm
    out['I_base_kA'] = I_base_kA

    # plausiblity check
    with np.errstate(invalid='ignore'):
        if (out['r_pu'] < 0).any() or (out['x_pu'] < 0).any():
            print("WARNING: negativ r_pu/x_pu")
        if (out['s_nom_pu'] <= 0).any():
            print("WARNING: negative s_nom_pu")
    return out

def build_radial_tree_from_edges(
    n_nodes: int,
    edges_df_pu: pd.DataFrame,
    root_idx: int,
    prefer_small_impedance: bool = True
) -> Dict[str, Any]:
    """
    Build a radial tree with parent/children from per-unit edge data.
    Select smallest |z| per (u,v) if parallel edges exist.
    Requires columns: 'u_idx','v_idx','r_pu','x_pu','s_nom_pu'
    """
    def ekey(a: int, b: int) -> Tuple[int,int]:
        return (a, b) if a < b else (b, a)

    best: Dict[Tuple[int,int], Dict[str, float]] = {}
    for _, row in edges_df_pu.iterrows():
        u, v = int(row["u_idx"]), int(row["v_idx"])
        rpu, xpu = float(row["r_pu"]), float(row["x_pu"])
        smax = float(row.get("s_nom_pu", np.inf))
        k = ekey(u, v)
        if k not in best:
            best[k] = {"r_pu": rpu, "x_pu": xpu, "s_nom_pu": smax}
        else:
            if prefer_small_impedance:
                old = best[k]
                if (rpu*rpu + xpu*xpu) < (old["r_pu"]**2 + old["x_pu"]**2):
                    best[k] = {"r_pu": rpu, "x_pu": xpu, "s_nom_pu": smax}

    # Undirected adjacency
    adj = defaultdict(list)
    for (a, b), _pars in best.items():
        adj[a].append(b)
        adj[b].append(a)

    parent = [-1]*n_nodes
    children: List[List[int]] = [[] for _ in range(n_nodes)]
    visited = set([root_idx])
    order_topo: List[int] = []
    q = deque([root_idx])

    while q:
        u = q.popleft()
        order_topo.append(u)
        for v in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                parent[v] = u
                children[u].append(v)
                q.append(v)

    tree_edges: List[Tuple[int,int]] = []
    eparams: Dict[Tuple[int,int], Dict[str,float]] = {}
    for v in range(n_nodes):
        u = parent[v]
        if u >= 0:
            a, b = (u, v) if u < v else (v, u)
            p = best[(a, b)]
            tree_edges.append((u, v))
            eparams[(u, v)] = dict(p)

    inc_out = {i: list(children[i]) for i in range(n_nodes)}
    inc_in = {i: (parent[i] if parent[i] >= 0 else None) for i in range(n_nodes)}

    return {
        "parent": parent,
        "children": children,
        "order_topo": order_topo,
        "tree_edges": tree_edges,
        "edge_params": eparams,
        "incidence_out": inc_out,
        "incidence_in": inc_in,
    }

def assemble_lindistflow_data(tree: Dict[str, Any], T: int, V_min: float = 0.95, V_max: float = 1.05) -> Dict[str, Any]:
    """
    Prepare LinDistFlow index sets and params from radial tree.
    """
    parent = tree["parent"]
    order_topo = tree["order_topo"]
    tree_edges: List[Tuple[int,int]] = tree["tree_edges"]
    edge_params = tree["edge_params"]

    N = len(parent)
    nodes = list(range(N))
    root = order_topo[0]

    r = {}
    x = {}
    smax = {}
    for (i, j) in tree_edges:
        pars = edge_params[(i, j)]
        r[(i, j)] = float(pars["r_pu"])
        x[(i, j)] = float(pars["x_pu"])
        smax[(i, j)] = float(pars.get("s_nom_pu", np.inf))

    return {
        "N": N,
        "T": T,
        "root": root,
        "nodes": nodes,
        "edges": tree_edges,
        "r": r,
        "x": x,
        "smax": smax,
        "V_min": V_min,
        "V_max": V_max,
        "incidence_in": tree["incidence_in"],
        "incidence_out": tree["incidence_out"],
    }

def add_lindistflow_to_model(
    model: gp.Model,
    data: Dict[str, Any],
    P_inj: Dict[Tuple[int,int], float],
    Q_inj: Dict[Tuple[int,int], float],
    fix_root_voltage: Optional[float] = 1.0,
    use_soc_lines: bool = True
) -> Tuple[gp.tupledict, gp.tupledict, gp.tupledict]:
    """
    Add LinDistFlow variables and constraints to an existing gurobipy model.

    Returns (V, P, Q) variable dicts.
    """
    N = data["N"]
    Tn = data["T"]
    root = data["root"]
    nodes: List[int] = data["nodes"]
    edges: List[Tuple[int,int]] = data["edges"]
    r = data["r"]
    x = data["x"]
    smax = data["smax"]
    V_min = data["V_min"]
    V_max = data["V_max"]
    inc_in = data["incidence_in"]
    inc_out = data["incidence_out"]

    # Variables
    V = model.addVars(nodes, data["T"], name="V", lb=V_min, ub=V_max)
    P = model.addVars(edges, data["T"], name="P", lb=-GRB.INFINITY, ub=GRB.INFINITY)
    Q = model.addVars(edges, data["T"], name="Q", lb=-GRB.INFINITY, ub=GRB.INFINITY)

    # Fix root voltage if desired
    if fix_root_voltage is not None:
        for t in range(Tn):
            model.addConstr(V[root, t] == float(fix_root_voltage), name=f"V_root[{t}]")

    # Voltage drop along edges
    for (i, j) in edges:
        rij, xij = r[(i, j)], x[(i, j)]
        for t in range(Tn):
            model.addConstr(
                V[j, t] == V[i, t] - (rij * P[i, j, t] + xij * Q[i, j, t]),
                name=f"volt_drop[{i},{j},{t}]"
            )

    # KCL at nodes
    for i in nodes:
        parent_node = inc_in[i]
        childs = inc_out[i]
        for t in range(Tn):
            # Active
            expr_in_P = gp.LinExpr(0.0)
            if parent_node is not None:
                expr_in_P += P[parent_node, i, t]
            expr_out_P = gp.quicksum(P[i, k, t] for k in childs)

            # --- FIX: handle both float and LinExpr ---
            inj_P = P_inj.get((i, t), 0.0)
            model.addConstr(
                expr_in_P - expr_out_P + inj_P == 0.0,
                name=f"kclP[{i},{t}]"            )

            # Reactive
            expr_in_Q = gp.LinExpr(0.0)
            if parent_node is not None:
                expr_in_Q += Q[parent_node, i, t]
            expr_out_Q = gp.quicksum(Q[i, k, t] for k in childs)

            inj_Q = Q_inj.get((i, t), 0.0)
            model.addConstr(
                expr_in_Q - expr_out_Q + inj_Q == 0.0,
                name=f"kclQ[{i},{t}]"            )

    # Line S limits
    if use_soc_lines:
        for (i, j) in edges:
            Smax = smax[(i, j)]
            if np.isfinite(Smax) and Smax > 0:
                for t in range(Tn):
                    model.addQConstr(
                        P[i, j, t] * P[i, j, t] + Q[i, j, t] * Q[i, j, t] <= Smax * Smax,
                        name=f"S_limit[{i},{j},{t}]"
                    )

    return V, P, Q

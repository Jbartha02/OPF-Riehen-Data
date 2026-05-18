import gurobipy as gp
from gurobipy import GRB
import numpy as np

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
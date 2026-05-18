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
    
    model = gp.Model("OPF")
    
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

    obj = -p_flex + -q_flex
    model.setObjective(obj, sense=GRB.MINIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        dummyfuncs.plot_first_10_nodes(conf, p_pv, p_pv_flex, p_hp, p_hp_flex, p_bess_pos, p_bess_neg)

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
    
    
    


if __name__ == "__main__":
    main() 
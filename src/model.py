import gurobipy as gp
from gurobipy import GRB
import os
from data_loader import *

def build_hpp_model(I_idx, J_idx, K_idx, C, Q, c, p, R, gamma):
    """
    Build the High Point Problem (HPP) using lists of lists.
    """
    m = gp.Model("Quarantine_Location_HPP")
    
    # --- UPPER LEVEL VARIABLES (Government) ---
    x = m.addVars(I_idx, vtype=GRB.BINARY, name="x")
    y = m.addVars(I_idx, J_idx, vtype=GRB.BINARY, name="y")
    T = m.addVars(I_idx, vtype=GRB.CONTINUOUS, lb=0.0, name="T")

    # --- LOWER LEVEL VARIABLES (Assignment/Travelers) ---
    z = m.addVars(I_idx, J_idx, K_idx, K_idx, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="z")
    
    # Auxiliary variables for user equilibrium (UE) condition
    r = m.addVars(J_idx, K_idx, vtype=GRB.BINARY, name="r")
    B = m.addVars(I_idx, J_idx, K_idx, vtype=GRB.CONTINUOUS, lb=0.0, name="B")
    u = m.addVars(J_idx, K_idx, vtype=GRB.CONTINUOUS, name="u")
    v = m.addVars(J_idx, K_idx, K_idx, vtype=GRB.BINARY, name="v")
    delta = m.addVars(I_idx, vtype=GRB.BINARY, name="delta")

    # --- OBJECTIVE FUNCTION (Eq. 36) ---
    obj_subsidy = gp.quicksum(T[i] for i in I_idx)
    
    obj_assignment = gp.quicksum(c[i][j] * Q[j][k] * z[i, j, k, w] 
                                 for i in I_idx for j in J_idx for k in K_idx for w in K_idx)
    
    obj_penalty = gamma * gp.quicksum(Q[j][k] * z[i, j, k, w] 
                                      for i in I_idx for j in J_idx for k in K_idx for w in K_idx if k != w)
    
    m.setObjective(obj_subsidy + obj_assignment + obj_penalty, GRB.MINIMIZE)

    # --- FEASIBILITY CONSTRAINTS (Eq 3 - 6) ---
    for i in I_idx:
        m.addConstr(gp.quicksum(y[i, j] for j in J_idx) == x[i], name=f"Alloc_Limit_{i}")

    for j in J_idx:
        # Total capacity assigned to node j
        m.addConstr(gp.quicksum(C[i][w] * y[i, j] for i in I_idx for w in K_idx) >= 
                    gp.quicksum(Q[j][k] for k in K_idx), name=f"Cap_Node_{j}")

    for j in J_idx:
        for k in K_idx:
            m.addConstr(gp.quicksum(z[i, j, k, w] for i in I_idx for w in K_idx) == 1, name=f"Assign_Demand_{j}_{k}")

    for i in I_idx:
        for j in J_idx:
            for w in K_idx:
                # Total demand of type w at hotel i cannot exceed physical capacity
                m.addConstr(gp.quicksum(Q[j][k] * z[i, j, k, w] for k in K_idx) <= 
                            C[i][w] * y[i, j], name=f"Cap_Hotel_{i}_{j}_{w}")

    # --- RELAXED USER EQUILIBRIUM CONSTRAINTS (Eq 24 - 28) ---
    for j in J_idx:
        for k in K_idx:
            m.addConstr(gp.quicksum(z[i, j, k, w] for i in I_idx for w in K_idx if w != k) <= r[j, k], name=f"UE_r_{j}_{k}")

    for j in J_idx:
        for w in K_idx:
            m.addConstr(gp.quicksum(B[i, j, w] for i in I_idx) <= 
                        sum(C[i][w] for i in I_idx) * (1 - r[j, w]), name=f"UE_B_sum_{j}_{w}")

    for i in I_idx:
        for j in J_idx:
            for w in K_idx:
                m.addConstr(C[i][w] * y[i, j] - gp.quicksum(Q[j][k] * z[i, j, k, w] for k in K_idx) <= B[i, j, w], name=f"UE_B_{i}_{j}_{w}")

    for j in J_idx:
        for k in K_idx:
            for w in K_idx:
                if k != w:
                    m.addConstr(u[j, k] - u[j, w] <= (1 - v[j, k, w]) * len(K_idx) - 1, name=f"UE_u_{j}_{k}_{w}")
                    m.addConstr(gp.quicksum(z[i, j, k, w] for i in I_idx) <= v[j, k, w], name=f"UE_v_{j}_{k}_{w}")

    # --- CONTRACTING COST LINEARIZATION (Eq 29 - 32) ---
    for i in I_idx:
        rev_expr = gp.quicksum(p[i][w] * Q[j][k] * z[i, j, k, w] for j in J_idx for k in K_idx for w in K_idx)
        full_rev_cap = sum(C[i][w] * p[i][w] for w in K_idx)
        
        m.addConstr(T[i] <= R[i] - rev_expr + delta[i] * (full_rev_cap - R[i]), name=f"T_lin1_{i}")
        m.addConstr(T[i] >= R[i] * x[i] - rev_expr, name=f"T_lin2_{i}")
        m.addConstr(T[i] <= R[i] * x[i], name=f"T_lin3_{i}")
        m.addConstr(T[i] <= (1 - delta[i]) * R[i], name=f"T_lin4_{i}")

    return m, x, y, z, T

def solve_hpp(file_id="16"):
    Q, C, c, p, R, gamma = get_data_for_model("1.xlsx", sheet_index=0)

    # Define final indices
    I_idx = list(range(len(R)))
    J_idx = list(range(len(Q)))
    K_idx = list(range(len(Q[0])))

    # 2. Build Model
    print("Building Gurobi model...")
    try:
        m, x, y, z, T = build_hpp_model(I_idx, J_idx, K_idx, C, Q, c, p, R, gamma)
        
        m.setParam('TimeLimit', 600)
        m.optimize()

        if m.Status == GRB.OPTIMAL:
            print(f"Optimal solution found: {m.ObjVal}")
            return m, x, y, z
    except IndexError as e:
        print(f"CRITICAL INDEX ERROR: {e}")
        print("Check Excel files for empty cells or extra columns.")
        return None

if __name__ == "__main__":
    result = solve_hpp("1")
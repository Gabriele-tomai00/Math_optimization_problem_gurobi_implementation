import os
import gurobipy as gp
from gurobipy import GRB
from data_loader import get_data_from_file_excel


# ---------------------------------------------------------------------------
# UE constraints (24)–(28)
# ---------------------------------------------------------------------------

def _add_ue_constraints(m, z, r, B, u, v,
                        I_idx, J_idx, K_idx, C, Q, y_param):
    """
    Add User-Equilibrium constraints (24)–(28).
    y_param: dict {(i,j): Gurobi Var or float/int}
    """
    # Eq. (24): Σ_{w≠k} Σ_i z^{kw}_ij <= r^k_j   [M=1]
    for j in J_idx:
        for k in K_idx:
            m.addConstr(
                gp.quicksum(z[i, j, k, w]
                            for i in I_idx for w in K_idx if w != k)
                <= r[j, k],
                name=f"UE24_{j}_{k}"
            )

    # Eq. (25): Σ_i B^w_ij <= Σ_i C^w_i * (1 - r^w_j)   [M = Σ_i C^w_i]
    M_25 = {w: sum(C[i][w] for i in I_idx) for w in K_idx}
    for j in J_idx:
        for w in K_idx:
            m.addConstr(
                gp.quicksum(B[i, j, w] for i in I_idx)
                <= M_25[w] * (1 - r[j, w]),
                name=f"UE25_{j}_{w}"
            )

    # Eq. (26): C^w_i * y_ij - Σ_k Q^k_j * z^{kw}_ij <= B^w_ij
    for i in I_idx:
        for j in J_idx:
            for w in K_idx:
                demand = gp.quicksum(Q[j][k] * z[i, j, k, w] for k in K_idx)
                m.addConstr(
                    C[i][w] * y_param[i, j] - demand <= B[i, j, w],
                    name=f"UE26_{i}_{j}_{w}"
                )

    # Eq. (27): u^k_j - u^w_j <= (1-v^{kw}_j)*|K| - 1,  k≠w   [M=|K|]
    M_27 = len(K_idx)
    for j in J_idx:
        for k in K_idx:
            for w in K_idx:
                if k != w:
                    m.addConstr(
                        u[j, k] - u[j, w]
                        <= (1 - v[j, k, w]) * M_27 - 1,
                        name=f"UE27_{j}_{k}_{w}"
                    )

    # Eq. (28): Σ_i z^{kw}_ij <= v^{kw}_j,  k≠w   [M=1]
    for j in J_idx:
        for k in K_idx:
            for w in K_idx:
                if k != w:
                    m.addConstr(
                        gp.quicksum(z[i, j, k, w] for i in I_idx)
                        <= v[j, k, w],
                        name=f"UE28_{j}_{k}_{w}"
                    )


# ---------------------------------------------------------------------------
# Contracting-cost linearisation (29)–(32)
# ---------------------------------------------------------------------------

def _add_contracting_linearisation(m, T, delta, z,
                                   I_idx, J_idx, K_idx,
                                   C, Q, p, R, x_param,
                                   name_prefix=""):
    """
    Add contracting-cost linearisation (29)–(32).
    T_i = max{0, R_i - revenue_i} * x_i
    x_param: dict {i: Gurobi Var or float/int}
    """
    for i in I_idx:
        rev_expr = gp.quicksum(
            p[i][w] * Q[j][k] * z[i, j, k, w]
            for j in J_idx for k in K_idx for w in K_idx
        )
        full_rev = sum(C[i][w] * p[i][w] for w in K_idx)

        m.addConstr(  # (29)
            T[i] <= R[i] - rev_expr + delta[i] * (full_rev - R[i]),
            name=f"{name_prefix}T29_{i}"
        )
        m.addConstr(  # (30)
            T[i] >= R[i] * x_param[i] - rev_expr,
            name=f"{name_prefix}T30_{i}"
        )
        m.addConstr(  # (31)
            T[i] <= R[i] * x_param[i],
            name=f"{name_prefix}T31_{i}"
        )
        m.addConstr(  # (32)
            T[i] <= (1 - delta[i]) * R[i],
            name=f"{name_prefix}T32_{i}"
        )


# ---------------------------------------------------------------------------
# Subproblem: fix (x̂,ŷ), maximise cost over UE-feasible assignments
# ---------------------------------------------------------------------------

def solve_subproblem(I_idx, J_idx, K_idx, C, Q, c, p, R, gamma,
                     x_fixed, y_fixed, verbose=False):
    x_bin = {i: round(x_fixed[i]) for i in I_idx}
    y_bin = {(i, j): round(y_fixed[i, j]) for i in I_idx for j in J_idx}

    m = gp.Model("Subproblem")
    m.setParam("OutputFlag", 1 if verbose else 0)
    m.setParam("TimeLimit", 300)

    z     = m.addVars(I_idx, J_idx, K_idx, K_idx,
                      vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="z")
    r     = m.addVars(J_idx, K_idx, vtype=GRB.BINARY, name="r")
    B     = m.addVars(I_idx, J_idx, K_idx, vtype=GRB.CONTINUOUS, lb=0.0, name="B")
    u     = m.addVars(J_idx, K_idx, vtype=GRB.CONTINUOUS, name="u")
    v     = m.addVars(J_idx, K_idx, K_idx, vtype=GRB.BINARY, name="v")
    T     = m.addVars(I_idx, vtype=GRB.CONTINUOUS, lb=0.0, name="T")
    delta = m.addVars(I_idx, vtype=GRB.BINARY, name="delta")

    m.setObjective(
        gp.quicksum(T[i] for i in I_idx)
        + gp.quicksum(c[i][j] * Q[j][k] * z[i, j, k, w]
                      for i in I_idx for j in J_idx
                      for k in K_idx for w in K_idx)
        + gamma * gp.quicksum(Q[j][k] * z[i, j, k, w]
                               for i in I_idx for j in J_idx
                               for k in K_idx for w in K_idx if w != k),
        GRB.MAXIMIZE
    )

    # Eq. (5): all demand assigned
    for j in J_idx:
        for k in K_idx:
            m.addConstr(
                gp.quicksum(z[i, j, k, w]
                            for i in I_idx for w in K_idx) == 1,
                name=f"Assign_{j}_{k}"
            )

    # Eq. (6): capacity (y is fixed)
    for i in I_idx:
        for j in J_idx:
            for w in K_idx:
                m.addConstr(
                    gp.quicksum(Q[j][k] * z[i, j, k, w] for k in K_idx)
                    <= C[i][w] * y_bin[i, j],
                    name=f"Cap_{i}_{j}_{w}"
                )

    _add_ue_constraints(m, z, r, B, u, v,
                        I_idx, J_idx, K_idx, C, Q, y_bin)
    _add_contracting_linearisation(m, T, delta, z,
                                   I_idx, J_idx, K_idx,
                                   C, Q, p, R, x_bin)
    m.optimize()

    if m.Status == GRB.OPTIMAL:
        z_vals = {
            (i, j, k, w): z[i, j, k, w].X
            for i in I_idx for j in J_idx
            for k in K_idx for w in K_idx
        }
        return m.ObjVal, z_vals
    else:
        print(f"  [Subproblem] Non-optimal status: {m.Status}")
        return None, None


# ---------------------------------------------------------------------------
# Main: Column-and-Row Generation algorithm
# ---------------------------------------------------------------------------

def solve_robust_crg(file_id="1", time_limit_master=600, epsilon=1e-4):
    # --- Load data ---
    file_path = os.path.join("..", "quarantine_hotel_instances", f"{file_id}.xlsx")
    data = get_data_from_file_excel(file_path, 0)

    Q     = [row for row in data["demand"]   if row]
    C     = [row for row in data["capacity"] if row]
    c     = [row for row in data["cost"]     if row]
    p     = [row for row in data["price"]    if row]
    R     = [val for val in data["revenue"]  if val is not None]
    gamma = data["penalty"]

    I_idx = list(range(len(R)))
    J_idx = list(range(len(Q)))
    K_idx = list(range(len(Q[0])))

    print(f"Instance {file_id}: |I|={len(I_idx)}, |J|={len(J_idx)}, |K|={len(K_idx)}")
    print(f"gamma = {gamma}\n")

    # --- Initialise Master Problem ---
    master = gp.Model("Master_CRG")
    master.setParam("OutputFlag", 1)
    master.setParam("TimeLimit", time_limit_master)

    x   = master.addVars(I_idx, vtype=GRB.BINARY, name="x")
    y   = master.addVars(I_idx, J_idx, vtype=GRB.BINARY, name="y")
    eta = master.addVar(lb=0.0, name="eta")

    master.setObjective(eta, GRB.MINIMIZE)

    # Eq. (3): Σ_j y_ij = x_i
    for i in I_idx:
        master.addConstr(
            gp.quicksum(y[i, j] for j in J_idx) == x[i],
            name=f"Struct3_{i}"
        )

    # Eq. (4): Σ_{i,w} C^w_i * y_ij >= Σ_k Q^k_j
    for j in J_idx:
        master.addConstr(
            gp.quicksum(C[i][w] * y[i, j]
                        for i in I_idx for w in K_idx)
            >= sum(Q[j][k] for k in K_idx),
            name=f"Struct4_{j}"
        )

    # --- CRG main loop ---
    UB        = float("inf")   # best true robust cost seen
    best_x    = None
    best_y    = None
    iteration = 0

    while True:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}")
        print(f"{'='*60}")

        # --- Solve Master ---
        master.optimize()

        if master.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            print(f"Master failed: status {master.Status}")
            break
        if master.SolCount == 0:
            print("Master: no feasible solution found.")
            break

        LB    = master.ObjVal
        x_val = {i: x[i].X for i in I_idx}
        y_val = {(i, j): y[i, j].X for i in I_idx for j in J_idx}
        x_bin = {i: round(x_val[i]) for i in I_idx}
        y_bin = {(i, j): round(y_val[i, j]) for i in I_idx for j in J_idx}

        print(f"  Master LB        = {LB:.4f}")
        print(f"  Best UB so far   = {UB:.4f}")
        print(f"  Selected hotels  = {[i for i in I_idx if x_bin[i]==1]}")

        # --- Convergence check BEFORE subproblem ---
        # If the master lower bound already meets or exceeds the best known
        # upper bound, no solution can do better than UB. We are done.
        if LB >= UB - epsilon:
            print(f"\n*** CONVERGENCE: LB ({LB:.4f}) >= UB ({UB:.4f}) - ε ***")
            print("  No improving solution possible. Optimal is stored in UB.")
            break

        # --- Solve Subproblem ---
        worst_cost, z_scenario = solve_subproblem(
            I_idx, J_idx, K_idx, C, Q, c, p, R, gamma, x_val, y_val
        )

        if worst_cost is None:
            print("  Subproblem failed — stopping.")
            break

        print(f"  Subproblem cost  = {worst_cost:.4f}")

        # Update UB: the true robust cost of this solution is worst_cost
        if worst_cost < UB:
            UB     = worst_cost
            best_x = x_bin.copy()
            best_y = y_bin.copy()
            print(f"  *** New best UB  = {UB:.4f} ***")

        print(f"  LB = {LB:.4f}   UB = {UB:.4f}   Gap = {UB - LB:.4f}")

        # --- Convergence check AFTER subproblem ---
        # If worst_cost <= LB + ε: the subproblem confirms that the current
        # solution's true cost matches the master lower bound.
        # Combined with LB being a lower bound for ALL solutions, this means
        # the current solution is globally optimal.
        if worst_cost <= LB + epsilon:
            print(f"\n*** CONVERGENCE: subproblem cost ({worst_cost:.4f}) ≈ LB ({LB:.4f}) ***")
            break

        # --- Add scenario cut ---
        s       = iteration
        T_s     = master.addVars(I_idx, lb=0.0, name=f"T_s{s}")
        delta_s = master.addVars(I_idx, vtype=GRB.BINARY, name=f"delta_s{s}")

        # Fixed transport + misplacement cost for scenario z_s
        C_A = sum(
            c[i][j] * Q[j][k] * z_scenario[i, j, k, w]
            for i in I_idx for j in J_idx
            for k in K_idx for w in K_idx
        )
        C_M = gamma * sum(
            Q[j][k] * z_scenario[i, j, k, w]
            for i in I_idx for j in J_idx
            for k in K_idx for w in K_idx if w != k
        )

        # Scenario cut: η >= Σ_i T_s_i(x) + C_A + C_M
        master.addConstr(
            eta >= gp.quicksum(T_s[i] for i in I_idx) + C_A + C_M,
            name=f"Cut_{s}"
        )

        # T_s_i depends on x (master variable) via eqs (30)-(32)
        # Revenue under z_s is a known constant for each hotel i
        for i in I_idx:
            rev_s    = sum(
                p[i][w] * Q[j][k] * z_scenario[i, j, k, w]
                for j in J_idx for k in K_idx for w in K_idx
            )
            full_rev = sum(C[i][w] * p[i][w] for w in K_idx)

            master.addConstr(  # (29)
                T_s[i] <= R[i] - rev_s + delta_s[i] * (full_rev - R[i]),
                name=f"T29_s{s}_{i}"
            )
            master.addConstr(  # (30)
                T_s[i] >= R[i] * x[i] - rev_s,
                name=f"T30_s{s}_{i}"
            )
            master.addConstr(  # (31)
                T_s[i] <= R[i] * x[i],
                name=f"T31_s{s}_{i}"
            )
            master.addConstr(  # (32)
                T_s[i] <= (1 - delta_s[i]) * R[i],
                name=f"T32_s{s}_{i}"
            )

    # --- Report ---
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)

    if best_x is not None:
        selected = [i for i in I_idx if best_x.get(i, 0) == 1]
        alloc    = {j: [i for i in I_idx if best_y.get((i, j), 0) == 1]
                    for j in J_idx}
        print(f"Optimal robust cost : {UB:.4f}")
        print(f"Selected hotels     : {selected}")
        print(f"Allocation (j→[i])  : {alloc}")
        print(f"Total iterations    : {iteration}")
    else:
        print("No optimal solution found.")

    return best_x, best_y, UB


if __name__ == "__main__":
    solve_robust_crg(file_id="1")
"""
Batch Runner — CRG on all 48 instances
=======================================
Iterates over:
  - 16 Excel files (1.xlsx ... 16.xlsx)
  - 3 sheets per file (sheet index 0, 1, 2)

For each instance runs solve_robust_crg() and collects:
  - |I|, |J|, |K|
  - optimal robust cost (UB)
  - total CRG iterations
  - wall-clock time
  - convergence status

Results are saved to results_table.csv (replicates Table 3 structure).
"""

import os
import time
import csv
import traceback

import gurobipy as gp
from gurobipy import GRB

from data_loader import get_data_from_file_excel


# ---------------------------------------------------------------------------
# UE constraints (24)-(28)
# ---------------------------------------------------------------------------

def _add_ue_constraints(m, z, r, B, u, v,
                        I_idx, J_idx, K_idx, C, Q, y_param):
    """Add User-Equilibrium constraints (24)-(28)."""

    # Eq. (24): Σ_{w≠k} Σ_i z^{kw}_ij <= r^k_j
    for j in J_idx:
        for k in K_idx:
            m.addConstr(
                gp.quicksum(z[i, j, k, w]
                            for i in I_idx for w in K_idx if w != k)
                <= r[j, k],
                name=f"UE24_{j}_{k}"
            )

    # Eq. (25): Σ_i B^w_ij <= Σ_i C^w_i * (1 - r^w_j)
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

    # Eq. (27): u^k_j - u^w_j <= (1-v^{kw}_j)*|K| - 1
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

    # Eq. (28): Σ_i z^{kw}_ij <= v^{kw}_j
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
# Contracting-cost linearisation (29)-(32)
# ---------------------------------------------------------------------------

def _add_contracting_linearisation(m, T, delta, z,
                                   I_idx, J_idx, K_idx,
                                   C, Q, p, R, x_param,
                                   name_prefix=""):
    """Add contracting-cost linearisation (29)-(32)."""
    for i in I_idx:
        rev_expr = gp.quicksum(
            p[i][w] * Q[j][k] * z[i, j, k, w]
            for j in J_idx for k in K_idx for w in K_idx
        )
        full_rev = sum(C[i][w] * p[i][w] for w in K_idx)

        m.addConstr(
            T[i] <= R[i] - rev_expr + delta[i] * (full_rev - R[i]),
            name=f"{name_prefix}T29_{i}"
        )
        m.addConstr(
            T[i] >= R[i] * x_param[i] - rev_expr,
            name=f"{name_prefix}T30_{i}"
        )
        m.addConstr(
            T[i] <= R[i] * x_param[i],
            name=f"{name_prefix}T31_{i}"
        )
        m.addConstr(
            T[i] <= (1 - delta[i]) * R[i],
            name=f"{name_prefix}T32_{i}"
        )


# ---------------------------------------------------------------------------
# Subproblem
# ---------------------------------------------------------------------------

def solve_subproblem(I_idx, J_idx, K_idx, C, Q, c, p, R, gamma,
                     x_fixed, y_fixed):
    """Solve lower-level subproblem: find worst-case UE assignment."""
    x_bin = {i: round(x_fixed[i]) for i in I_idx}
    y_bin = {(i, j): round(y_fixed[i, j]) for i in I_idx for j in J_idx}

    m = gp.Model("Subproblem")
    m.setParam("OutputFlag", 0)
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

    # Eq. (6): capacity
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
        return None, None


# ---------------------------------------------------------------------------
# CRG solver (single instance)
# ---------------------------------------------------------------------------

def solve_crg_instance(Q, C, c, p, R, gamma,
                       time_limit_total=7200, epsilon=1e-4):
    I_idx = list(range(len(R)))
    J_idx = list(range(len(Q)))
    K_idx = list(range(len(Q[0])))

    max_subsidy  = sum(R[i] for i in I_idx)
    max_assign   = sum(max(c[i][j] for i in I_idx) * sum(Q[j][k] for k in K_idx)
                       for j in J_idx)
    max_mismatch = gamma * sum(sum(Q[j][k] for k in K_idx) for j in J_idx)
    M_big = max_subsidy + max_assign + max_mismatch  # global upper bound

    master = gp.Model("Master_CRG")
    master.setParam("OutputFlag", 0)

    x   = master.addVars(I_idx, vtype=GRB.BINARY, name="x")
    y   = master.addVars(I_idx, J_idx, vtype=GRB.BINARY, name="y")
    eta = master.addVar(lb=0.0, ub=M_big, name="eta")

    master.setObjective(eta, GRB.MINIMIZE)

    for i in I_idx:
        master.addConstr(gp.quicksum(y[i, j] for j in J_idx) == x[i],
                         name=f"Struct3_{i}")
    for j in J_idx:
        master.addConstr(
            gp.quicksum(C[i][w] * y[i, j] for i in I_idx for w in K_idx)
            >= sum(Q[j][k] for k in K_idx),
            name=f"Struct4_{j}"
        )

    UB        = float("inf")
    best_x    = None
    best_y    = None
    iteration = 0
    status    = "optimal"
    t_start   = time.time()

    while True:
        elapsed = time.time() - t_start
        if elapsed >= time_limit_total:
            status = "timeout"
            break

        master.setParam("TimeLimit", max(1.0, time_limit_total - elapsed))
        master.optimize()

        if master.Status == GRB.TIME_LIMIT:
            status = "timeout"
            if master.SolCount == 0:
                break
        elif master.Status not in (GRB.OPTIMAL,):
            status = "error"
            break

        if master.SolCount == 0:
            status = "error"
            break

        LB    = master.ObjVal
        x_val = {i: x[i].X for i in I_idx}
        y_val = {(i, j): y[i, j].X for i in I_idx for j in J_idx}

        if UB - LB <= epsilon:
            break

        iteration += 1

        worst_cost, z_scenario = solve_subproblem(
            I_idx, J_idx, K_idx, C, Q, c, p, R, gamma, x_val, y_val
        )

        if worst_cost is None:
            status = "error"
            break

        # Update UB — BEFORE building the cut, so cut_M uses tighter bound
        if worst_cost < UB:
            UB     = worst_cost
            best_x = {i: round(x_val[i]) for i in I_idx}
            best_y = {(i, j): round(y_val[i, j])
                      for i in I_idx for j in J_idx}

        x_bar = {i: round(x_val[i]) for i in I_idx}
        y_bar = {(i, j): round(y_val[i, j]) for i in I_idx for j in J_idx}

        dist_expr = (
            gp.quicksum((1 - x[i])   for i in I_idx     if x_bar[i] == 1)
          + gp.quicksum(x[i]          for i in I_idx     if x_bar[i] == 0)
          + gp.quicksum((1 - y[i, j]) for i in I_idx
                                       for j in J_idx    if y_bar[i, j] == 1)
          + gp.quicksum(y[i, j]       for i in I_idx
                                       for j in J_idx    if y_bar[i, j] == 0)
        )

        # Value-function cut with dynamic M = current UB (tighter over time)
        cut_M = UB
        master.addConstr(
            eta >= worst_cost - cut_M * dist_expr,
            name=f"ValFunc_{iteration}"
        )

        # No-good cut: never revisit this (x,y)
        master.addConstr(dist_expr >= 1, name=f"NoGood_{iteration}")

        # Progress log every 100 iterations
        if iteration % 100 == 0:
            print(f"    iter={iteration}, LB={LB:.1f}, UB={UB:.1f}, "
                  f"gap={UB-LB:.1f}, elapsed={elapsed:.0f}s")

    return UB, best_x, best_y, iteration, status
    
# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_all_instances(data_dir="../quarantine_hotel_instances",
                      time_limit=7200,
                      output_csv="results_table.csv"):
    """
    Run CRG on all 48 instances (16 files × 3 sheets).
    Saves results to CSV.
    """
    results = []

    # CSV header — mirrors Table 3 structure of the paper
    header = [
        "file", "sheet",
        "|I|", "|J|", "|K|",
        "gamma",
        "robust_cost",
        "iterations",
        "time_s",
        "status"
    ]

    print(f"{'='*70}")
    print(f"BATCH CRG — {16} files × 3 sheets = 48 instances")
    print(f"Time limit per instance: {time_limit}s")
    print(f"{'='*70}\n")

    for file_id in range(1, 17):
        file_path = os.path.join(data_dir, f"{file_id}.xlsx")

        if not os.path.exists(file_path):
            print(f"[SKIP] {file_id}.xlsx not found")
            continue

        for sheet_idx in range(3):
            instance_label = f"{file_id}.xlsx / sheet {sheet_idx}"
            print(f"--- {instance_label} ---", flush=True)

            # Load data
            try:
                data = get_data_from_file_excel(file_path, sheet_idx)
                if not data:
                    print(f"  [SKIP] sheet {sheet_idx} is not a valid instance")
                    continue

                Q     = [row for row in data["demand"]   if row]
                C     = [row for row in data["capacity"] if row]
                c     = [row for row in data["cost"]     if row]
                p     = [row for row in data["price"]    if row]
                R     = [val for val in data["revenue"]  if val is not None]
                gamma = data["penalty"]

            except Exception as e:
                print(f"  [ERROR] loading data: {e}")
                results.append({
                    "file": file_id, "sheet": sheet_idx,
                    "|I|": "?", "|J|": "?", "|K|": "?",
                    "gamma": "?",
                    "robust_cost": "ERROR",
                    "iterations": 0,
                    "time_s": 0,
                    "status": f"load_error: {e}"
                })
                continue

            n_I = len(R)
            n_J = len(Q)
            n_K = len(Q[0]) if Q else 0

            print(f"  |I|={n_I}, |J|={n_J}, |K|={n_K}, gamma={gamma}")

            # Run CRG
            t_start = time.time()
            try:
                UB, best_x, best_y, iters, status = solve_crg_instance(
                    Q, C, c, p, R, gamma,
                    time_limit_total=time_limit
                )
            except Exception as e:
                elapsed = time.time() - t_start
                print(f"  [ERROR] solver crashed: {e}")
                traceback.print_exc()
                results.append({
                    "file": file_id, "sheet": sheet_idx,
                    "|I|": n_I, "|J|": n_J, "|K|": n_K,
                    "gamma": gamma,
                    "robust_cost": "ERROR",
                    "iterations": 0,
                    "time_s": round(elapsed, 1),
                    "status": f"solver_error: {e}"
                })
                continue

            elapsed = time.time() - t_start

            cost_str = f"{UB:.2f}" if UB < float("inf") else "inf"
            print(f"  → cost={cost_str}, iters={iters}, "
                  f"time={elapsed:.1f}s, status={status}")

            results.append({
                "file": file_id, "sheet": sheet_idx,
                "|I|": n_I, "|J|": n_J, "|K|": n_K,
                "gamma": gamma,
                "robust_cost": cost_str,
                "iterations": iters,
                "time_s": round(elapsed, 1),
                "status": status
            })

    # Save to CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*70}")
    print(f"Done. Results saved to: {output_csv}")
    print(f"Total instances processed: {len(results)}")
    print(f"{'='*70}")

    # Summary table printed to console
    print(f"\n{'File':>5} {'Sheet':>6} {'|I|':>4} {'|J|':>4} {'|K|':>4} "
          f"{'gamma':>6} {'Cost':>12} {'Iters':>6} {'Time(s)':>8} {'Status':>10}")
    print("-" * 75)
    for r in results:
        print(f"{r['file']:>5} {r['sheet']:>6} {r['|I|']:>4} {r['|J|']:>4} "
              f"{r['|K|']:>4} {r['gamma']:>6} {r['robust_cost']:>12} "
              f"{r['iterations']:>6} {r['time_s']:>8} {r['status']:>10}")

    return results


if __name__ == "__main__":
    run_all_instances(
        data_dir="../quarantine_hotel_instances",
        time_limit=7200,          # 2 hours per instance, as in the paper
        output_csv="results_table.csv"
    )
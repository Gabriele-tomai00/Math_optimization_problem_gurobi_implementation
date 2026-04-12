# FLDA_decomposition.py

import gurobipy as gp
from gurobipy import GRB
import os
import csv
import time
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import get_data_from_file_excel, validate_dimensions


# =========================
# MASTER
# =========================
def build_master(Q, C, c, p, R, gamma):

    I = range(len(R))
    J = range(len(Q))
    K = range(len(Q[0]))

    model = gp.Model("MASTER")

    x = model.addVars(I, vtype=GRB.BINARY, name="x")
    y = model.addVars(I, J, vtype=GRB.BINARY, name="y")
    theta = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="theta")

    # allocation only if open
    for i in I:
        model.addConstr(gp.quicksum(y[i, j] for j in J) == x[i])

    # capacity per node
    for j in J:
        model.addConstr(
            gp.quicksum(C[i][w] * y[i, j] for i in I for w in K)
            >= gp.quicksum(Q[j][k] for k in K)
        )

    model.setObjective(theta, GRB.MINIMIZE)

    return model, x, y, theta


# =========================
# SUBPROBLEM (worst-case)
# =========================
def solve_subproblem(Q, C, c, gamma, x_val, y_val):

    I = range(len(x_val))
    J = range(len(Q))
    K = range(len(Q[0]))

    model = gp.Model("SUB")

    z = model.addVars(I, J, K, K, lb=0, ub=1, name="z")
    r = model.addVars(J, K, vtype=GRB.BINARY, name="r")
    v = model.addVars(J, K, K, vtype=GRB.BINARY, name="v")
    u = model.addVars(J, K, name="u")
    B = model.addVars(I, J, K, lb=0, name="B")

    # assignment
    for j in J:
        for k in K:
            model.addConstr(
                gp.quicksum(z[i, j, k, w] for i in I for w in K) == 1
            )

    # capacity
    for i in I:
        for j in J:
            for w in K:
                model.addConstr(
                    gp.quicksum(Q[j][k] * z[i, j, k, w] for k in K)
                    <= C[i][w] * y_val[i, j]
                )

    # UE constraints
    for j in J:
        for k in K:
            model.addConstr(
                gp.quicksum(z[i, j, k, w]
                            for i in I for w in K if w != k)
                <= r[j, k]
            )

    for j in J:
        for w in K:
            model.addConstr(
                gp.quicksum(B[i, j, w] for i in I)
                <= gp.quicksum(C[i][w] for i in I) * (1 - r[j, w])
            )

    for i in I:
        for j in J:
            for w in K:
                model.addConstr(
                    C[i][w] * y_val[i, j]
                    - gp.quicksum(Q[j][k] * z[i, j, k, w] for k in K)
                    <= B[i, j, w]
                )

    # arcs
    for j in J:
        for k in K:
            for w in K:
                if k != w:
                    model.addConstr(
                        gp.quicksum(z[i, j, k, w] for i in I)
                        <= v[j, k, w]
                    )

    # no cycles
    K_size = len(K)
    for j in J:
        for k in K:
            for w in K:
                if k != w:
                    model.addConstr(
                        u[j, k] - u[j, w]
                        <= (1 - v[j, k, w]) * K_size - 1
                    )

    # objective (worst-case)
    assignment_cost = gp.quicksum(
        c[i][j] * Q[j][k] * z[i, j, k, w]
        for i in I for j in J for k in K for w in K
    )

    misplacement_cost = gp.quicksum(
        gamma * Q[j][k] * z[i, j, k, w]
        for i in I for j in J for k in K for w in K if w != k
    )

    model.setObjective(assignment_cost + misplacement_cost, GRB.MAXIMIZE)

    model.optimize()

    return model


# =========================
# EXTRACT z*
# =========================
def extract_z(model, I, J, K):
    z_star = {}
    for i in I:
        for j in J:
            for k in K:
                for w in K:
                    z_star[i, j, k, w] = model.getVarByName(
                        f"z[{i},{j},{k},{w}]"
                    ).X
    return z_star


# =========================
# ADD CUT
# =========================
def add_cut(master, theta, x, y, z_star, Q, c, gamma):

    I = range(len(x))
    J = range(len(Q))
    K = range(len(Q[0]))

    expr = 0

    for i in I:
        for j in J:
            for k in K:
                for w in K:
                    coeff = Q[j][k] * z_star[i, j, k, w]

                    expr += c[i][j] * coeff * y[i, j]

                    if w != k:
                        expr += gamma * coeff * y[i, j]

    master.addConstr(theta >= expr)


# =========================
# SOLVER
# =========================
def solve_flda(Q, C, c, p, R, gamma, max_iter=30, tol=1e-4):

    I = range(len(R))
    J = range(len(Q))
    K = range(len(Q[0]))

    master, x, y, theta = build_master(Q, C, c, p, R, gamma)

    UB = float("inf")
    LB = -float("inf")

    start = time.time()

    for it in range(max_iter):

        master.optimize()

        x_val = {i: x[i].X for i in I}
        y_val = {(i, j): y[i, j].X for i in I for j in J}

        sub = solve_subproblem(Q, C, c, gamma, x_val, y_val)

        if sub.status != GRB.OPTIMAL:
            print("Subproblem not optimal!")
            break

        Z_val = sub.ObjVal

        LB = max(LB, Z_val)
        UB = theta.X

        print(f"[Iter {it}] LB={LB:.4f}, UB={UB:.4f}")

        if abs(UB - LB) <= tol:
            print("Converged")
            break

        z_star = extract_z(sub, I, J, K)

        add_cut(master, theta, x, y, z_star, Q, c, gamma)

    end = time.time()

    return master, end - start


# =========================
# RUN INSTANCE
# =========================
def run_instance(file_idx, sheet_idx, output_file):

    file_name = f"{file_idx}.xlsx"
    file_path = os.path.join("..", "quarantine_hotel_instances", file_name)

    if not os.path.exists(file_path):
        print(f"{file_name} not found")
        return

    data = get_data_from_file_excel(file_path, sheet_idx)

    if not data or "demand" not in data:
        return

    Q = data["demand"]
    C = data["capacity"]
    c = data["cost"]
    p = data["price"]
    R = data["revenue"]
    gamma = data["penalty"]

    if not validate_dimensions(Q, C, c, p, R):
        return

    model, runtime = solve_flda(Q, C, c, p, R, gamma)

    obj = model.ObjVal

    header = ["file", "sheet", "objective", "time_sec"]
    file_exists = os.path.isfile(output_file)

    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)

        writer.writerow([
            file_idx,
            sheet_idx,
            round(obj, 2),
            round(runtime, 4)
        ])


# =========================
# PLOTS
# =========================
def generate_plots(csv_file):

    df = pd.read_csv(csv_file)

    plt.figure()
    plt.plot(df["file"], df["time_sec"], marker="o")
    plt.xlabel("File")
    plt.ylabel("Time (s)")
    plt.title("FLDA Decomposition Time")
    plt.grid(True)
    plt.savefig("flda_time.png")

    plt.figure()
    plt.plot(df["file"], df["objective"], marker="o")
    plt.xlabel("File")
    plt.ylabel("Objective")
    plt.title("FLDA Objective")
    plt.grid(True)
    plt.savefig("flda_obj.png")


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    output_file = "flda_results.csv"

    if os.path.exists(output_file):
        os.remove(output_file)

    for file_idx in range(1, 17):
        for sheet_idx in range(0, 3):
            print(f"\n--- File {file_idx}, Sheet {sheet_idx} ---")
            run_instance(file_idx, sheet_idx, output_file)

    generate_plots(output_file)

    print("\nDone.")
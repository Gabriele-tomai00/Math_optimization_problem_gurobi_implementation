# FLDA.py

import os
import csv
import time
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB

from data_loader import get_data_from_file_excel, validate_dimensions


# ---------------------------------------------------------
# 1. Exact FLDA solver (unchanged)
# ---------------------------------------------------------
def solve_flda(Q, C, c, p, R, gamma):

    I = range(len(R))
    J = range(len(Q))
    K = range(len(Q[0]))
    K_size = len(K)

    model = gp.Model("FLDA_exact")

    x = model.addVars(I, vtype=GRB.BINARY, name="x")
    y = model.addVars(I, J, vtype=GRB.BINARY, name="y")
    z = model.addVars(I, J, K, K, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="z")

    T = model.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name="T")
    delta = model.addVars(I, vtype=GRB.BINARY, name="delta")

    r = model.addVars(J, K, vtype=GRB.BINARY, name="r")
    u = model.addVars(J, K, vtype=GRB.CONTINUOUS, name="u")
    v = model.addVars(J, K, K, vtype=GRB.BINARY, name="v")
    B = model.addVars(I, J, K, lb=0, vtype=GRB.CONTINUOUS, name="B")

    # Constraints (3)-(6)
    for i in I:
        model.addConstr(gp.quicksum(y[i, j] for j in J) == x[i])

    for j in J:
        model.addConstr(
            gp.quicksum(C[i][w] * y[i, j] for i in I for w in K)
            >= gp.quicksum(Q[j][k] for k in K)
        )

    for j in J:
        for k in K:
            model.addConstr(
                gp.quicksum(z[i, j, k, w] for i in I for w in K) == 1
            )

    for i in I:
        for j in J:
            for w in K:
                model.addConstr(
                    gp.quicksum(Q[j][k] * z[i, j, k, w] for k in K)
                    <= C[i][w] * y[i, j]
                )

    # UE constraints (24)-(28)
    for j in J:
        for k in K:
            model.addConstr(
                gp.quicksum(z[i, j, k, w] for i in I for w in K if w != k)
                <= r[j, k]
            )

    for j in J:
        for w in K:
            model.addConstr(
                gp.quicksum(B[i, j, w] for i in I)
                <= sum(C[i][w] for i in I) * (1 - r[j, w])
            )

    for i in I:
        for j in J:
            for w in K:
                model.addConstr(
                    C[i][w] * y[i, j]
                    - gp.quicksum(Q[j][k] * z[i, j, k, w] for k in K)
                    <= B[i, j, w]
                )

    for j in J:
        for k in K:
            for w in K:
                if k != w:
                    model.addConstr(
                        u[j, k] - u[j, w] <= (1 - v[j, k, w]) * K_size - 1
                    )

    for j in J:
        for k in K:
            for w in K:
                if k != w:
                    model.addConstr(
                        gp.quicksum(z[i, j, k, w] for i in I)
                        <= v[j, k, w]
                    )

    # Contracting cost (29)-(32)
    for i in I:
        model.addConstr(
            T[i] <= R[i]
            - gp.quicksum(p[i][w] * Q[j][k] * z[i, j, k, w]
                          for j in J for k in K for w in K)
            + delta[i] * (sum(C[i][w] * p[i][w] for w in K) - R[i])
        )

        model.addConstr(
            T[i] >= R[i] * x[i]
            - gp.quicksum(p[i][w] * Q[j][k] * z[i, j, k, w]
                          for j in J for k in K for w in K)
        )

        model.addConstr(T[i] <= R[i] * x[i])
        model.addConstr(T[i] <= (1 - delta[i]) * R[i])

    # Objective
    assignment_cost = gp.quicksum(
        c[i][j] * Q[j][k] * z[i, j, k, w]
        for i in I for j in J for k in K for w in K
    )

    misplacement_cost = gp.quicksum(
        gamma * Q[j][k] * z[i, j, k, w]
        for i in I for j in J for k in K for w in K if w != k
    )

    model.setObjective(
        gp.quicksum(T[i] for i in I) +
        assignment_cost +
        misplacement_cost,
        GRB.MINIMIZE
    )

    model.optimize()
    return model


# ---------------------------------------------------------
# 2. Run a single instance (file + sheet)
# ---------------------------------------------------------
def run_instance(file_idx, sheet_idx, output_file="results_flda.csv"):

    file_path = f"../quarantine_hotel_instances/{file_idx}.xlsx"

    print(f"\n=== Processing file {file_idx}.xlsx — sheet {sheet_idx} ===")

    data = get_data_from_file_excel(file_path, sheet_idx)

    Q = [row for row in data["demand"] if len(row) > 0]
    C = [row for row in data["capacity"] if len(row) > 0]
    c = [row for row in data["cost"] if len(row) > 0]
    p = [row for row in data["price"] if len(row) > 0]
    R = [val for val in data["revenue"] if val is not None]
    gamma = data["penalty"]

    if not validate_dimensions(data["demand"], data["capacity"], data["cost"], data["price"], data["revenue"]):
        print("Dimension validation failed, skipping.")
        return

    start = time.time()
    model = solve_flda(Q, C, c, p, R, gamma)
    end = time.time()

    if model.Status != GRB.OPTIMAL:
        print(f"Model infeasible or not optimal. Status = {model.Status}")
        return

    obj = model.ObjVal
    hotels_selected = sum(model.getVarByName(f"x[{i}]").X > 0.5 for i in range(len(R)))

    assign_cost = sum(
        c[i][j] * Q[j][k] * model.getVarByName(f"z[{i},{j},{k},{w}]").X
        for i in range(len(R))
        for j in range(len(Q))
        for k in range(len(Q[0]))
        for w in range(len(Q[0]))
    )

    misplacement_cost = sum(
        gamma * Q[j][k] * model.getVarByName(f"z[{i},{j},{k},{w}]").X
        for i in range(len(R))
        for j in range(len(Q))
        for k in range(len(Q[0]))
        for w in range(len(Q[0]))
        if w != k
    )

    contract_cost = sum(model.getVarByName(f"T[{i}]").X for i in range(len(R)))

    header = [
        "file", "sheet", "objective", "time_sec", "hotels_selected",
        "assignment_cost", "misplacement_cost", "contract_cost"
    ]

    file_exists = os.path.isfile(output_file)

    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([
            file_idx,
            sheet_idx,
            round(obj, 2),
            round(end - start, 4),
            hotels_selected,
            round(assign_cost, 2),
            round(misplacement_cost, 2),
            round(contract_cost, 2)
        ])


# ---------------------------------------------------------
# 3. Generate plots
# ---------------------------------------------------------
def generate_plots(csv_file="results_flda.csv"):
    import pandas as pd
    df = pd.read_csv(csv_file)

    # Plot 1: solve time
    plt.figure()
    plt.plot(df["file"], df["time_sec"], marker="o")
    plt.xlabel("File index")
    plt.ylabel("Solve time (seconds)")
    plt.title("FLDA Scalability: Solve Time")
    plt.grid(True)
    plt.savefig("flda_scalability_time.png")

    # Plot 2: objective
    plt.figure()
    plt.plot(df["file"], df["objective"], marker="o", color="green")
    plt.xlabel("File index")
    plt.ylabel("Objective value")
    plt.title("FLDA Scalability: Objective Value")
    plt.grid(True)
    plt.savefig("flda_scalability_objective.png")

    # Plot 3: cost breakdown
    plt.figure()
    plt.plot(df["file"], df["assignment_cost"], label="Assignment cost")
    plt.plot(df["file"], df["misplacement_cost"], label="Misplacement cost")
    plt.plot(df["file"], df["contract_cost"], label="Contract cost")
    plt.xlabel("File index")
    plt.ylabel("Cost")
    plt.title("FLDA Cost Breakdown")
    plt.legend()
    plt.grid(True)
    plt.savefig("flda_cost_breakdown.png")


# ---------------------------------------------------------
# 4. Main: run all 48 instances
# ---------------------------------------------------------
if __name__ == "__main__":

    results_file = "flda_results.csv"

    if os.path.exists(results_file):
        os.remove(results_file)
    for file_idx in range(1, 17):      # 1.xlsx ... 16.xlsx
        for sheet_idx in range(0, 3):
            run_instance(file_idx, sheet_idx, results_file)

    generate_plots(results_file)

    print("\nAll instances processed. Results saved to flda_results.csv")
    print("Plots saved as PNG files.")

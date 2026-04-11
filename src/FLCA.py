import gurobipy as gp
from gurobipy import GRB
import os
import csv
import time
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import get_data_from_file_excel, validate_dimensions


def debug_solution(model, x, y, z, Q, C):
    I = range(len(C))
    J = range(len(Q))
    K = range(len(Q[0]))

    print("\n--- DEBUG SOLUTION ---")

    print("\n[1] Assignment completeness:")
    for j in J:
        for k in K:
            val = sum(z[i, j, k, w].X for i in I for w in K)
            print(f"Node {j}, type {k} -> {val:.4f}")

    print("\n[2] Capacity usage:")
    for i in I:
        for w in K:
            used = sum(Q[j][k] * z[i, j, k, w].X for j in J for k in K)
            cap = C[i][w]
            if used > 1e-6:
                print(f"Hotel {i}, type {w}: used {used:.2f} / cap {cap}")

    print("\n[3] Misplacement:")
    for j in J:
        for k in K:
            misplaced = sum(
                z[i, j, k, w].X
                for i in I for w in K if w != k
            )
            print(f"Node {j}, type {k}: misplaced {misplaced:.4f}")

    print("\n[4] Selected hotels:")
    for i in I:
        if x[i].X > 0.5:
            print(f"Hotel {i} selected")

    print("\n[5] y consistency:")
    for i in I:
        assigned_nodes = sum(y[i, j].X for j in J)
        print(f"Hotel {i}: x={x[i].X}, sum(y)={assigned_nodes}")

    print("\n[6] Flow vs y:")
    for i in I:
        for j in J:
            flow = sum(z[i, j, k, w].X for k in K for w in K)
            if flow > 1e-6:
                print(f"(i={i}, j={j}) -> flow {flow:.2f}, y={y[i,j].X}")

    print("\n[7] Objective breakdown:")
    print(f"Objective value: {model.ObjVal}")


def solve_flca(Q, C, c, p, R, gamma):
    """
    Q[j][k] demand
    C[i][w] capacity
    c[i][j] assignment cost
    p[i][w] price
    R[i] revenue target
    gamma penalty scalar
    """

    I = range(len(R))          # hotels
    J = range(len(Q))          # demand nodes
    K = range(len(Q[0]))       # room types

    model = gp.Model("FLCA")

    # --- Variables ---
    x = model.addVars(I, vtype=GRB.BINARY, name="x")
    y = model.addVars(I, J, vtype=GRB.BINARY, name="y")
    z = model.addVars(I, J, K, K, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="z")
    T = model.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name="T")

    # --- Constraint (3): allocation only if selected ---
    for i in I:
        model.addConstr(gp.quicksum(y[i, j] for j in J) == x[i],
                        name=f"alloc_only_if_open[{i}]")

    # --- Constraint (4): enough capacity per node ---
    for j in J:
        model.addConstr(
            gp.quicksum(C[i][w] * y[i, j] for i in I for w in K)
            >= gp.quicksum(Q[j][k] for k in K),
            name=f"capacity_node[{j}]"
        )

    # --- Constraint (5): assign all demand ---
    for j in J:
        for k in K:
            model.addConstr(
                gp.quicksum(z[i, j, k, w] for i in I for w in K) == 1,
                name=f"assign_all[{j},{k}]"
            )

    # --- Constraint (6): capacity constraints ---
    for i in I:
        for j in J:
            for w in K:
                model.addConstr(
                    gp.quicksum(Q[j][k] * z[i, j, k, w] for k in K)
                    <= C[i][w] * y[i, j],
                    name=f"cap[{i},{j},{w}]"
                )

    # --- Linearized contracting cost ---
    for i in I:
        model.addConstr(
            T[i] >= R[i] * x[i] -
            gp.quicksum(p[i][w] * Q[j][k] * z[i, j, k, w]
                        for j in J for k in K for w in K),
            name=f"T_low[{i}]"
        )

    # --- Objective ---
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


def run_instance(file_idx, sheet_idx, output_file="results_flca.csv"):
    file_name = f"{file_idx}.xlsx"
    file_path = os.path.join("..", "quarantine_hotel_instances", file_name)

    if not os.path.exists(file_path):
        print(f"File {file_name} not found, skipping...")
        return

    print(f"\n=== Processing file {file_name} — sheet index {sheet_idx} ===")

    data = get_data_from_file_excel(file_path, sheet_idx)

    # sheet non valido (es. Sheet1 o foglio vuoto)
    if not data or "demand" not in data:
        print(f"Sheet {sheet_idx} is not a valid instance. Skipping.")
        return

    Q = [row for row in data["demand"] if len(row) > 0]
    C = [row for row in data["capacity"] if len(row) > 0]
    c = [row for row in data["cost"] if len(row) > 0]
    p = [row for row in data["price"] if len(row) > 0]
    R = [val for val in data["revenue"] if val is not None]
    gamma = data["penalty"]

    if not validate_dimensions(
        data["demand"],
        data["capacity"],
        data["cost"],
        data["price"],
        data["revenue"]
    ):
        print("Dimension validation failed, skipping.")
        return

    start = time.time()
    model = solve_flca(Q, C, c, p, R, gamma)
    end = time.time()

    obj = model.ObjVal

    hotels_selected = sum(
        model.getVarByName(f"x[{i}]").X > 0.5 for i in range(len(R))
    )

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

    contract_cost = sum(
        model.getVarByName(f"T[{i}]").X for i in range(len(R))
    )

    header = [
        "file", "sheet", "objective", "time_sec",
        "hotels_selected", "assignment_cost",
        "misplacement_cost", "contract_cost"
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
def generate_plots_flca(csv_file="results_flca.csv"):
    df = pd.read_csv(csv_file)

    # Plot 1: solve time
    plt.figure()
    plt.plot(df["file"], df["time_sec"], marker="o")
    plt.xlabel("File")
    plt.ylabel("Solve time (seconds)")
    plt.title("FLCA Scalability: Solve Time")
    plt.grid(True)
    plt.savefig("flca_scalability_time.png")

    # Plot 2: objective
    plt.figure()
    plt.plot(df["file"], df["objective"], marker="o", color="green")
    plt.xlabel("File")
    plt.ylabel("Objective value")
    plt.title("FLCA Scalability: Objective Value")
    plt.grid(True)
    plt.savefig("flca_scalability_objective.png")

    # Plot 3: cost breakdown
    plt.figure()
    plt.plot(df["file"], df["assignment_cost"], label="Assignment cost")
    plt.plot(df["file"], df["misplacement_cost"], label="Misplacement cost")
    plt.plot(df["file"], df["contract_cost"], label="Contract cost")
    plt.xlabel("File")
    plt.ylabel("Cost")
    plt.title("FLCA Cost Breakdown")
    plt.legend()
    plt.grid(True)
    plt.savefig("flca_cost_breakdown.png")


if __name__ == "__main__":
    results_file = "flca_results.csv"

    if os.path.exists(results_file):
        os.remove(results_file)

    # 16 file, 3 fogli numerici per file: indici 0,1,2
    for file_idx in range(1, 17):
        for sheet_idx in range(0, 3):
            run_instance(file_idx, sheet_idx, results_file)

    generate_plots_flca(results_file)

    print("\nAll instances processed. Results saved to flca_results.csv")
    print("Plots saved as PNG files.")

# ils.py

import argparse
import random
import copy
import os
import time
import csv

import gurobipy as gp
from gurobipy import GRB
from data_loader import get_data_from_file_excel, validate_dimensions
from utils import generate_plots_ils, format_time


def solve_lower_level(x_fixed, y_fixed, Q, C, c, p, R, gamma):
    """Solve the Lower Level problem for given fixed x and y.
    Given a configuration of open hotels (x_fixed) and their assigned nodes (y_fixed),
    this function computes the worst-case objective value for the follower's response.
    Returns the objective value and cost breakdown.
    """
    I, J, K = range(len(R)), range(len(Q)), range(len(Q[0]))
    model = gp.Model("LowerLevel_WorstCase")
    model.Params.OutputFlag = 0

    # flux: how many people from node j of room type k are assigned to hotel i in a room of type w
    z = model.addVars(I, J, K, K, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="z")

    r     = model.addVars(J, K,    vtype=GRB.BINARY,     name="r")
    u     = model.addVars(J, K,    vtype=GRB.CONTINUOUS, name="u")
    v     = model.addVars(J, K, K, vtype=GRB.BINARY,     name="v")
    T     = model.addVars(I,       lb=0, vtype=GRB.CONTINUOUS, name="T")
    delta = model.addVars(I,       vtype=GRB.BINARY,     name="delta")
    B     = model.addVars(I, J, K, lb=0, vtype=GRB.CONTINUOUS, name="B")

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
                    <= C[i][w] * y_fixed[i, j],
                    name=f"cap[{i},{j},{w}]"
                )

    # --- UE Conditions (Eq. 24-28) ---
    for j in J:
        for k in K:
            # [Eq. 24] Misplacement indicator: r=1 if guests are assigned to room type w != k
            model.addConstr(
                gp.quicksum(z[i, j, k, w] for i in I for w in K if w != k) <= r[j, k]
            )
            # [Eq. 25] Complementarity: misplacement only if no residual capacity
            model.addConstr(
                gp.quicksum(B[i, j, k] for i in I) <= sum(C[i][k] for i in I) * (1 - r[j, k])
            )
            for i in I:
                # [Eq. 26] Definition of residual capacity B for room type k in hotel i
                model.addConstr(
                    C[i][k] * y_fixed[i, j] - gp.quicksum(Q[j][w] * z[i, j, w, k] for w in K)
                    <= B[i, j, k]
                )

    for j in J:
        for k in K:
            for w in K:
                if k != w:
                    # [Eq. 27] Elimination of misplacement loops (MTZ-based ordering)
                    model.addConstr(u[j, k] - u[j, w] <= (1 - v[j, k, w]) * len(K) - 1)
                    # [Eq. 28] Linking physical assignment z to logical precedence v
                    model.addConstr(gp.quicksum(z[i, j, k, w] for i in I) <= v[j, k, w])

    # --- Contracting cost linearization (Eq. 29-32) ---
    for i in I:
        revenue_expr = gp.quicksum(p[i][w] * Q[j][k] * z[i, j, k, w]
                                   for j in J for k in K for w in K)
        max_rev = sum(C[i][w] * p[i][w] for w in K)

        # [Eq. 29] Upper bound for T_i
        model.addConstr(T[i] <= R[i] - revenue_expr + delta[i] * (max_rev - R[i]))
        # [Eq. 30] Lower bound: T_i must cover the shortfall
        model.addConstr(T[i] >= R[i] * x_fixed[i] - revenue_expr)
        # [Eq. 31] T_i exists only if hotel i is open
        model.addConstr(T[i] <= R[i] * x_fixed[i])
        # [Eq. 32] T_i is 0 if target is met (delta_i=1)
        model.addConstr(T[i] <= (1 - delta[i]) * R[i])

    # [Eq. 21] Objective: maximize total worst-case cost
    obj = (
        gp.quicksum(T[i] for i in I)
        + gp.quicksum(c[i][j] * Q[j][k] * z[i, j, k, w]
                      for i in I for j in J for k in K for w in K)
        + gp.quicksum(gamma * Q[j][k] * z[i, j, k, w]
                      for i in I for j in J for k in K for w in K if w != k)
    )
    model.setObjective(obj, GRB.MAXIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        contract_cost = sum(T[i].X for i in I)
        assign_cost   = sum(c[i][j] * Q[j][k] * z[i, j, k, w].X
                            for i in I for j in J for k in K for w in K)
        misplace_cost = sum(gamma * Q[j][k] * z[i, j, k, w].X
                            for i in I for j in J for k in K for w in K if w != k)
        return model.ObjVal, contract_cost, assign_cost, misplace_cost

    return float('inf'), 0, 0, 0


def generate_initial_solution(Q, C, R):
    """Generate the initial solution by minimizing the number of open hotels (Eq. 35).
    Q: demand, C: capacity, R: revenue target. Returns (x_dict, y_dict).
    """
    I, J, K = range(len(R)), range(len(Q)), range(len(Q[0]))
    model = gp.Model("Initial_Solution")
    model.Params.OutputFlag = 0

    # Constraint (12): x and y must be binay
    x = model.addVars(I,    vtype=GRB.BINARY, name="x")
    y = model.addVars(I, J, vtype=GRB.BINARY, name="y")

    # Constraint (3): each open hotel is assigned to exactly one node
    for i in I:
        model.addConstr(
            gp.quicksum(y[i, j] for j in J) == x[i],
            name=f"alloc_only_if_open[{i}]"
        )

    # Constraint (4): total capacity assigned to node j must cover its demand
    for j in J:
        model.addConstr(
            gp.quicksum(C[i][w] * y[i, j] for i in I for w in K)
            >= gp.quicksum(Q[j][k] for k in K),
            name=f"capacity_node[{j}]"
        )

    # (Eq. 35) minimizing the number of open hotels
    model.setObjective(gp.quicksum(x[i] for i in I), GRB.MINIMIZE)
    model.optimize()

    return (
        {i: round(x[i].X) for i in I},
        {(i, j): round(y[i, j].X) for i in I for j in J}
    )


def is_feasible(x_dict, y_dict, Q, C):
    """Verify constraints (3) and (4) for a given configuration."""
    I, J, K = range(len(C)), range(len(Q)), range(len(Q[0]))
    for i in I:
        if sum(y_dict[i, j] for j in J) != x_dict[i]:
            return False
    for j in J:
        cap = sum(C[i][w] * y_dict[i, j] for i in I for w in K)
        dem = sum(Q[j][k] for k in K)
        if cap < dem:
            return False
    return True


def local_search(s, Q, C, c, p, R, gamma):
    """Execute swap operations on nodes assigned to open hotels.
    Returns the locally optimal (x, y) and its objective value,
    avoiding a redundant lower-level solve in the caller.
    """
    I, J = range(len(R)), range(len(Q))
    current_x   = copy.deepcopy(s[0])
    best_local_y = copy.deepcopy(s[1])

    best_local_Z, _, _, _ = solve_lower_level(
        current_x, best_local_y, Q, C, c, p, R, gamma
    )

    while True:
        improved = False

        # Generate all pairwise swaps of node assignments between open hotels
        open_hotels = [i for i in I if current_x[i] == 1]
        neighbors   = []

        for idx1 in range(len(open_hotels)):
            for idx2 in range(idx1 + 1, len(open_hotels)):
                i1, i2 = open_hotels[idx1], open_hotels[idx2]
                j1 = next((j for j in J if best_local_y[i1, j] == 1), None)
                j2 = next((j for j in J if best_local_y[i2, j] == 1), None)

                if j1 is not None and j2 is not None and j1 != j2:
                    new_y = copy.deepcopy(best_local_y)
                    new_y[i1, j1], new_y[i1, j2] = 0, 1
                    new_y[i2, j2], new_y[i2, j1] = 0, 1
                    if is_feasible(current_x, new_y, Q, C):
                        neighbors.append(new_y)

        # Evaluate all neighbors and select the best one (steepest descent)
        best_neighbor_y = None
        best_neighbor_Z = float('inf')

        for n_y in neighbors:
            Z_val, _, _, _ = solve_lower_level(current_x, n_y, Q, C, c, p, R, gamma)
            if Z_val < best_neighbor_Z:
                best_neighbor_Z = Z_val
                best_neighbor_y = n_y

        if best_neighbor_Z < best_local_Z:
            best_local_Z = best_neighbor_Z
            best_local_y = copy.deepcopy(best_neighbor_y)
            improved = True

        if not improved:
            break

    return current_x, best_local_y, best_local_Z


def solve_HPP(x_fixed, y_partial_fixed, Q, C, c, p, R, gamma):
    """Solve the HPP problem (single-level relaxation) to obtain a lower bound."""
    I, J, K = range(len(R)), range(len(Q)), range(len(Q[0]))
    model = gp.Model("HPP")
    model.Params.OutputFlag = 0

    x     = model.addVars(I,                        vtype=GRB.BINARY,     name="x")
    y     = model.addVars(I, J,                     vtype=GRB.BINARY,     name="y")
    z     = model.addVars(I, J, K, K, lb=0, ub=1,   vtype=GRB.CONTINUOUS, name="z")
    u     = model.addVars(J, K,                     vtype=GRB.CONTINUOUS, name="u")
    T     = model.addVars(I,          lb=0,         vtype=GRB.CONTINUOUS, name="T")
    r     = model.addVars(J, K,                     vtype=GRB.BINARY,     name="r")
    v     = model.addVars(J, K, K,                  vtype=GRB.BINARY,     name="v")
    delta = model.addVars(I,                        vtype=GRB.BINARY,     name="delta")
    B     = model.addVars(I, J, K,    lb=0,         vtype=GRB.CONTINUOUS, name="B")

    for i in I:
        model.addConstr(x[i] == x_fixed[i])

    for (i, j), val in y_partial_fixed.items():
        if val is not None:
            model.addConstr(y[i, j] == val)

    # Constraint (3): each open hotel assigned to exactly one node
    for i in I:
        model.addConstr(
            gp.quicksum(y[i, j] for j in J) == x[i],
            name=f"alloc_only_if_open[{i}]"
        )

    # Constraint (4): capacity covers demand for each node
    for j in J:
        model.addConstr(
            gp.quicksum(C[i][w] * y[i, j] for i in I for w in K)
            >= gp.quicksum(Q[j][k] for k in K),
            name=f"capacity_node[{j}]"
        )

    # Constraint (5): assign all demand
    for j in J:
        for k in K:
            model.addConstr(
                gp.quicksum(z[i, j, k, w] for i in I for w in K) == 1,
                name=f"assign_all[{j},{k}]"
            )

    # Constraint (6): capacity per room type
    for i in I:
        for j in J:
            for w in K:
                model.addConstr(
                    gp.quicksum(Q[j][k] * z[i, j, k, w] for k in K)
                    <= C[i][w] * y[i, j],
                    name=f"cap[{i},{j},{w}]"
                )

    # UE Conditions (Eq. 24-28)
    for j in J:
        for k in K:
            model.addConstr(
                gp.quicksum(z[i, j, k, w] for i in I for w in K if w != k) <= r[j, k]
            )
            model.addConstr(
                gp.quicksum(B[i, j, k] for i in I)
                <= sum(C[i][k] for i in I) * (1 - r[j, k])
            )
            for i in I:
                model.addConstr(
                    C[i][k] * y[i, j] - gp.quicksum(Q[j][w] * z[i, j, w, k] for w in K)
                    <= B[i, j, k]
                )

    for j in J:
        for k in K:
            for w in K:
                if k != w:
                    model.addConstr(u[j, k] - u[j, w] <= (1 - v[j, k, w]) * len(K) - 1)
                    model.addConstr(gp.quicksum(z[i, j, k, w] for i in I) <= v[j, k, w])

    # Contracting cost constraints (Eq. 29-32)
    for i in I:
        actual_rev = gp.quicksum(p[i][w] * Q[j][k] * z[i, j, k, w]
                                 for j in J for k in K for w in K)
        max_rev = sum(C[i][w] * p[i][w] for w in K)

        model.addConstr(T[i] <= R[i] - actual_rev + delta[i] * (max_rev - R[i]))
        model.addConstr(T[i] >= R[i] * x[i] - actual_rev)   # was: x_fixed[i]
        model.addConstr(T[i] <= R[i] * x[i])                 # was: x_fixed[i]
        model.addConstr(T[i] <= (1 - delta[i]) * R[i])

    # Objective: minimize (leader's perspective, lower bound on worst-case cost)
    obj = (
        gp.quicksum(T[i] for i in I)
        + gp.quicksum(c[i][j] * Q[j][k] * z[i, j, k, w]
                      for i in I for j in J for k in K for w in K)
        + gp.quicksum(gamma * Q[j][k] * z[i, j, k, w]
                      for i in I for j in J for k in K for w in K if w != k)
    )
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        y_out = {(i, j): round(y[i, j].X) for i in I for j in J}
        return model.ObjVal, y_out

    return float('inf'), None


def random_allocate(x_dict, Q, C):
    """Randomly assign each open hotel to one demand node (constraint 3),
    attempting to satisfy the capacity constraint (constraint 4).
    Returns y_dict if a feasible allocation is found, None otherwise.
    """
    I, J = range(len(x_dict)), range(len(Q))
    open_hotels = [i for i in I if x_dict[i] == 1]

    for _ in range(200):
        y_candidate = {(i, j): 0 for i in I for j in J}
        for i in open_hotels:
            j = random.choice(list(J))
            y_candidate[i, j] = 1
        if is_feasible(x_dict, y_candidate, Q, C):
            return y_candidate

    return None


def perturb(s_best, Q, C, c, p, R, gamma, Z_best, max_attempts=20):
    """Execute the diversification phase (Perturbation).

    Stage 1: Randomly change the number of selected hotels (N != N*).
             HPP is used ONLY as a screening tool (LB check), NOT to generate y.
             A random feasible allocation is generated instead.
    Stage 2: Partially free the allocation (y) while keeping current hotel
             selection (x). Uses HPP to identify a promising reallocation.
    Returns the perturbed solution, or "GLOBAL_OPTIMUM" if no improvement is possible.
    """
    I = range(len(R))
    x_best, y_best = s_best
    n_hotels_best = sum(x_best.values())

    # --- Stage 1: Structural perturbation (hotel selection) ---
    for _ in range(max_attempts):
        n_hotels_new = random.choice(
            [n for n in range(1, len(I) + 1) if n != n_hotels_best]
        )
        x_perturbed = copy.deepcopy(x_best)

        if n_hotels_new > n_hotels_best:
            closed = [i for i in I if x_perturbed[i] == 0]
            for i in random.sample(closed, n_hotels_new - n_hotels_best):
                x_perturbed[i] = 1
        else:
            opened = [i for i in I if x_perturbed[i] == 1]
            for i in random.sample(opened, n_hotels_best - n_hotels_new):
                x_perturbed[i] = 0

        # HPP provides a lower bound; skip if even the best case cannot beat Z_best
        z_lb, _ = solve_HPP(x_perturbed, {}, Q, C, c, p, R, gamma)

        if z_lb < Z_best:
            y_random = random_allocate(x_perturbed, Q, C)
            if y_random is not None:
                return (x_perturbed, y_random)

            # Fallback: if random allocation fails, reuse HPP's y
            _, y_hpp = solve_HPP(x_perturbed, {}, Q, C, c, p, R, gamma)
            if y_hpp is not None:
                return (x_perturbed, y_hpp)

    # --- Stage 2: Allocation perturbation (fallback) ---
    y_partial = copy.deepcopy(y_best)
    nodes_to_reallocate = list(y_partial.keys())
    random.shuffle(nodes_to_reallocate)

    for node_key in nodes_to_reallocate:
        y_partial[node_key] = None
        z_lb, y_hpp = solve_HPP(x_best, y_partial, Q, C, c, p, R, gamma)
        if z_lb < Z_best:
            return (x_best, y_hpp)

    # Neither stage found a solution that can improve Z_best
    return "GLOBAL_OPTIMUM"


def run_ils(Q, C, c, p, R, gamma, tau_max=100):
    """Run the Iterated Local Search (ILS) algorithm.

    LEADER:   the government — decides which hotels to open and how to assign demand nodes.
    FOLLOWER: the users — maximise their cost given the leader's decision (worst-case response).

    Parameters
    ----------
    Q       : list[list[float]]  — Q[j][k]: demand of node j for room type k
    C       : list[list[float]]  — C[i][w]: capacity of hotel i for room type w
    c       : list[list[float]]  — c[i][j]: unit assignment cost (node j → hotel i)
    p       : list[list[float]]  — p[i][w]: revenue per unit of room type w at hotel i
    R       : list[float]        — R[i]: minimum revenue target for hotel i
    gamma   : float              — misplacement penalty
    tau_max : int                — maximum ILS iterations (default 100)

    Returns
    -------
    best_sol      : tuple  — (x_dict, y_dict) best solution found
    Z_best        : float  — worst-case objective value of best_sol
    best_breakdown: tuple  — (contract_cost, assign_cost, misplace_cost)
    """
    initial_sol = generate_initial_solution(Q, C, R)
    Z_best, C_cont, C_ass, C_mis = solve_lower_level(
        initial_sol[0], initial_sol[1], Q, C, c, p, R, gamma
    )

    best_sol       = copy.deepcopy(initial_sol)
    best_breakdown = (C_cont, C_ass, C_mis)
    s_current      = copy.deepcopy(initial_sol)

    for tau in range(tau_max):
        # --- Phase 1: Intensification (Local Search) ---
        local_x, local_y, Z_local = local_search(s_current, Q, C, c, p, R, gamma)
        local_sol = (local_x, local_y)

        # --- Phase 2: Update global best (solve breakdown only when needed) ---
        if Z_local < Z_best:
            _, c_c, c_a, c_m = solve_lower_level(local_x, local_y, Q, C, c, p, R, gamma)
            Z_best         = Z_local
            best_sol       = copy.deepcopy(local_sol)
            best_breakdown = (c_c, c_a, c_m)

        # --- Phase 3: Diversification (Perturbation) ---
        s_next = perturb(local_sol, Q, C, c, p, R, gamma, Z_best)

        if s_next == "GLOBAL_OPTIMUM":
            print("Global optimum confirmed by HPP bound!")
            break

        s_current = copy.deepcopy(s_next)

    return best_sol, Z_best, best_breakdown


def run_instance(file_idx, sheet_idx, output_file="ils_results.csv"):
    file_name = f"{file_idx}.xlsx"
    file_path = os.path.join("..", "quarantine_hotel_instances", file_name)

    if not os.path.exists(file_path):
        print(f"File {file_name} not found, skipping...")
        return

    data = get_data_from_file_excel(file_path, sheet_idx)
    if not data or "demand" not in data:
        return

    Q     = [row for row in data["demand"]   if len(row) > 0]
    C     = [row for row in data["capacity"] if len(row) > 0]
    c     = [row for row in data["cost"]     if len(row) > 0]
    p     = [row for row in data["price"]    if len(row) > 0]
    R     = [val for val in data["revenue"]  if val is not None]
    gamma = data["penalty"]

    if not validate_dimensions(Q, C, c, p, R):
        return

    print(f"--- Running Instance: File {file_idx}, Sheet {sheet_idx} ---")
    
    start_ils = time.time()
    s_best, Z_best, breakdown = run_ils(Q, C, c, p, R, gamma, tau_max=20)
    time_ils = time.time() - start_ils

    x_ils, _               = s_best
    num_hotels             = sum(x_ils.values())
    total_hotels           = len(x_ils)
    hotels_ratio           = f"{num_hotels}/{total_hotels}"
    c_contract, c_assign, c_misplace = breakdown

    header = [
        "file", "sheet", "objective", "time_sec", "time_formatted",
        "hotels_selected", "assignment_cost",
        "misplacement_cost", "contract_cost"
    ]

    file_exists = os.path.isfile(output_file)
    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([
            file_idx, sheet_idx, int(Z_best), int(time_ils),
            format_time(time_ils),
            hotels_ratio, int(c_assign), int(c_misplace), int(c_contract)
        ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--time-limit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop after 2 hours and save partial results (default: off)."
    )
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        default=False,
        help="Run a small test instead of all instances. Edit the test block in __main__ to customize."
    )
    args = parser.parse_args()

    start_time = time.time()
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    results_file = os.path.join(results_dir, "ils_results.csv")

    if os.path.exists(results_file):
        os.remove(results_file)

    if args.test:
        # edit here if you want to test something
        run_instance(1, 0, results_file)
        run_instance(1, 1, results_file)
        run_instance(2, 0, results_file)
        run_instance(2, 1, results_file)
    else:
        MAX_TIME = 2 * 3600
        timed_out = False
        for file_idx in range(1, 17):
            for sheet_idx in range(0, 3):
                if args.time_limit and (time.time() - start_time) >= MAX_TIME:
                    timed_out = True
                    break
                run_instance(file_idx, sheet_idx, results_file)
            if timed_out:
                break

        if timed_out:
            print(f"\nTime limit reached (2h). Partial results saved to {results_file}")

    if os.path.isfile(results_file):
        generate_plots_ils(results_file)
    print(f"Total time needed: {format_time(time.time() - start_time)}")


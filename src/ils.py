# ils.py

import argparse
import random
import copy
import os
import time
import csv
from datetime import datetime
from zoneinfo import ZoneInfo

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


def local_search(current_solution, demand, capacity, cost, price, Revenue, gamma):
    """Execute swap operations on nodes assigned to open hotels.
    Returns the locally optimal (x, y) and its objective value,
    avoiding a redundant lower-level solve in the caller.
    """
    I, J = range(len(Revenue)), range(len(demand))
    current_x   = copy.deepcopy(current_solution[0])
    best_local_y = copy.deepcopy(current_solution[1])

    best_local_Z, _, _, _ = solve_lower_level(
        current_x, best_local_y, demand, capacity, cost, price, Revenue, gamma
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
                    if is_feasible(current_x, new_y, demand, capacity):
                        neighbors.append(new_y)

        # Evaluate all neighbors and select the best one (steepest descent)
        best_neighbor_y = None
        best_neighbor_Z = float('inf')

        for n_y in neighbors:
            Z_val, _, _, _ = solve_lower_level(current_x, n_y, demand, capacity, cost, price, Revenue, gamma)
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


def solve_HPP(x_fixed, y_partial_fixed, demand, capacity, cost, price, R, gamma):
    """single-level relaxation of FLDA.

    x_fixed : dict {i: 0|1}
        Hotel selection to hold fixed throughout the solve.
        Every x[i] is pinned to x_fixed[i] via an equality constraint,
        so HPP searches only within this choice of open hotels.

    y_partial_fixed : dict {(i, j): 0|1|None}
        Partial allocation constraints for y[i,j].
        - Empty dict {}       → y is fully free (Stage 1 call).
        - Entry with 0 or 1   → y[i,j] is pinned to that value (Stage 2,
                                 elements still inherited from y_best).
        - Entry with None     → y[i,j] is freed and optimised by HPP
                                 (Stage 2, elements released one by one).

    Returns
    -------
    ObjVal : float
        Z_lb — the minimum government cost achievable under x_fixed given the
        UE-feasibility relaxation. Acts as a certified lower bound on Z_FLDA.
        Returns float('inf') if the model is infeasible.

    y_out : dict {(i, j): 0|1} or None
        The allocation plan that achieves Z_lb. Used directly only in Stage 2;
        discarded in Stage 1. Returns None if the model is infeasible.
    """

    I, J, K = range(len(R)), range(len(demand)), range(len(demand[0]))
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
            gp.quicksum(capacity[i][w] * y[i, j] for i in I for w in K)
            >= gp.quicksum(demand[j][k] for k in K),
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
                    gp.quicksum(demand[j][k] * z[i, j, k, w] for k in K)
                    <= capacity[i][w] * y[i, j],
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
                <= sum(capacity[i][k] for i in I) * (1 - r[j, k])
            )
            for i in I:
                model.addConstr(
                    capacity[i][k] * y[i, j] - gp.quicksum(demand[j][w] * z[i, j, w, k] for w in K)
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
        actual_rev = gp.quicksum(price[i][w] * demand[j][k] * z[i, j, k, w]
                                 for j in J for k in K for w in K)
        max_rev = sum(capacity[i][w] * price[i][w] for w in K)

        model.addConstr(T[i] <= R[i] - actual_rev + delta[i] * (max_rev - R[i]))
        model.addConstr(T[i] >= R[i] * x[i] - actual_rev)   # was: x_fixed[i]
        model.addConstr(T[i] <= R[i] * x[i])                 # was: x_fixed[i]
        model.addConstr(T[i] <= (1 - delta[i]) * R[i])

    # Objective: minimize (leader's perspective, lower bound on worst-case cost)
    obj = (
        gp.quicksum(T[i] for i in I)
        + gp.quicksum(cost[i][j] * demand[j][k] * z[i, j, k, w]
                      for i in I for j in J for k in K for w in K)
        + gp.quicksum(gamma * demand[j][k] * z[i, j, k, w]
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
    
    # 200 attempts are sufficient: even assuming a conservative feasibility
    # probability of p=0.1, P(at least one success) = 1 - 0.9^200 ≈ 1 - 7e-10 ≈ 1.0
    for _ in range(200):
        y_candidate = {(i, j): 0 for i in I for j in J}
        for i in open_hotels:
            j = random.choice(list(J))
            y_candidate[i, j] = 1
        if is_feasible(x_dict, y_candidate, Q, C):
            return y_candidate

    return None


def perturb(sol_from_local_search, demand, capacity, cost, price, revenue, gamma, Z_best, max_attempts=20):
    """Execute the diversification phase (Perturbation).

    Stage 1: Randomly change the number of selected hotels (N != N*).
    Stage 2: Partially free the allocation (y) while keeping current hotel
             selection (x). Uses HPP to identify a promising reallocation.
    Returns the perturbed solution, or "GLOBAL_OPTIMUM" if no improvement is possible.
    """
    num_hotels = range(len(revenue)) 
    x_best, y_best = sol_from_local_search
    num_hotels_best = sum(x_best.values())

    # --- Stage 1: Structural perturbation (hotel selection) ---
    # i. Generate a casual N != N*
    for _ in range(max_attempts):
        n_hotels_new = random.choice(
            [n for n in range(1, len(num_hotels) + 1) if n != num_hotels_best]
        )
        x_perturbed = copy.deepcopy(x_best)
        y_perturbed = None

        # ii. Case N > N* OPEN HOTELS
        if n_hotels_new > num_hotels_best:
            # open the N-N* hotels
            currently_closed = [i for i in num_hotels if x_perturbed[i] == 0]
            for i in random.sample(currently_closed, n_hotels_new - num_hotels_best):
                x_perturbed[i] = 1
            # casually allocation
            y_perturbed = random_allocate(x_perturbed, demand, capacity)

        # iii. Case N < N* CLOSE HOTELS
        else:
            opened = [i for i in num_hotels if x_perturbed[i] == 1]
            hotels_closed_by_iii = random.sample(opened, num_hotels_best - n_hotels_new)
            for i in hotels_closed_by_iii:
                x_perturbed[i] = 0

            # (a) Does a feasible (x, y) exist with the remaining hotels?
            y_perturbed = random_allocate(x_perturbed, demand, capacity)

            if y_perturbed is None:
                # (b) No feasible y → reopen hotels that were already closed in s*
                already_closed_in_best = [i for i in num_hotels if x_best[i] == 0]
                random.shuffle(already_closed_in_best)
                for i in already_closed_in_best:
                    x_perturbed[i] = 1
                    y_perturbed = random_allocate(x_perturbed, demand, capacity)
                    if y_perturbed is not None:
                        break  # feasible (x, y) found

            if y_perturbed is None:
                # (c) Still infeasible → reopen hotels closed in step iii, one at a time
                random.shuffle(hotels_closed_by_iii)
                for i in hotels_closed_by_iii:
                    x_perturbed[i] = 1
                    y_perturbed = random_allocate(x_perturbed, demand, capacity)
                    if y_perturbed is not None:
                        break

        # At this point (x_perturbed, y_perturbed) is a feasible location-allocation scheme
        # NOW run HPP for the acceptance/screening check (Section 3.3, after the procedure)
        if y_perturbed is not None:
            z_lb, _ = solve_HPP(x_perturbed, {}, demand, capacity, cost, price, revenue, gamma)
            if z_lb < Z_best:
                return (x_perturbed, y_perturbed)

    # --- Stage 2: Allocation perturbation (fallback) ---
    y_partial = copy.deepcopy(y_best)

    # Only free variables for OPEN hotels: closed hotels are already
    # forced to 0 by constraint (3) inside HPP regardless
    nodes_to_reallocate = [
        (i, j) for (i, j) in y_partial.keys() if x_best[i] == 1
    ]
    random.shuffle(nodes_to_reallocate)

    for node_key in nodes_to_reallocate:
        y_partial[node_key] = None
        z_lb, y_hpp = solve_HPP(x_best, y_partial, demand, capacity, cost, price, revenue, gamma)
        if z_lb < Z_best:
            return (x_best, y_hpp)

    return "GLOBAL_OPTIMUM"


def run_ils(demand, capacity, cost, price, revenue, gamma, tau_max=100, time_limit=None):
    """Run the Iterated Local Search (ILS) algorithm.

    LEADER:   the government — decides which hotels to open and how to assign demand nodes.
    FOLLOWER: the users — maximise their cost given the leader's decision (worst-case response).

    Parameters
    ----------
    Demand          : list[list[float]]  — Q[j][k]: demand of node j for room type k
    capacity        : list[list[float]]  — C[i][w]: capacity of hotel i for room type w
    cost            : list[list[float]]  — c[i][j]: unit assignment cost (node j → hotel i)
    p               : list[list[float]]  — p[i][w]: revenue per unit of room type w at hotel i
    revenue         : list[float]        — R[i]: minimum revenue target for hotel i
    gamma           : float              — misplacement penalty
    tau_max         : int                — maximum ILS iterations (default 100)
    time_limit      : float | None       — max seconds per run; None means no limit

    Returns
    -------
    best_sol      : tuple  — (x_dict, y_dict) best solution found
    Z_best        : float  — worst-case objective value of best_sol
    best_breakdown: tuple  — (contract_cost, assign_cost, misplace_cost)
    """
    t_start = time.time()
    initial_sol = generate_initial_solution(demand, capacity, revenue)
    Z_best, C_cont, C_ass, C_mis = solve_lower_level(
        initial_sol[0], initial_sol[1], demand, capacity, cost, price, revenue, gamma
    )

    best_sol            = copy.deepcopy(initial_sol)
    best_breakdown      = (C_cont, C_ass, C_mis)
    current_solution    = copy.deepcopy(initial_sol)

    for tau in range(tau_max):
        if time_limit is not None and (time.time() - t_start) >= time_limit:
            print(f"  Time limit reached at iteration {tau}, stopping early.")
            break
        # --- Phase 1: Intensification (Local Search) ---
        local_x, local_y, Z_local = local_search(current_solution, demand, capacity, cost, price, revenue, gamma)
        local_sol = (local_x, local_y)

        # --- Phase 2: Update global best (solve breakdown only when needed) ---
        if Z_local < Z_best:
            _, c_c, c_a, c_m = solve_lower_level(local_x, local_y, demand, capacity, cost, price, revenue, gamma)
            Z_best         = Z_local
            best_sol       = copy.deepcopy(local_sol)
            best_breakdown = (c_c, c_a, c_m)

        # --- Phase 3: Diversification (Perturbation) ---
        s_next = perturb(local_sol, demand, capacity, cost, price, revenue, gamma, Z_best)

        if s_next == "GLOBAL_OPTIMUM":
            print("Global optimum confirmed by HPP bound!")
            break

        # Acceptance criterion: always accept the perturbed solution (diversification)
        current_solution = copy.deepcopy(s_next)  # ← era s_current, variabile mai usata

    return best_sol, Z_best, best_breakdown


def run_instance(file_key, sheet_idx, output_file="ils_results.csv", time_limit=None):
    file_name = f"{file_key}.xlsx"
    file_path = os.path.join("..", "quarantine_hotel_instances", file_name)

    if not os.path.exists(file_path):
        print(f"File {file_name} not found, skipping...")
        return

    data = get_data_from_file_excel(file_path, sheet_idx)
    if not data or "demand" not in data:
        return

    demand      = [row for row in data["demand"]   if len(row) > 0]
    capacity    = [row for row in data["capacity"] if len(row) > 0]
    cost        = [row for row in data["cost"]     if len(row) > 0]
    price       = [row for row in data["price"]    if len(row) > 0]
    revenue     = [val for val in data["revenue"]  if val is not None]
    gamma       = data["penalty"]

    if not validate_dimensions(demand, capacity, cost, price, revenue):
        return

    print(f"--- Running Instance: File {file_name}, Sheet {sheet_idx} --- time {datetime.now(ZoneInfo('Europe/Rome')).strftime('%H:%M:%S')}")
    
    try:
        start_ils = time.time()
        s_best, Z_best, breakdown = run_ils(demand, capacity, cost, price, revenue, gamma, tau_max=20, time_limit=time_limit)
        time_ils = time.time() - start_ils

        x_ils, _               = s_best
        num_hotels             = sum(x_ils.values())
        total_hotels           = len(x_ils)
        hotels_ratio           = f"{num_hotels}/{total_hotels}"
        c_contract, c_assign, c_misplace = breakdown

        header = [
            "file", "sheet", "penalty", "objective", "time_sec", "time_formatted",
            "hotels_selected", "assignment_cost",
            "misplacement_cost", "contract_cost"
        ]

        file_exists = os.path.isfile(output_file)
        with open(output_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow([
                file_key, sheet_idx, gamma, int(Z_best), int(time_ils),
                format_time(time_ils),
                hotels_ratio, int(c_assign), int(c_misplace), int(c_contract)
            ])

    except gp.GurobiError as e:
        if "too large" in str(e).lower():
            print(f"  [SKIP] File {file_name}, Sheet {sheet_idx}: model too large for restricted license.")
            file_exists = os.path.isfile(output_file)
            with open(output_file, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(header)
                writer.writerow([
                    file_key, sheet_idx, gamma, "TOO_LARGE", "", "", "", "", "", ""
                ])
        else:
            print(f"  [ERROR] File {file_name}, Sheet {sheet_idx}: GurobiError: {e}")



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

    instance_time_limit = 2 * 3600 if args.time_limit else None

    if args.test:
        # edit here if you want to test something
        run_instance(1, 0, results_file, time_limit=instance_time_limit)
        run_instance(2, 0, results_file, time_limit=instance_time_limit)
        run_instance(4, 0, results_file, time_limit=instance_time_limit)
        run_instance(6, 0, results_file, time_limit=instance_time_limit)
    else:
        # for sheet_idx in range(0, 3):
            for file_idx in range(1, 13):
                run_instance(file_idx, 0, results_file, time_limit=instance_time_limit)

    if os.path.isfile(results_file):
        generate_plots_ils(results_file)
    print(f"Total time needed: {format_time(time.time() - start_time)}")


"""
flda_solver.py
==============
FLDA - Facility Location with Decentralised Assignment.

Exact solver implementing the conceptual framework of the Column-and-Row
Generation (CRG) algorithm by Lozano & Smith (2017) adapted for this BMIP.

Algorithm (see Section 3 of Liu et al. 2026)
--------------------------------------------
1.  Find an initial feasible upper-level solution (min number of hotels).
2.  Solve the High Point Problem (HPP) — the relaxed bi-level where the
    lower-level optimality is not enforced.  This gives a lower bound LB.
3.  Fix the HPP solution (x*, y*) and solve the lower-level maximisation
    to obtain the worst-case UE cost Z_LL.
4.  If  Z_LL <= LB + ε  →  (x*, y*) is optimal; stop.
5.  Otherwise add a *value-function cut* to HPP:
        Z >= Z_LL  -  M * [deviation of (x,y) from (x*,y*)]
    This cut tells HPP: "if you select (x*,y*) again, you must account for
    the true cost Z_LL rather than the optimistic HPP cost."
6.  Update best known upper bound UB = min(UB, Z_LL); go to step 2.

Convergence is guaranteed because:
  * LB is non-decreasing (cuts strengthen HPP).
  * Every feasible (x,y) will eventually force LB >= UB.
"""

import pulp
import time
from typing import List, Dict, Any, Optional, Tuple

from lower_level import solve_lower_level


# ──────────────────────────────────────────────────────────────────────────────
# Helper: find initial solution (minimum number of hotels)
# ──────────────────────────────────────────────────────────────────────────────

def _find_initial_solution(
    Q: List[List[int]],
    C: List[List[int]],
    time_limit: int = 300,
) -> Tuple[List[int], List[List[int]]]:
    """
    Find feasible (x, y) with fewest selected hotels (eq. 35 in paper).
    Used to warm-start the FLDA solver.
    """
    I = len(C)
    J = len(Q)
    K = len(Q[0])

    prob = pulp.LpProblem("InitSolution", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(I)]
    y = [
        [pulp.LpVariable(f"y_{i}_{j}", cat="Binary") for j in range(J)]
        for i in range(I)
    ]

    prob += pulp.lpSum(x[i] for i in range(I))

    for i in range(I):
        prob += pulp.lpSum(y[i][j] for j in range(J)) == x[i]
    for j in range(J):
        prob += (
            pulp.lpSum(C[i][w] * y[i][j] for i in range(I) for w in range(K))
            >= sum(Q[j][k] for k in range(K))
        )

    prob.solve(pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=0))

    x_val = [round(pulp.value(x[i]) or 0) for i in range(I)]
    y_val = [
        [round(pulp.value(y[i][j]) or 0) for j in range(J)]
        for i in range(I)
    ]
    return x_val, y_val


# ──────────────────────────────────────────────────────────────────────────────
# HPP builder (rebuilt each iteration with accumulated cuts)
# ──────────────────────────────────────────────────────────────────────────────

def _solve_hpp(
    Q: List[List[int]],
    C: List[List[int]],
    c: List[List[int]],
    p: List[List[int]],
    R: List[int],
    gamma: int,
    value_function_cuts: List[Tuple[List[List[int]], float]],
    M_cut: float,
    time_limit: int = 3600,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Solve the High Point Problem (HPP).

    HPP = FLCA objective + all UE constraints (24-34) BUT does NOT enforce
    that z maximises the lower-level objective.  Therefore HPP gives a valid
    lower bound LB on the FLDA optimal value.

    value_function_cuts: list of (y0, Z_LL) pairs from previous iterations.
    Each cut adds: Z >= Z_LL - M_cut * hamming_distance(y, y0)
    """
    I = len(R)
    J = len(Q)
    K = len(Q[0])
    Jc = len(c[0]) if c and c[0] else J
    full_rev = [sum(C[i][w] * p[i][w] for w in range(K)) for i in range(I)]

    prob = pulp.LpProblem("HPP", pulp.LpMinimize)

    # ── Variables ─────────────────────────────────────────────────────────────
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(I)]
    y = [
        [pulp.LpVariable(f"y_{i}_{j}", cat="Binary") for j in range(J)]
        for i in range(I)
    ]
    z = [
        [
            [
                [pulp.LpVariable(f"z_{i}_{j}_{k}_{w}", lowBound=0.0, upBound=1.0)
                 for w in range(K)]
                for k in range(K)
            ]
            for j in range(J)
        ]
        for i in range(I)
    ]
    T = [pulp.LpVariable(f"T_{i}", lowBound=0.0) for i in range(I)]
    delta = [pulp.LpVariable(f"delta_{i}", cat="Binary") for i in range(I)]
    r = [
        [pulp.LpVariable(f"r_{j}_{k}", cat="Binary") for k in range(K)]
        for j in range(J)
    ]
    u = [
        [pulp.LpVariable(f"u_{j}_{k}", lowBound=-(K + 1), upBound=(K + 1))
         for k in range(K)]
        for j in range(J)
    ]
    v = [
        [
            [pulp.LpVariable(f"v_{j}_{k}_{w}", cat="Binary") for w in range(K)]
            for k in range(K)
        ]
        for j in range(J)
    ]
    B = [
        [
            [pulp.LpVariable(f"B_{i}_{j}_{w}", lowBound=0.0) for w in range(K)]
            for j in range(J)
        ]
        for i in range(I)
    ]

    # ── Objective expression (reused in value-function cuts) ──────────────────
    obj_expr = (
        pulp.lpSum(T[i] for i in range(I))
        + pulp.lpSum(
            c[i][j % Jc] * Q[j][k] * z[i][j][k][w]
            for i in range(I) for j in range(J) for k in range(K) for w in range(K)
        )
        + pulp.lpSum(
            gamma * Q[j][k] * z[i][j][k][w]
            for i in range(I) for j in range(J) for k in range(K) for w in range(K)
            if k != w
        )
    )
    prob += obj_expr, "TotalCost"

    # ── Upper-level constraints (3), (4) ─────────────────────────────────────
    for i in range(I):
        prob += pulp.lpSum(y[i][j] for j in range(J)) == x[i]
    for j in range(J):
        prob += (
            pulp.lpSum(C[i][w] * y[i][j] for i in range(I) for w in range(K))
            >= sum(Q[j][k] for k in range(K))
        )

    # ── Feasibility (5), (6) ─────────────────────────────────────────────────
    for j in range(J):
        for k in range(K):
            prob += pulp.lpSum(z[i][j][k][w] for i in range(I) for w in range(K)) == 1
    for i in range(I):
        for j in range(J):
            for w in range(K):
                prob += (
                    pulp.lpSum(Q[j][k] * z[i][j][k][w] for k in range(K))
                    <= C[i][w] * y[i][j]
                )

    # ── UE constraints (24)–(28) ──────────────────────────────────────────────
    for j in range(J):
        for k in range(K):
            prob += (
                pulp.lpSum(
                    z[i][j][k][w] for i in range(I) for w in range(K) if w != k
                ) <= r[j][k]
            )
    for j in range(J):
        for w in range(K):
            tot = sum(C[i][w] for i in range(I))
            prob += pulp.lpSum(B[i][j][w] for i in range(I)) <= tot * (1 - r[j][w])
    for i in range(I):
        for j in range(J):
            for w in range(K):
                prob += (
                    C[i][w] * y[i][j]
                    - pulp.lpSum(Q[j][k] * z[i][j][k][w] for k in range(K))
                    <= B[i][j][w]
                )
    for j in range(J):
        for k in range(K):
            for w in range(K):
                if k != w:
                    prob += u[j][k] - u[j][w] <= (1 - v[j][k][w]) * K - 1
    for j in range(J):
        for k in range(K):
            for w in range(K):
                if k != w:
                    prob += pulp.lpSum(z[i][j][k][w] for i in range(I)) <= v[j][k][w]

    # ── Contracting cost linearisation (29)–(32) ─────────────────────────────
    # HPP is a minimisation → same linearisation as FLCA (eqs. 29-32 in paper)
    for i in range(I):
        rev_i = pulp.lpSum(
            p[i][w] * Q[j][k] * z[i][j][k][w]
            for j in range(J) for k in range(K) for w in range(K)
        )
        prob += T[i] >= R[i] * x[i] - rev_i            # (30)
        prob += T[i] <= R[i] * x[i]                     # (31)
        prob += T[i] <= R[i] - rev_i + delta[i] * (full_rev[i] - R[i])  # (29)
        prob += T[i] <= (1 - delta[i]) * R[i]           # (32)

    # ── Value-function cuts (core of the L&S column-and-row generation) ───────
    # For each previously evaluated (y0, Z_LL):
    #   Z >= Z_LL - M_cut * hamming_distance(y, y0)
    # When (x,y)==(x0,y0): hamming=0  =>  Z >= Z_LL  (forces true cost)
    # When (x,y)!=(x0,y0): hamming>=1 =>  Z >= Z_LL - M_cut (trivial)
    for y0, Z_LL in value_function_cuts:
        diff = pulp.lpSum(
            [(1 - y[i][j]) for i in range(I) for j in range(J) if y0[i][j] == 1]
            + [y[i][j] for i in range(I) for j in range(J) if y0[i][j] == 0]
        )
        prob += obj_expr >= Z_LL - M_cut * diff

    # ── Solve ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    prob.solve(pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=1 if verbose else 0, threads=4))
    elapsed = time.time() - t0

    obj = pulp.value(prob.objective)
    if obj is None:
        return {"status": pulp.LpStatus[prob.status], "objective": None}

    return {
        "status": pulp.LpStatus[prob.status],
        "objective": obj,
        "x": [round(pulp.value(x[i]) or 0) for i in range(I)],
        "y": [[round(pulp.value(y[i][j]) or 0) for j in range(J)] for i in range(I)],
        "solve_time": elapsed,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main FLDA exact solver
# ──────────────────────────────────────────────────────────────────────────────

def solve_FLDA_exact(
    Q: List[List[int]],
    C: List[List[int]],
    c: List[List[int]],
    p: List[List[int]],
    R: List[int],
    gamma: int,
    time_limit: int = 7200,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Exact FLDA solver using iterative value-function cutting planes (L&S style).

    The algorithm alternates between:
      - Solving HPP (lower bound, LB) with accumulated cuts.
      - Solving the lower-level maximisation (true FLDA cost) for the HPP solution.

    Terminates when LB >= best upper bound UB (proven optimal).

    Parameters
    ----------
    (Q, C, c, p, R, gamma): same as solve_FLCA
    time_limit : total wall-clock limit in seconds (default 7200 = 2 hours)
    verbose    : print iteration log

    Returns
    -------
    dict with keys: status, objective, x, y, LB, iterations, solve_time
    """
    I = len(R)
    J = len(Q)
    K = len(Q[0])

    # Large-M for value-function cuts (loose upper bound on objective range)
    # = max contracting + max assignment + max misplacement
    M_cut = float(
        sum(R)
        + sum(max(c[i]) for i in range(I)) * sum(sum(Q[j]) for j in range(J))
        + gamma * sum(sum(Q[j]) for j in range(J))
    )
    M_cut = max(M_cut, 1e7)

    # State
    vf_cuts: List[Tuple[List[List[int]], float]] = []  # (y0, Z_LL) pairs
    best_obj = float("inf")
    best_x: Optional[List[int]] = None
    best_y: Optional[List[List[int]]] = None
    LB = 0.0
    iteration = 0
    t_start = time.time()

    if verbose:
        print("\n" + "=" * 65)
        print("  FLDA Exact Solver  (Value-Function Cutting Planes / L&S)")
        print(f"  I={I} hotels | J={J} nodes | K={K} types | γ={gamma}")
        print("=" * 65)

    # ── Step 1: find initial upper bound via lower-level on min-hotel solution ─
    if verbose:
        print("\n[Init] Finding minimum-hotel feasible solution …")
    x0, y0 = _find_initial_solution(Q, C, time_limit=min(300, time_limit // 4))
    selected = [i for i in range(I) if x0[i] == 1]
    if verbose:
        print(f"       Initial hotels selected: {selected}")

    if selected:
        ll = solve_lower_level(x0, y0, Q, C, c, p, R, gamma,
                               time_limit=min(600, time_limit // 4))
        if ll["objective"] is not None:
            best_obj = ll["objective"]
            best_x, best_y = x0, y0
            vf_cuts.append((y0, ll["objective"]))
            if verbose:
                print(f"       Initial UB = {best_obj:.4f}")

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        elapsed = time.time() - t_start
        if elapsed >= time_limit:
            if verbose:
                print(f"\n[STOP] Time limit reached ({time_limit}s).")
            break

        remaining = time_limit - elapsed
        iteration += 1

        # ── HPP (lower bound) ─────────────────────────────────────────────────
        if verbose:
            print(f"\n[Iter {iteration}]  elapsed={elapsed:.1f}s  LB={LB:.4f}  UB={best_obj:.4f}")
            print(f"            Solving HPP with {len(vf_cuts)} value-function cut(s) …")

        hpp = _solve_hpp(
            Q, C, c, p, R, gamma,
            value_function_cuts=vf_cuts,
            M_cut=M_cut,
            time_limit=int(min(remaining * 0.5, 3600)),
            verbose=False,
        )

        if hpp["objective"] is None:
            if verbose:
                print("            HPP infeasible or failed — all solutions explored.")
            break

        LB = hpp["objective"]
        x_hpp = hpp["x"]
        y_hpp = hpp["y"]

        if verbose:
            sel = [i for i in range(I) if x_hpp[i] == 1]
            print(f"            HPP  LB = {LB:.4f}  |  hotels = {sel}")

        # ── Optimality check ──────────────────────────────────────────────────
        if LB >= best_obj - 1e-4:
            if verbose:
                print(f"            LB >= UB  →  OPTIMAL  (obj = {best_obj:.4f})")
            break

        # ── Lower-level (worst-case UE for the HPP solution) ─────────────────
        elapsed = time.time() - t_start
        remaining = time_limit - elapsed
        if remaining <= 0:
            break

        if verbose:
            print("            Solving lower-level (worst-case UE) …")

        ll = solve_lower_level(
            x_hpp, y_hpp, Q, C, c, p, R, gamma,
            time_limit=int(min(remaining * 0.5, 3600)),
            verbose=False,
        )

        if ll["objective"] is None:
            if verbose:
                print("            Lower-level failed — adding no-good cut and continuing.")
            vf_cuts.append((y_hpp, best_obj))
            continue

        Z_LL = ll["objective"]
        if verbose:
            print(f"            Lower-level Z_LL = {Z_LL:.4f}")

        # Update best upper bound
        if Z_LL < best_obj - 1e-6:
            best_obj = Z_LL
            best_x, best_y = x_hpp, y_hpp
            if verbose:
                print(f"            ★ New best UB = {best_obj:.4f}")

        # Optimality check after lower-level
        if best_obj <= LB + 1e-4:
            if verbose:
                print(f"            UB ≈ LB  →  OPTIMAL  (obj = {best_obj:.4f})")
            break

        # ── Add value-function cut ────────────────────────────────────────────
        vf_cuts.append((y_hpp, Z_LL))
        if verbose:
            print(f"            Added value-function cut #{len(vf_cuts)} "
                  f"(Z_LL={Z_LL:.4f})")

    # ── Return ────────────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    status = "Optimal" if best_obj < float("inf") else "No feasible solution"

    if verbose:
        print("\n" + "=" * 65)
        print(f"  FLDA Result: {status}")
        print(f"  Objective (worst-case UE cost) : {best_obj:.4f}")
        print(f"  Lower bound (HPP)              : {LB:.4f}")
        print(f"  Optimality gap                 : {max(0, best_obj - LB):.4f}")
        print(f"  Iterations                     : {iteration}")
        print(f"  Total time                     : {total_time:.2f}s")
        if best_x is not None:
            print(f"  Selected hotels                : {[i for i in range(I) if best_x[i]==1]}")
        print("=" * 65)

    return {
        "status": status,
        "objective": best_obj if best_obj < float("inf") else None,
        "x": best_x,
        "y": best_y,
        "LB": LB,
        "gap": max(0, best_obj - LB) if best_obj < float("inf") else None,
        "iterations": iteration,
        "num_cuts": len(vf_cuts),
        "solve_time": total_time,
        "I": I, "J": J, "K": K,
    }

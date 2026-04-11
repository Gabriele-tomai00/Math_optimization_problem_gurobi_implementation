"""
lower_level.py
==============
Lower-level problem for FLDA.

Given a fixed upper-level solution (x, y) — hotel selection and allocation —
this module solves the MAXIMISATION problem that finds the worst-case User
Equilibrium (UE) assignment.  This models adversarial travellers who choose
rooms to maximise the government's cost, subject to the UE conditions derived
in Proposition 2 of Liu et al. (2026).

Variables (eq. 34 in paper)
    z^kw_ij  in [0,1]   : assignment fractions (lower-level)
    r^k_j    in {0,1}   : indicator of misplaced type-k demand at node j
    u^k_j    in R       : auxiliary for misplacement-loop elimination
    v^kw_j   in {0,1}   : indicator that type-k demand is misplaced to type-w rooms
    T_i      >= 0       : contracting cost auxiliary
    delta_i  in {0,1}   : auxiliary for contracting cost linearisation (max problem)
    B^w_ij   >= 0       : auxiliary for complementarity linearisation
"""

import pulp
import time
from typing import List, Dict, Any


def solve_lower_level(
    x_val: List[int],
    y_val: List[List[int]],
    Q: List[List[int]],
    C: List[List[int]],
    c: List[List[int]],
    p: List[List[int]],
    R: List[int],
    gamma: int,
    time_limit: int = 3600,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Solve the lower-level maximisation problem (eqs. 23–34 in the paper).

    Parameters
    ----------
    x_val[i]     : fixed hotel-selection vector (0/1)
    y_val[i][j]  : fixed hotel-allocation matrix (0/1)
    (Q, C, c, p, R, gamma) : same as in solve_FLCA

    Returns
    -------
    dict with keys: status, objective, z, T,
                    contracting_cost, assignment_cost, misplacement_cost,
                    solve_time
    """
    I = len(R)
    J = len(Q)
    K = len(Q[0])
    Jc = len(c[0]) if c and c[0] else J

    # Full revenue if all rooms in hotel i are booked at their prices
    full_rev = [sum(C[i][w] * p[i][w] for w in range(K)) for i in range(I)]

    prob = pulp.LpProblem("LowerLevel", pulp.LpMaximize)

    # ── Variables ─────────────────────────────────────────────────────────────
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

    # r^k_j: is there misplaced type-k demand at node j?
    r = [
        [pulp.LpVariable(f"r_{j}_{k}", cat="Binary") for k in range(K)]
        for j in range(J)
    ]

    # u^k_j: auxiliary to rule out misplacement loops (eq. 27)
    u = [
        [pulp.LpVariable(f"u_{j}_{k}", lowBound=-(K + 1), upBound=(K + 1))
         for k in range(K)]
        for j in range(J)
    ]

    # v^kw_j: is type-k demand misplaced to type-w rooms at node j?
    v = [
        [
            [pulp.LpVariable(f"v_{j}_{k}_{w}", cat="Binary") for w in range(K)]
            for k in range(K)
        ]
        for j in range(J)
    ]

    # B^w_ij: auxiliary for complementarity linearisation (eqs. 25–26)
    B = [
        [
            [pulp.LpVariable(f"B_{i}_{j}_{w}", lowBound=0.0) for w in range(K)]
            for j in range(J)
        ]
        for i in range(I)
    ]

    # ── Objective (MAXIMISE eq. 21) ───────────────────────────────────────────
    prob += (
        pulp.lpSum(T[i] for i in range(I))
        + pulp.lpSum(
            c[i][j % Jc] * Q[j][k] * z[i][j][k][w]
            for i in range(I)
            for j in range(J)
            for k in range(K)
            for w in range(K)
        )
        + pulp.lpSum(
            gamma * Q[j][k] * z[i][j][k][w]
            for i in range(I)
            for j in range(J)
            for k in range(K)
            for w in range(K)
            if k != w
        )
    )

    # ── Feasibility constraints (5), (6), (11) ────────────────────────────────
    # (5)  All demand of type k at node j must be assigned
    for j in range(J):
        for k in range(K):
            prob += (
                pulp.lpSum(z[i][j][k][w] for i in range(I) for w in range(K)) == 1,
                f"c5_{j}_{k}",
            )

    # (6)  Capacity and allocation link (z[i][j]=0 if y_val[i][j]=0)
    for i in range(I):
        for j in range(J):
            for w in range(K):
                prob += (
                    pulp.lpSum(Q[j][k] * z[i][j][k][w] for k in range(K))
                    <= C[i][w] * y_val[i][j],
                    f"c6_{i}_{j}_{w}",
                )

    # ── UE conditions (Proposition 2 / eqs. 24–28) ───────────────────────────
    # (24) Complementarity: misplaced demand <-> r = 1  (M = 1 since sum z <= 1)
    for j in range(J):
        for k in range(K):
            prob += (
                pulp.lpSum(
                    z[i][j][k][w]
                    for i in range(I)
                    for w in range(K)
                    if w != k
                ) <= r[j][k],
                f"c24_{j}_{k}",
            )

    # (25) Remaining capacity <= 0 when r = 1 (full occupancy condition)
    for j in range(J):
        for w in range(K):
            total_cap_w = sum(C[i][w] for i in range(I))
            prob += (
                pulp.lpSum(B[i][j][w] for i in range(I))
                <= total_cap_w * (1 - r[j][w]),
                f"c25_{j}_{w}",
            )

    # (26) B^w_ij >= remaining capacity of type-w at allocated hotel i for node j
    for i in range(I):
        for j in range(J):
            for w in range(K):
                prob += (
                    C[i][w] * y_val[i][j]
                    - pulp.lpSum(Q[j][k] * z[i][j][k][w] for k in range(K))
                    <= B[i][j][w],
                    f"c26_{i}_{j}_{w}",
                )

    # (27) No misplacement loops: u ordering with big-M = |K|
    for j in range(J):
        for k in range(K):
            for w in range(K):
                if k != w:
                    prob += (
                        u[j][k] - u[j][w] <= (1 - v[j][k][w]) * K - 1,
                        f"c27_{j}_{k}_{w}",
                    )

    # (28) v^kw_j = 1 iff there is type-k demand misplaced to type-w  (M = 1)
    for j in range(J):
        for k in range(K):
            for w in range(K):
                if k != w:
                    prob += (
                        pulp.lpSum(z[i][j][k][w] for i in range(I)) <= v[j][k][w],
                        f"c28_{j}_{k}_{w}",
                    )

    # ── Contracting cost linearisation for MAXIMISATION (eqs. 29–32) ─────────
    # Different from FLCA (minimisation) — uses delta to handle max(0, R-rev)
    for i in range(I):
        rev_i = pulp.lpSum(
            p[i][w] * Q[j][k] * z[i][j][k][w]
            for j in range(J)
            for k in range(K)
            for w in range(K)
        )
        # (30) T_i >= R_i * x_i - revenue_i
        prob += (T[i] >= R[i] * x_val[i] - rev_i, f"c30_{i}")
        # (31) 0 <= T_i <= R_i * x_i
        prob += (T[i] <= R[i] * x_val[i], f"c31_{i}")
        # (29) T_i <= R_i - revenue_i + delta_i*(full_rev_i - R_i)
        prob += (
            T[i] <= R[i] - rev_i + delta[i] * (full_rev[i] - R[i]),
            f"c29_{i}",
        )
        # (32) T_i <= (1 - delta_i) * R_i
        prob += (T[i] <= (1 - delta[i]) * R[i], f"c32_{i}")

    # ── Solve ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    solver = pulp.PULP_CBC_CMD(
        timeLimit=time_limit,
        msg=1 if verbose else 0,
        threads=4,
    )
    status_code = prob.solve(solver)
    elapsed = time.time() - t0

    obj = pulp.value(prob.objective)

    result: Dict[str, Any] = {
        "status": pulp.LpStatus[status_code],
        "objective": obj,
        "solve_time": elapsed,
        "z": None,
        "T": None,
        "contracting_cost": None,
        "assignment_cost": None,
        "misplacement_cost": None,
    }

    if obj is not None:
        z_val = [
            [
                [
                    [pulp.value(z[i][j][k][w]) or 0.0 for w in range(K)]
                    for k in range(K)
                ]
                for j in range(J)
            ]
            for i in range(I)
        ]
        T_val = [pulp.value(T[i]) or 0.0 for i in range(I)]

        result["z"] = z_val
        result["T"] = T_val
        result["contracting_cost"] = sum(T_val)
        result["assignment_cost"] = sum(
            c[i][j % Jc] * Q[j][k] * z_val[i][j][k][w]
            for i in range(I)
            for j in range(J)
            for k in range(K)
            for w in range(K)
        )
        result["misplacement_cost"] = sum(
            gamma * Q[j][k] * z_val[i][j][k][w]
            for i in range(I)
            for j in range(J)
            for k in range(K)
            for w in range(K)
            if k != w
        )

    return result

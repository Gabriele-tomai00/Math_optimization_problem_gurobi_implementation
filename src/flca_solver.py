"""
flca_solver.py
==============
FLCA - Facility Location with Centralized Assignment.

Exact MILP formulation from Liu et al. (2026), EJOR.
The government jointly optimizes location (x), allocation (y), and assignment (z)
to minimize: contracting cost + assignment cost + misplacement cost.
"""

import pulp
import time
from typing import List, Dict, Any


def solve_FLCA(
    Q: List[List[int]],
    C: List[List[int]],
    c: List[List[int]],
    p: List[List[int]],
    R: List[int],
    gamma: int,
    time_limit: int = 3600,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Solve FLCA as a standard MILP (equations 13-15 in the paper).

    Parameters
    ----------
    Q[j][k]  : demand of type-k at demand node j              (J x K)
    C[i][k]  : capacity of type-k rooms at hotel i            (I x K)
    c[i][j]  : unit assignment cost, hotel i <- node j        (I x J)
    p[i][k]  : price of type-k rooms at hotel i               (I x K)
    R[i]     : target revenue for hotel i                     (length I)
    gamma    : misplacement penalty coefficient (scalar, same for all k!=w)

    Note: in the provided dataset the cost matrix has K columns. When J == K
    (which holds for all standard instances) c[i][j] == c[i][k].

    Returns
    -------
    dict with keys: status, objective, x, y, z, T,
                    contracting_cost, assignment_cost, misplacement_cost,
                    solve_time, I, J, K
    """
    I = len(R)
    J = len(Q)
    K = len(Q[0])
    Jc = len(c[0]) if c and c[0] else J   # actual columns in c (may equal K)

    prob = pulp.LpProblem("FLCA", pulp.LpMinimize)

    # ── Decision variables ────────────────────────────────────────────────────
    # x[i] in {0,1}: hotel i selected as designated quarantine hotel
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(I)]

    # y[i][j] in {0,1}: hotel i allocated to demand node j
    y = [
        [pulp.LpVariable(f"y_{i}_{j}", cat="Binary") for j in range(J)]
        for i in range(I)
    ]

    # z[i][j][k][w] in [0,1]: fraction of type-k demand at node j
    #                          assigned to type-w rooms at hotel i
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

    # T[i] >= 0: auxiliary variable for linearised contracting cost of hotel i
    T = [pulp.LpVariable(f"T_{i}", lowBound=0.0) for i in range(I)]

    # ── Objective (eq. 13) ────────────────────────────────────────────────────
    contracting = pulp.lpSum(T[i] for i in range(I))

    assignment = pulp.lpSum(
        c[i][j % Jc] * Q[j][k] * z[i][j][k][w]
        for i in range(I)
        for j in range(J)
        for k in range(K)
        for w in range(K)
    )

    misplacement = pulp.lpSum(
        gamma * Q[j][k] * z[i][j][k][w]
        for i in range(I)
        for j in range(J)
        for k in range(K)
        for w in range(K)
        if k != w
    )

    prob += contracting + assignment + misplacement, "TotalCost"

    # ── Constraints ───────────────────────────────────────────────────────────

    # (3)  sum_j y_ij = x_i  — selected hotel allocated to exactly one node
    for i in range(I):
        prob += (
            pulp.lpSum(y[i][j] for j in range(J)) == x[i],
            f"c3_{i}",
        )

    # (4)  sum_i sum_w C^w_i * y_ij >= sum_k Q^k_j  — enough rooms per node
    for j in range(J):
        prob += (
            pulp.lpSum(C[i][w] * y[i][j] for i in range(I) for w in range(K))
            >= sum(Q[j][k] for k in range(K)),
            f"c4_{j}",
        )

    # (5)  sum_i sum_w z^kw_ij = 1  — all demand of type k at j assigned
    for j in range(J):
        for k in range(K):
            prob += (
                pulp.lpSum(z[i][j][k][w] for i in range(I) for w in range(K)) == 1,
                f"c5_{j}_{k}",
            )

    # (6)  sum_k Q^k_j * z^kw_ij <= C^w_i * y_ij  — capacity & allocation link
    for i in range(I):
        for j in range(J):
            for w in range(K):
                prob += (
                    pulp.lpSum(Q[j][k] * z[i][j][k][w] for k in range(K))
                    <= C[i][w] * y[i][j],
                    f"c6_{i}_{j}_{w}",
                )

    # (14) T_i >= R_i*x_i - revenue_i  — contracting cost lower bound
    for i in range(I):
        revenue_i = pulp.lpSum(
            p[i][w] * Q[j][k] * z[i][j][k][w]
            for j in range(J)
            for k in range(K)
            for w in range(K)
        )
        prob += (T[i] >= R[i] * x[i] - revenue_i, f"c14_{i}")

    # ── Solve ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    solver = pulp.PULP_CBC_CMD(
        timeLimit=time_limit,
        msg=1 if verbose else 0,
        threads=4,
    )
    status_code = prob.solve(solver)
    elapsed = time.time() - t0

    result: Dict[str, Any] = {
        "status": pulp.LpStatus[status_code],
        "objective": pulp.value(prob.objective),
        "solve_time": elapsed,
        "I": I, "J": J, "K": K,
        "x": None, "y": None, "z": None, "T": None,
        "contracting_cost": None,
        "assignment_cost": None,
        "misplacement_cost": None,
    }

    if pulp.value(prob.objective) is not None:
        x_val = [round(pulp.value(x[i]) or 0) for i in range(I)]
        y_val = [
            [round(pulp.value(y[i][j]) or 0) for j in range(J)]
            for i in range(I)
        ]
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

        result["x"] = x_val
        result["y"] = y_val
        result["z"] = z_val
        result["T"] = [pulp.value(T[i]) or 0.0 for i in range(I)]

        # Recompute cost components from solution values
        result["contracting_cost"] = sum(
            max(
                0.0,
                R[i] - sum(
                    p[i][w] * Q[j][k] * z_val[i][j][k][w]
                    for j in range(J)
                    for k in range(K)
                    for w in range(K)
                ),
            ) * x_val[i]
            for i in range(I)
        )
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

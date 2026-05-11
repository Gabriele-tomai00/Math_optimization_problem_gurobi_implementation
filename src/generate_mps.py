# generate_mps.py
import sys
import os
import argparse
from gurobipy import Model, GRB, quicksum
from data_loader import get_data_from_file_excel, validate_dimensions

INSTANCES_DIR = os.path.join("..", "quarantine_hotel_instances")
OUTPUT_DIR    = os.path.join("..", "mps_models")


def build_and_write_mps(file_name, sheet_index):
    file_path = os.path.join(INSTANCES_DIR, file_name)
    if not os.path.exists(file_path):
        print(f"Errore: file {file_path} non trovato.")
        return False

    data = get_data_from_file_excel(file_path, sheet_index)
    if not data:
        print(f"Errore: foglio {sheet_index} non valido in {file_name}.")
        return False

    Q_raw   = [row for row in data["demand"]   if row]
    C_raw   = [row for row in data["capacity"] if row]
    c_raw   = [row for row in data["cost"]     if row]
    p_raw   = [row for row in data["price"]    if row]
    R_raw   = [val for val in data["revenue"]  if val is not None]
    penalty = data["penalty"]

    if not validate_dimensions(Q_raw, C_raw, c_raw, p_raw, R_raw):
        print(f"Errore: validazione dimensioni fallita per {file_name} sheet {sheet_index}.")
        return False

    # Index sets
    I = list(range(len(R_raw)))    # hotels
    J = list(range(len(Q_raw)))    # airports / demand nodes
    K = list(range(len(Q_raw[0]))) # room types

    # Dict accessors — key format "i,j" used throughout the model
    Q       = {f"{j},{k}": Q_raw[j][k] for j in J for k in K}
    Ccap    = {f"{i},{w}": C_raw[i][w] for i in I for w in K}
    c       = {f"{i},{j}": c_raw[i][j] for i in I for j in J}
    p_price = {f"{i},{w}": p_raw[i][w] for i in I for w in K}
    R       = {str(i):     R_raw[i]    for i in I}

    M = len(K)

    m = Model("FLDA_MIBS")
    m.setParam("OutputFlag", 0)

    # ---------- Variabili upper-level ----------
    x     = m.addVars(I, vtype=GRB.BINARY, name="x")
    y     = m.addVars(I, J, vtype=GRB.BINARY, name="y")

    # ---------- Variabili lower-level ----------
    z     = m.addVars(I, J, K, K, lb=0.0, ub=1.0, name="z")
    B     = m.addVars(I, J, K, lb=0.0, name="B")
    r     = m.addVars(J, K, vtype=GRB.BINARY, name="r")
    u     = m.addVars(J, K, vtype=GRB.CONTINUOUS, lb=0.0, name="u")
    vvar  = m.addVars(J, K, K, vtype=GRB.BINARY, name="v")
    delta = m.addVars(I, vtype=GRB.BINARY, name="delta")
    T     = m.addVars(I, lb=0.0, name="T")

    # ---------- Rinomina variabili (MibS-safe) ----------
    for i in I:
        x[i].VarName = f"x_{i}"
        for j in J:
            y[i, j].VarName = f"y_{i}_{j}"

    for i in I:
        for j in J:
            for w in K:
                B[i, j, w].VarName = f"B_{i}_{j}_{w}"
                for k in K:
                    z[i, j, k, w].VarName = f"z_{i}_{j}_{k}_{w}"

    for j in J:
        for k in K:
            r[j, k].VarName = f"r_{j}_{k}"
            u[j, k].VarName = f"u_{j}_{k}"
            for w in K:
                vvar[j, k, w].VarName = f"v_{j}_{k}_{w}"

    for i in I:
        delta[i].VarName = f"delta_{i}"
        T[i].VarName     = f"T_{i}"

    # ---------- Obiettivo (21) ----------
    obj = quicksum(T[i] for i in I)

    for i in I:
        for j in J:
            for k in K:
                for w in K:
                    obj += c[f"{i},{j}"] * Q[f"{j},{k}"] * z[i, j, k, w]
                    if k != w:
                        obj += penalty * Q[f"{j},{k}"] * z[i, j, k, w]

    m.setObjective(obj, GRB.MINIMIZE)

    # ---------- Vincoli upper-level ----------
    # (3) sum_j y_ij == x_i
    for i in I:
        m.addConstr(quicksum(y[i, j] for j in J) <= x[i], name=f"alloc_eq_le_{i}")
        m.addConstr(quicksum(y[i, j] for j in J) >= x[i], name=f"alloc_eq_ge_{i}")
        for j in J:
            m.addConstr(y[i, j] <= x[i], name=f"alloc_only_if_open_{i}_{j}")

    # (4)
    for j in J:
        m.addConstr(
            quicksum(Ccap[f"{i},{w}"] * y[i, j] for i in I for w in K)
            >= quicksum(Q[f"{j},{k}"] for k in K),
            name=f"capacity_cover_{j}"
        )

    # ---------- Vincoli lower-level ----------
    # (5) sum_{i,w} z^{kw}_{ij} == 1
    for j in J:
        for k in K:
            expr = quicksum(z[i, j, k, w] for i in I for w in K)
            m.addConstr(expr <= 1, name=f"ll_assign_le_{j}_{k}")
            m.addConstr(expr >= 1, name=f"ll_assign_ge_{j}_{k}")

    # (6)
    for i in I:
        for j in J:
            for w in K:
                m.addConstr(
                    quicksum(Q[f"{j},{k}"] * z[i, j, k, w] for k in K)
                    <= Ccap[f"{i},{w}"] * y[i, j],
                    name=f"ll_cap_{i}_{j}_{w}"
                )

    # (24)
    for j in J:
        for k in K:
            m.addConstr(
                quicksum(z[i, j, k, w] for i in I for w in K if w != k) <= r[j, k],
                name=f"v24_{j}_{k}"
            )

    # (25)
    for j in J:
        for w in K:
            m.addConstr(
                quicksum(B[i, j, w] for i in I)
                <= quicksum(Ccap[f"{i},{w}"] * (1 - r[j, w]) for i in I),
                name=f"v25_{j}_{w}"
            )

    # (26)
    for i in I:
        for j in J:
            for w in K:
                m.addConstr(
                    Ccap[f"{i},{w}"] * y[i, j]
                    - quicksum(Q[f"{j},{k}"] * z[i, j, k, w] for k in K)
                    <= B[i, j, w],
                    name=f"v26_{i}_{j}_{w}"
                )

    # (27)
    for j in J:
        for k in K:
            for w in K:
                if k != w:
                    m.addConstr(
                        u[j, k] - u[j, w] <= (1 - vvar[j, k, w]) * M - 1,
                        name=f"v27_{j}_{k}_{w}"
                    )

    # (28)
    for j in J:
        for k in K:
            for w in K:
                if k != w:
                    m.addConstr(
                        quicksum(z[i, j, k, w] for i in I) <= vvar[j, k, w],
                        name=f"v28_{j}_{k}_{w}"
                    )

    # (29)-(32)
    for i in I:
        rev = quicksum(
            p_price[f"{i},{w}"] * Q[f"{j},{k}"] * z[i, j, k, w]
            for j in J for k in K for w in K
        )
        coeff_delta = (
            sum(Ccap[f"{i},{w}"] * p_price[f"{i},{w}"] for w in K) - R[str(i)]
        )
        m.addConstr(T[i] <= R[str(i)] - rev + coeff_delta * delta[i], name=f"v29_{i}")
        m.addConstr(T[i] >= R[str(i)] * x[i] - rev,                   name=f"v30_{i}")
        m.addConstr(T[i] <= R[str(i)] * x[i],                         name=f"v31_{i}")
        m.addConstr(T[i] <= R[str(i)] * (1 - delta[i]),               name=f"v32_{i}")

    m.update()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem    = os.path.splitext(file_name)[0]
    out_path = os.path.join(OUTPUT_DIR, f"flda_{stem}_sheet{sheet_index}.mps")
    m.write(out_path)
    print(f"Scritto {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Genera file MPS per MibS da istanze Excel.")
    parser.add_argument("excel_file",    nargs="?", help="Nome file Excel (es. 1.xlsx)")
    parser.add_argument("sheet_index",  nargs="?", type=int, default=0, help="Indice foglio (default 0)")
    parser.add_argument("-a", "--all",  action="store_true", help="Genera MPS per tutti i file (1-16) e tutti i fogli (0-2)")
    args = parser.parse_args()

    if args.all:
        # for sheet_idx in range(0, 3):
        #     for file_idx in range(1, 17):
        #         build_and_write_mps(f"{file_idx}.xlsx", sheet_idx)
        for file_idx in range(1, 13):
            build_and_write_mps(f"{file_idx}.xlsx", 0)
    else:
        if not args.excel_file:
            parser.print_help()
            sys.exit(1)
        build_and_write_mps(args.excel_file, args.sheet_index)


if __name__ == "__main__":
    main()

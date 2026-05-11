# generate_aux.py
import sys
import os
import argparse

MPS_DIR = os.path.join("..", "mps_models")
AUX_DIR = os.path.join("..", "aux_models")

LL_VAR_PREFIXES = ["z_", "B_", "r_", "u_", "v_", "delta_", "T_"]

LL_CONSTR_PREFIXES = [
    "ll_assign_le_",
    "ll_assign_ge_",
    "ll_cap_",
    "v24_",
    "v25_",
    "v26_",
    "v27_",
    "v28_",
    "v29_",
    "v30_",
    "v31_",
    "v32_",
]


def build_and_write_aux(file_name, sheet_index):
    stem     = os.path.splitext(file_name)[0]
    mps_path = os.path.join(MPS_DIR, f"flda_{stem}_sheet{sheet_index}.mps")

    if not os.path.exists(mps_path):
        print(f"Errore: file MPS non trovato: {mps_path}")
        return False

    with open(mps_path, "r") as f:
        lines = f.readlines()

    # --- ordine variabili (COLUMNS) ---
    var_order = []
    seen_vars = set()
    in_columns = False
    for line in lines:
        s = line.strip()
        if s.startswith("COLUMNS"):
            in_columns = True
            continue
        if s.startswith("RHS") or s.startswith("BOUNDS") or s.startswith("ENDATA"):
            in_columns = False
        if not in_columns or "'MARKER'" in line:
            continue
        parts = s.split()
        if parts and parts[0] not in seen_vars:
            var_order.append(parts[0])
            seen_vars.add(parts[0])

    # --- ordine vincoli (ROWS, escludendo N) ---
    con_order = []
    in_rows = False
    for line in lines:
        s = line.strip()
        if s.startswith("ROWS"):
            in_rows = True
            continue
        if s.startswith("COLUMNS"):
            in_rows = False
        if not in_rows:
            continue
        parts = s.split()
        if len(parts) == 2:
            rtype, cname = parts
            if rtype != "N":
                con_order.append(cname)

    ll_var_indices = [
        i for i, v in enumerate(var_order)
        if any(v.startswith(p) for p in LL_VAR_PREFIXES)
    ]
    ll_con_indices = [
        i for i, c in enumerate(con_order)
        if any(c.startswith(p) for p in LL_CONSTR_PREFIXES)
    ]

    os.makedirs(AUX_DIR, exist_ok=True)
    aux_path = os.path.join(AUX_DIR, f"flda_{stem}_sheet{sheet_index}.aux")

    with open(aux_path, "w", newline="\n") as f:
        f.write(f"N {len(ll_var_indices)}\n")
        f.write(f"M {len(ll_con_indices)}\n")
        for i in ll_var_indices:
            f.write(f"LC {i}\n")
        for i in ll_con_indices:
            f.write(f"LR {i}\n")
        for _ in ll_var_indices:
            f.write("LO 0\n")
        # lower-level è un max → OS 1
        f.write("OS 1\n")

    print(f"Scritto {aux_path}")
    print(f"  Variabili lower-level : {len(ll_var_indices)}")
    print(f"  Vincoli  lower-level  : {len(ll_con_indices)}")
    print(f"  Avvia MibS con:")
    print(f"    mibs -Alps_instance {mps_path} -MibS_auxiliaryInfoFile {aux_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Genera file AUX per MibS da file MPS.")
    parser.add_argument("excel_file",  nargs="?", help="Nome file Excel (es. 1.xlsx)")
    parser.add_argument("sheet_index", nargs="?", type=int, default=0, help="Indice foglio (default 0)")
    parser.add_argument("-a", "--all", action="store_true", help="Genera AUX per tutti i file (1-16) e tutti i fogli (0-2)")
    args = parser.parse_args()

    if args.all:
        for sheet_idx in range(0, 3):
            for file_idx in range(1, 17):
                build_and_write_aux(f"{file_idx}.xlsx", sheet_idx)
    else:
        if not args.excel_file:
            parser.print_help()
            sys.exit(1)
        build_and_write_aux(args.excel_file, args.sheet_index)


if __name__ == "__main__":
    main()

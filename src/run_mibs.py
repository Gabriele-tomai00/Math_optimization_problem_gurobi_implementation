# run_mibs.py
import sys
import os
import argparse
import subprocess

from generate_mps import build_and_write_mps
from generate_aux import build_and_write_aux

MPS_DIR     = os.path.join("..", "mps_models")
AUX_DIR     = os.path.join("..", "aux_models")
RESULTS_DIR = os.path.join("..", "results", "mibs")


def run_instance(file_name, sheet_index):
    stem     = os.path.splitext(file_name)[0]
    mps_path = os.path.abspath(os.path.join(MPS_DIR, f"flda_{stem}_sheet{sheet_index}.mps"))
    aux_path = os.path.abspath(os.path.join(AUX_DIR, f"flda_{stem}_sheet{sheet_index}.txt"))
    out_path = os.path.abspath(os.path.join(RESULTS_DIR, f"mibs_{stem}_sheet{sheet_index}.txt"))

    print(f"\n{'='*50}")
    print(f"Istanza: {file_name}  sheet {sheet_index}")
    print(f"{'='*50}")

    if not build_and_write_mps(file_name, sheet_index):
        print(f"  Skipping: generazione MPS fallita.")
        return False
    if not build_and_write_aux(file_name, sheet_index):
        print(f"  Skipping: generazione AUX fallita.")
        return False

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cmd = [
        "mibs",
        "-Alps_instance",           mps_path,
        "-MibS_auxiliaryInfoFile",  aux_path,
        "-Alps_timeLimit 7200"
    ]
    print(f"Lancio: {' '.join(cmd)}")

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    with open(out_path, "w") as f:
        f.write(result.stdout)

    print(result.stdout, end="")
    print(f"\nOutput salvato in {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Genera MPS e AUX, poi lancia MibS per ogni istanza.\n"
            "Default (test): file 1-2, foglio 0.\n"
            "--all: tutti i file (1-16), tutti i fogli (0-2)."
        )
    )
    parser.add_argument(
        "-a", "--all",
        action="store_true",
        help="Esegui tutte le istanze (file 1-16, fogli 0-2).",
    )
    args = parser.parse_args()

    if args.all:
        for sheet_idx in range(0, 3):
            for file_idx in range(1, 17):
                run_instance(f"{file_idx}.xlsx", sheet_idx)
    else:
        for file_idx in range(1, 3):
            run_instance(f"{file_idx}.xlsx", 0)


if __name__ == "__main__":
    main()

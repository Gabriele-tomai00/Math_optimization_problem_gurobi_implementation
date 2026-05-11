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
    print(f"Instance: {file_name}  sheet {sheet_index}")
    print(f"{'='*50}")

    if not build_and_write_mps(file_name, sheet_index):
        print(f"  Skipping: MPS generation failed.")
        return False
    if not build_and_write_aux(file_name, sheet_index):
        print(f"  Skipping: AUX generation failed.")
        return False

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cmd = [
        "mibs",
        "-Alps_instance",           mps_path,
        "-MibS_auxiliaryInfoFile",  aux_path,
        "-Alps_timeLimit",          "7200"
    ]
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    with open(out_path, "w") as f:
        f.write(result.stdout)

    print(result.stdout, end="")
    print(f"\nOutput saved to {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate MPS and AUX files, then run MibS for each instance.\n"
            "Default (test): files 1-2, sheet 0.\n"
            "--all: all files (1-16), sheet 0.\n"
            "--range START END: files from START to END (inclusive), sheet 0."
        )
    )
    parser.add_argument(
        "-a", "--all",
        action="store_true",
        help="Run all instances (files 1-16).",
    )
    parser.add_argument(
        "-r", "--range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Run files from START to END inclusive (1-16).",
    )
    args = parser.parse_args()

    if args.range:
        start, end = args.range
        if not (1 <= start <= 16 and 1 <= end <= 16 and start <= end):
            parser.error("START and END must be between 1 and 16, with START <= END.")
        for file_idx in range(start, end + 1):
            run_instance(f"{file_idx}.xlsx", 0)
    elif args.all:
        for file_idx in range(1, 17):
            run_instance(f"{file_idx}.xlsx", 0)
    else:
        for file_idx in range(1, 3):
            run_instance(f"{file_idx}.xlsx", 0)


if __name__ == "__main__":
    main()

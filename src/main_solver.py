"""
main_solver.py
==============
Entry point for the quarantine hotel assignment solver.

Usage
-----
    # Run all sheets of a single Excel file:
    python main_solver.py path/to/1.xlsx

    # Run all .xlsx files in a directory:
    python main_solver.py path/to/quarantine_hotel_instances/

    # Run with synthetic data (no Excel files needed — useful for testing):
    python main_solver.py --synthetic

    # Only run FLCA (skip FLDA):
    python main_solver.py path/to/1.xlsx --flca-only

    # Only run FLDA:
    python main_solver.py path/to/1.xlsx --flda-only

    # Limit FLDA time (seconds):
    python main_solver.py path/to/1.xlsx --time-limit 3600

Notes
-----
- The cost matrix c in the Excel files has K columns (room types).
  When J == K (all standard instances), c[i][j] == c[i][k].
- Each numeric-titled sheet in an Excel file is treated as one instance.
"""

import os
import sys
import json
import time
import random
import argparse
from typing import List, Dict, Any, Optional, Tuple

import openpyxl

# Make sure local modules are importable
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import get_data_from_file_excel, validate_dimensions
from flca_solver import solve_FLCA
from flda_solver import solve_FLDA_exact


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic data generator (for quick self-testing)
# ──────────────────────────────────────────────────────────────────────────────

def generate_synthetic_instance(
    I: int = 12,
    J: int = 3,
    K: int = 3,
    seed: int = 42,
    label: str = "synthetic",
) -> Dict[str, Any]:
    """
    Generate a random but feasible problem instance.

    Follows the data ranges described in Table 2 of Liu et al. (2026):
      - demand Q^k_j  ~ U[10, 50]
      - capacity C^w_i ~ U[20, 80]   (ensured sum-feasible)
      - cost c_ij      ~ U[1, 10]
      - price p^w_i    ~ U[50, 200]
      - revenue R_i    ~ U[0.3, 0.7] * full_revenue_i
      - gamma          in {100, 200, 300, 400, 500}
    """
    rng = random.Random(seed)

    Q = [[rng.randint(10, 50) for _ in range(K)] for _ in range(J)]
    # Ensure total capacity >= total demand for feasibility
    total_demand = sum(sum(row) for row in Q)
    min_cap_per_hotel = max(5, total_demand // (I * K // 2))
    C = [[rng.randint(min_cap_per_hotel, min_cap_per_hotel * 3) for _ in range(K)]
         for _ in range(I)]

    # c is stored as I x K (same convention as Excel data)
    c = [[rng.randint(1, 10) for _ in range(K)] for _ in range(I)]
    p = [[rng.randint(50, 200) for _ in range(K)] for _ in range(I)]

    # Revenue target: 30-70 % of full potential revenue
    R = [
        int(rng.uniform(0.3, 0.7) * sum(C[i][k] * p[i][k] for k in range(K)))
        for i in range(I)
    ]
    gamma = rng.choice([100, 200, 300, 400, 500])

    return {
        "label": label,
        "Q": Q, "C": C, "c": c, "p": p, "R": R, "gamma": gamma,
        "I": I, "J": J, "K": K,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_instances_from_excel(file_path: str) -> List[Dict[str, Any]]:
    """
    Load all numeric-titled sheets from an Excel file.
    Returns a list of dicts, each containing Q, C, c, p, R, gamma, label.
    """
    instances = []
    try:
        all_data = get_data_from_file_excel(file_path, sheet_index=None)
    except Exception as exc:
        print(f"  [ERROR] Cannot read {file_path}: {exc}")
        return instances

    for sheet_name, data in all_data.items():
        Q = [row for row in data["demand"]  if row]
        C = [row for row in data["capacity"] if row]
        c = [row for row in data["cost"]    if row]
        p = [row for row in data["price"]   if row]
        R = [v   for v   in data["revenue"] if v is not None]
        gamma = data["penalty"]

        if not validate_dimensions(Q, C, c, p, R):
            print(f"  [WARN] Sheet '{sheet_name}' failed dimension check — skipped.")
            continue

        instances.append({
            "label": f"{os.path.basename(file_path)}::sheet{sheet_name}",
            "Q": Q, "C": C, "c": c, "p": p, "R": R, "gamma": gamma,
            "I": len(R), "J": len(Q), "K": len(Q[0]),
        })

    return instances


def collect_instances(path: str) -> List[Dict[str, Any]]:
    """Collect all instances from a path (file or directory)."""
    instances = []
    if os.path.isdir(path):
        files = sorted(
            f for f in os.listdir(path) if f.endswith(".xlsx")
        )
        if not files:
            print(f"No .xlsx files found in {path}")
        for fname in files:
            fpath = os.path.join(path, fname)
            print(f"\nLoading {fname} …")
            instances.extend(load_instances_from_excel(fpath))
    elif os.path.isfile(path):
        print(f"\nLoading {os.path.basename(path)} …")
        instances.extend(load_instances_from_excel(path))
    else:
        print(f"[ERROR] Path not found: {path}")
    return instances


# ──────────────────────────────────────────────────────────────────────────────
# Result printing
# ──────────────────────────────────────────────────────────────────────────────

def _bar(width: int = 65) -> str:
    return "─" * width


def print_instance_header(inst: Dict[str, Any]) -> None:
    print("\n" + "═" * 65)
    print(f"  Instance : {inst['label']}")
    print(f"  Size     : I={inst['I']} hotels  J={inst['J']} nodes  K={inst['K']} types")
    print(f"  γ (gamma): {inst['gamma']}")
    print("═" * 65)


def print_flca_result(res: Dict[str, Any], inst: Dict[str, Any]) -> None:
    I, J, K = inst["I"], inst["J"], inst["K"]
    print(f"\n  ┌{'─'*61}┐")
    print(f"  │{'FLCA — Centralized Assignment (exact MILP)':^61}│")
    print(f"  ├{'─'*61}┤")
    print(f"  │  Status        : {res['status']:<42}│")
    print(f"  │  Objective     : {res['objective'] if res['objective'] is not None else 'N/A':<42.4f}│"
          if res['objective'] is not None else
          f"  │  Objective     : {'N/A':<42}│")
    print(f"  │  Solve time    : {res['solve_time']:<42.2f}│")
    if res.get("contracting_cost") is not None:
        print(f"  ├{'─'*61}┤")
        print(f"  │  Contracting ↑ : {res['contracting_cost']:<42.2f}│")
        print(f"  │  Assignment    : {res['assignment_cost']:<42.2f}│")
        print(f"  │  Misplacement  : {res['misplacement_cost']:<42.2f}│")
    if res.get("x") is not None:
        selected = [i for i in range(I) if res["x"][i] == 1]
        print(f"  ├{'─'*61}┤")
        print(f"  │  Selected hotels : {str(selected):<40}│")
        alloc = [(i, j) for i in range(I) for j in range(J)
                 if res["y"] and res["y"][i][j] == 1]
        for i, j in alloc:
            print(f"  │    Hotel {i:2d}  →  Node {j}  {'':<35}│")
    print(f"  └{'─'*61}┘")


def print_flda_result(res: Dict[str, Any], inst: Dict[str, Any]) -> None:
    I, J = inst["I"], inst["J"]
    print(f"\n  ┌{'─'*61}┐")
    print(f"  │{'FLDA — Decentralized Assignment (exact, L&S)':^61}│")
    print(f"  ├{'─'*61}┤")
    print(f"  │  Status        : {res['status']:<42}│")
    obj_str = f"{res['objective']:.4f}" if res["objective"] is not None else "N/A"
    lb_str  = f"{res['LB']:.4f}"  if res.get("LB") is not None else "N/A"
    gap_str = f"{res['gap']:.4f}" if res.get("gap") is not None else "N/A"
    print(f"  │  Objective(UB) : {obj_str:<42}│")
    print(f"  │  Lower Bound   : {lb_str:<42}│")
    print(f"  │  Gap           : {gap_str:<42}│")
    print(f"  │  Iterations    : {res.get('iterations', 'N/A'):<42}│")
    print(f"  │  VF cuts added : {res.get('num_cuts', 'N/A'):<42}│")
    print(f"  │  Solve time    : {res['solve_time']:<42.2f}│")
    if res.get("x") is not None:
        selected = [i for i in range(I) if res["x"][i] == 1]
        print(f"  ├{'─'*61}┤")
        print(f"  │  Selected hotels : {str(selected):<40}│")
        if res.get("y"):
            alloc = [(i, j) for i in range(I) for j in range(J)
                     if res["y"][i][j] == 1]
            for i, j in alloc:
                print(f"  │    Hotel {i:2d}  →  Node {j}  {'':<35}│")
    print(f"  └{'─'*61}┘")


def print_comparison(flca: Dict[str, Any], flda: Dict[str, Any]) -> None:
    """Print FLCA vs FLDA comparison (decentralisation price of anarchy)."""
    if flca.get("objective") and flda.get("objective"):
        poa = flda["objective"] / flca["objective"]
        print(f"\n  ▶ Price of Anarchy (FLDA / FLCA) = {poa:.4f}  "
              f"(+{(poa - 1)*100:.1f}% overhead from decentralisation)")


def save_results_json(
    results: List[Dict[str, Any]],
    output_path: str = "results.json",
) -> None:
    """Save all results to a JSON file for downstream analysis."""
    # Strip non-serialisable z matrices (large, rarely needed after solving)
    clean = []
    for r in results:
        entry = {k: v for k, v in r.items() if k not in ("z",)}
        clean.append(entry)
    with open(output_path, "w") as fh:
        json.dump(clean, fh, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Per-instance solver orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def run_instance(
    inst: Dict[str, Any],
    run_flca: bool = True,
    run_flda: bool = True,
    flca_time: int = 3600,
    flda_time: int = 7200,
    verbose_solver: bool = False,
) -> Dict[str, Any]:
    """Run FLCA and/or FLDA on a single instance, return combined result dict."""
    Q, C, c, p, R, gamma = inst["Q"], inst["C"], inst["c"], inst["p"], inst["R"], inst["gamma"]

    print_instance_header(inst)

    record: Dict[str, Any] = {"label": inst["label"], "I": inst["I"],
                               "J": inst["J"], "K": inst["K"], "gamma": gamma}
    flca_res = flda_res = None

    # ── FLCA ──────────────────────────────────────────────────────────────────
    if run_flca:
        print("\n  → Solving FLCA …")
        t0 = time.time()
        flca_res = solve_FLCA(Q, C, c, p, R, gamma,
                              time_limit=flca_time, verbose=verbose_solver)
        print_flca_result(flca_res, inst)
        record["flca"] = {
            "status":           flca_res["status"],
            "objective":        flca_res["objective"],
            "solve_time":       flca_res["solve_time"],
            "contracting_cost": flca_res.get("contracting_cost"),
            "assignment_cost":  flca_res.get("assignment_cost"),
            "misplacement_cost":flca_res.get("misplacement_cost"),
            "selected_hotels":  [i for i in range(inst["I"])
                                 if flca_res.get("x") and flca_res["x"][i] == 1],
        }

    # ── FLDA ──────────────────────────────────────────────────────────────────
    if run_flda:
        print("\n  → Solving FLDA (exact, bi-level) …")
        flda_res = solve_FLDA_exact(Q, C, c, p, R, gamma,
                                    time_limit=flda_time, verbose=True)
        print_flda_result(flda_res, inst)
        record["flda"] = {
            "status":          flda_res["status"],
            "objective":       flda_res["objective"],
            "LB":              flda_res.get("LB"),
            "gap":             flda_res.get("gap"),
            "iterations":      flda_res.get("iterations"),
            "num_cuts":        flda_res.get("num_cuts"),
            "solve_time":      flda_res["solve_time"],
            "selected_hotels": [i for i in range(inst["I"])
                                if flda_res.get("x") and flda_res["x"][i] == 1],
        }

    # ── Comparison ────────────────────────────────────────────────────────────
    if flca_res and flda_res:
        print_comparison(flca_res, flda_res)
        if flca_res.get("objective") and flda_res.get("objective"):
            record["price_of_anarchy"] = flda_res["objective"] / flca_res["objective"]

    return record


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quarantine hotel assignment solver (FLCA + FLDA exact).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to .xlsx file or directory containing .xlsx files.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data instead of Excel files (for testing).",
    )
    parser.add_argument(
        "--synth-I", type=int, default=12,
        help="Number of hotels in synthetic instance (default: 12).",
    )
    parser.add_argument(
        "--synth-J", type=int, default=3,
        help="Number of demand nodes in synthetic instance (default: 3).",
    )
    parser.add_argument(
        "--synth-K", type=int, default=3,
        help="Number of room types in synthetic instance (default: 3).",
    )
    parser.add_argument(
        "--synth-seed", type=int, default=42,
        help="Random seed for synthetic data (default: 42).",
    )
    parser.add_argument(
        "--flca-only", action="store_true",
        help="Run only FLCA (skip FLDA).",
    )
    parser.add_argument(
        "--flda-only", action="store_true",
        help="Run only FLDA (skip FLCA).",
    )
    parser.add_argument(
        "--time-limit", type=int, default=7200,
        help="Time limit in seconds for FLDA (default: 7200).",
    )
    parser.add_argument(
        "--flca-time", type=int, default=3600,
        help="Time limit in seconds for FLCA (default: 3600).",
    )
    parser.add_argument(
        "--sheet", type=int, default=None,
        help="If set, process only this sheet index (0-based) of each Excel file.",
    )
    parser.add_argument(
        "--output", type=str, default="results.json",
        help="JSON file to write results to (default: results.json).",
    )
    parser.add_argument(
        "--verbose-solver", action="store_true",
        help="Show raw CBC solver output.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_flca = not args.flda_only
    run_flda = not args.flca_only

    # ── Collect instances ──────────────────────────────────────────────────────
    instances: List[Dict[str, Any]] = []

    if args.synthetic:
        print(f"\nGenerating synthetic instance "
              f"(I={args.synth_I}, J={args.synth_J}, K={args.synth_K}, "
              f"seed={args.synth_seed}) …")
        inst = generate_synthetic_instance(
            I=args.synth_I,
            J=args.synth_J,
            K=args.synth_K,
            seed=args.synth_seed,
        )
        instances.append(inst)
    elif args.path is not None:
        instances = collect_instances(args.path)
    else:
        # Auto-discover relative to the script
        default_dirs = [
            os.path.join(os.path.dirname(__file__), "..", "quarantine_hotel_instances"),
            os.path.join(os.path.dirname(__file__), "quarantine_hotel_instances"),
        ]
        found = False
        for d in default_dirs:
            d = os.path.abspath(d)
            if os.path.isdir(d):
                print(f"\nAuto-discovered data directory: {d}")
                instances = collect_instances(d)
                found = True
                break
        if not found:
            print("\nNo path specified and no data directory found.")
            print("Use --synthetic for a quick test, or pass the path to your .xlsx files.")
            parser.print_help()
            sys.exit(1)

    if not instances:
        print("\nNo valid instances found. Exiting.")
        sys.exit(1)

    print(f"\n{'='*65}")
    print(f"  {len(instances)} instance(s) to solve")
    print(f"  FLCA: {'yes' if run_flca else 'no'}  |  "
          f"FLDA: {'yes' if run_flda else 'no'}  |  "
          f"FLDA time limit: {args.time_limit}s")
    print(f"{'='*65}")

    # ── Solve each instance ───────────────────────────────────────────────────
    all_results = []
    for idx, inst in enumerate(instances):
        print(f"\n[{idx + 1}/{len(instances)}]")
        try:
            rec = run_instance(
                inst,
                run_flca=run_flca,
                run_flda=run_flda,
                flca_time=args.flca_time,
                flda_time=args.time_limit,
                verbose_solver=args.verbose_solver,
            )
            all_results.append(rec)
        except Exception as exc:
            print(f"  [ERROR] Instance {inst['label']} failed: {exc}")
            import traceback
            traceback.print_exc()

    # ── Summary ───────────────────────────────────────────────────────────────
    if len(all_results) > 1:
        print(f"\n{'═'*65}")
        print(f"  SUMMARY  ({len(all_results)} instances)")
        print(f"{'═'*65}")
        header = f"  {'Instance':<30} {'FLCA':>10} {'FLDA':>10} {'PoA':>8}"
        print(header)
        print(f"  {'-'*60}")
        for r in all_results:
            flca_obj = r.get("flca", {}).get("objective")
            flda_obj = r.get("flda", {}).get("objective")
            poa      = r.get("price_of_anarchy")
            label    = r["label"][-30:]
            print(
                f"  {label:<30} "
                f"{(f'{flca_obj:.1f}' if flca_obj else 'N/A'):>10} "
                f"{(f'{flda_obj:.1f}' if flda_obj else 'N/A'):>10} "
                f"{(f'{poa:.3f}' if poa else 'N/A'):>8}"
            )

    # ── Save JSON ─────────────────────────────────────────────────────────────
    save_results_json(all_results, args.output)
    print(f"\nDone. Total instances solved: {len(all_results)}")


if __name__ == "__main__":
    main()

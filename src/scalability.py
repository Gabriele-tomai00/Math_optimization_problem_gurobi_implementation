# scalability.py
# Scalability analysis: ILS (heuristic) vs MibS (exact method).
#
# Metrics per algorithm:
#   ILS  → gap from lower bound, computational time
#   MibS → optimality gap (equivalent of Gurobi gap), computational time
#   Then → direct comparison (objective values, Δ%)
#
# Expected directory layout (relative to repo root):
#   results/mibs/mibs_<N>_sheet<S>.txt   ← MibS raw output files
#   results/ils_results.csv              ← ILS results (written by ils.py)
#   results/scalability_results.csv      ← output: merged comparison table
#   results/plots/                       ← output: .png plots

import os
import re
import glob

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

# ── Paths ──────────────────────────────────────────────────────────────────────

# scalability.py lives inside src/, so go up two levels to reach the repo root
REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIBS_DIR   = os.path.join(REPO_ROOT, "results", "mibs")
ILS_CSV    = os.path.join(REPO_ROOT, "results_full", "ils_results.csv")
OUTPUT_CSV = os.path.join(REPO_ROOT, "results", "scalability_results.csv")
PLOTS_DIR  = os.path.join(REPO_ROOT, "results", "plots")


# ── MibS parsing ──────────────────────────────────────────────────────────────

def parse_mibs_file(path: str) -> dict:
    """
    Extract key metrics from a single MibS output text file.

    Fields returned
    ---------------
    instance     : int   file index parsed from the filename
    sheet        : int   sheet index parsed from the filename
    ul_vars      : int   number of upper-level variables (problem size proxy)
    ll_vars      : int   number of lower-level variables
    lower_bound  : float best proven lower bound at end of search
    best_obj     : float best feasible (upper-bound) solution found; None if no solution
    opt_gap_pct  : float MibS optimality gap = (best_obj - LB) / LB * 100
                         0.0  -> proven optimal
                         >0   -> feasible but not proven optimal (e.g. timeout)
                         None -> no feasible solution found
    wall_time_s  : float total wall-clock time in seconds
    status       : str   "optimal" | "time_limit" | "no_solution"
    open_hotels  : int   number of hotels selected in the best solution
    """
    with open(path) as f:
        text = f.read()

    row = {}

    # --- Instance and sheet from filename (mibs_<N>_sheet<S>.txt) ---
    m = re.search(r"mibs_(\d+)_sheet(\d+)", os.path.basename(path))
    if m:
        row["instance"] = int(m.group(1))
        row["sheet"]    = int(m.group(2))

    # --- Problem size ---
    m = re.search(r"Number of UL Variables:\s*(\d+)", text)
    row["ul_vars"] = int(m.group(1)) if m else None

    # ll_vars appears twice in the file; take the last occurrence (inside the
    # "Analyzing problem structure" section)
    ll_matches = re.findall(r"Number of LL Variables:\s*(\d+)", text)
    row["ll_vars"] = int(ll_matches[-1]) if ll_matches else None

    # --- Lower bound: last numeric value in the "Lower Bound" column of the
    #     branch-and-bound search table.
    #     Table format:  "  <nodes>   <UB>   <LB>   <gap>%   <time>   <left>"
    #     We match rows that have a gap% value (i.e. a feasible UB exists).
    lb_matches = re.findall(
        r"^\s*\d+\s+[\d.]+\s+([\d.]+)\s+[\d.]+%",
        text, re.MULTILINE
    )
    row["lower_bound"] = float(lb_matches[-1]) if lb_matches else None

    # --- Best feasible objective (upper bound on optimum) ---
    m = re.search(r"Alps0260I Best solution found had quality\s*([\d.]+)", text)
    row["best_obj"] = float(m.group(1)) if m else None

    # --- MibS optimality gap (equivalent of Gurobi's MIP gap) ---
    # Reported directly by MibS as "Relative optimality gap is X%"
    m = re.search(r"Blis0058I Relative optimality gap is\s*([\d.]+)%", text)
    row["opt_gap_pct"] = float(m.group(1)) if m else None

    # --- Wall-clock time ---
    m = re.search(r"Alps0278I Search wall-clock time:\s*([\d.]+)\s*seconds", text)
    row["wall_time_s"] = float(m.group(1)) if m else None

    # --- Search status ---
    if "Alps0208I Search completed." in text:
        row["status"] = "optimal"
    elif "Alps0209I Search time limit exceeded." in text:
        row["status"] = "time_limit"
    else:
        row["status"] = "no_solution"

    # --- Open hotels: "x[i] = 1" lines in the solution block ---
    row["open_hotels"] = len(re.findall(r"^x\[\d+\] = 1", text, re.MULTILINE))

    return row


def load_mibs_results(mibs_dir: str) -> pd.DataFrame:
    """Parse all MibS output files in mibs_dir and return a DataFrame."""
    files = sorted(glob.glob(os.path.join(mibs_dir, "mibs_*_sheet*.txt")))
    if not files:
        raise FileNotFoundError(f"No MibS output files found in: {mibs_dir}")
    rows = [parse_mibs_file(f) for f in files]
    return pd.DataFrame(rows).sort_values(["instance", "sheet"]).reset_index(drop=True)


# ── ILS loading ───────────────────────────────────────────────────────────────

def load_ils_results(ils_csv: str) -> pd.DataFrame:
    """
    Load ILS results CSV (written by ils.py).
    Renames columns for clarity and drops empty/invalid rows.
    """
    df = pd.read_csv(ils_csv)
    df = df.rename(columns={"file": "instance", "objective": "ils_obj"})
    df = df.dropna(subset=["instance"])
    df = df[pd.to_numeric(df["ils_obj"], errors="coerce").notna()].copy()
    df["ils_obj"]  = df["ils_obj"].astype(float)
    df["time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce")
    return df


# ── Comparison table ──────────────────────────────────────────────────────────

def build_comparison(mibs_df: pd.DataFrame, ils_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge ILS and MibS on (instance, sheet) and compute all gap metrics.

    Metrics computed
    ----------------
    ils_gap_from_lb : (ILS_obj - LB) / LB * 100
        How far the ILS heuristic solution is from the best proven lower bound.
        Lower is better. 0% would mean ILS found the proven optimum.

    delta_pct : (MibS_obj - ILS_obj) / ILS_obj * 100
        Direct comparison: positive = MibS is worse than ILS.
    """
    ils_agg = (
        ils_df.groupby(["instance", "sheet"])
        .agg(ils_obj=("ils_obj", "mean"), ils_time_s=("time_sec", "mean"))
        .reset_index()
    )

    df = pd.merge(mibs_df, ils_agg, on=["instance", "sheet"], how="inner")

    # ILS gap from lower bound
    df["ils_gap_from_lb"] = df.apply(
        lambda r: (
            round((r["ils_obj"] - r["lower_bound"]) / r["lower_bound"] * 100, 2)
            if pd.notna(r["lower_bound"]) and r["lower_bound"] != 0
            else None
        ),
        axis=1,
    )

    # Direct comparison gap
    df["delta_pct"] = df.apply(
        lambda r: (
            round((r["best_obj"] - r["ils_obj"]) / r["ils_obj"] * 100, 2)
            if pd.notna(r["best_obj"]) and r["ils_obj"] != 0
            else None
        ),
        axis=1,
    )

    cols = [
        "instance", "sheet", "ul_vars", "ll_vars",
        # ILS metrics
        "ils_obj", "ils_time_s", "ils_gap_from_lb",
        # MibS metrics
        "best_obj", "lower_bound", "opt_gap_pct", "wall_time_s",
        # Comparison
        "delta_pct",
        "status", "open_hotels",
    ]
    return df[cols].sort_values(["instance", "sheet"]).reset_index(drop=True)


# ── Plotting ───────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _setup_xaxis(ax, labels):
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")


def plot_all(df: pd.DataFrame) -> None:
    """
    Generate 5 plots, grouped by algorithm then comparison:

      ILS  (1) Computational time
           (2) Gap from lower bound

      MibS (3) Computational time
           (4) Optimality gap

      Both (5) Objective value comparison (ILS vs MibS)
    """
    df = df.copy()
    df["label"] = df["instance"].astype(str) + "-s" + df["sheet"].astype(str)
    labels = df["label"].tolist()
    x = list(range(len(df)))

    # ── (1) ILS — Computational time ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, df["ils_time_s"], color="steelblue", edgecolor="white")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("ILS — Computational Time per Instance")
    _setup_xaxis(ax, labels)
    _save(fig, "1_ils_time.png")

    # ── (2) ILS — Gap from lower bound ────────────────────────────────────────
    gap_ils = df.dropna(subset=["ils_gap_from_lb"])
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(gap_ils)), gap_ils["ils_gap_from_lb"],
           color="steelblue", edgecolor="white")
    ax.set_ylabel("Gap from LB (%) = (ILS obj − LB) / LB × 100")
    ax.set_title("ILS — Gap from Lower Bound\n(how far ILS is from the proven optimum bound)")
    _setup_xaxis(ax, gap_ils["label"].tolist())
    _save(fig, "2_ils_gap_from_lb.png")

    # ── (3) MibS — Computational time ─────────────────────────────────────────
    colors_time = [
        "tomato" if s == "time_limit" else
        "orange" if s == "no_solution" else
        "salmon"
        for s in df["status"]
    ]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, df["wall_time_s"], color=colors_time, edgecolor="white")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("MibS — Computational Time per Instance")
    legend_elements = [
        Patch(facecolor="salmon", label="Optimal"),
        Patch(facecolor="tomato", label="Time limit hit"),
        Patch(facecolor="orange", label="No solution found"),
    ]
    ax.legend(handles=legend_elements)
    _setup_xaxis(ax, labels)
    _save(fig, "3_mibs_time.png")

    # ── (4) MibS — Optimality gap ─────────────────────────────────────────────
    gap_mibs = df.dropna(subset=["opt_gap_pct"])
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(gap_mibs)), gap_mibs["opt_gap_pct"],
           color="salmon", edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Optimality gap (%) = (best obj − LB) / LB × 100")
    ax.set_title("MibS — Optimality Gap\n(0% = proven optimal; higher = search terminated early)")
    _setup_xaxis(ax, gap_mibs["label"].tolist())
    _save(fig, "4_mibs_opt_gap.png")

    # ── (5) Comparison — Objective values ILS vs MibS ─────────────────────────
    has_mibs = df["best_obj"].notna()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, df["ils_obj"], marker="o", label="ILS", color="steelblue")
    ax.plot(
        [i for i, v in zip(x, has_mibs) if v],
        df.loc[has_mibs, "best_obj"],
        marker="s", label="MibS", color="tomato", linestyle="--"
    )
    ax.set_ylabel("Objective value (lower is better)")
    ax.set_title("Comparison — Objective Value: ILS vs MibS")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    _setup_xaxis(ax, labels)
    _save(fig, "5_comparison_objective.png")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading MibS results...")
    mibs_df = load_mibs_results(MIBS_DIR)
    print(f"  {len(mibs_df)} MibS instances loaded.")

    print("Loading ILS results...")
    ils_df = load_ils_results(ILS_CSV)
    print(f"  {len(ils_df)} ILS rows loaded.")

    print("Building comparison table...")
    df = build_comparison(mibs_df, ils_df)
    print(f"  {len(df)} matched instances.\n")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Comparison table saved to: {OUTPUT_CSV}\n")

    # Summary table
    print(f"{'inst':>4} {'sh':>2} {'ul':>5} | "
          f"{'ILS obj':>10} {'ILS t(s)':>8} {'ILS gap-LB%':>11} | "
          f"{'MibS obj':>10} {'MibS t(s)':>9} {'MibS gap%':>9} | "
          f"{'Δ%':>7}  status")
    print("─" * 100)
    for _, r in df.iterrows():
        def fmt(v, w, decimals=1):
            return f"{v:>{w}.{decimals}f}" if pd.notna(v) else "N/A".rjust(w)
        print(
            f"{int(r['instance']):>4} {int(r['sheet']):>2} {r['ul_vars']:>5} | "
            f"{r['ils_obj']:>10.0f} {r['ils_time_s']:>8.1f} {fmt(r['ils_gap_from_lb'], 11)} | "
            f"{fmt(r['best_obj'], 10, 0)} {fmt(r['wall_time_s'], 9)} {fmt(r['opt_gap_pct'], 9, 2)} | "
            f"{fmt(r['delta_pct'], 7)}  {r['status']}"
        )

    print()
    solved = df[df["status"] == "optimal"]
    print(f"MibS solved to optimality : {len(solved)}/{len(df)} instances")
    print(f"MibS hit time limit       : {(df['status'] == 'time_limit').sum()} instances")
    print(f"MibS found no solution    : {(df['status'] == 'no_solution').sum()} instances")

    if len(solved) > 0:
        print(f"\nOn optimally-solved instances:")
        print(f"  Avg ILS gap from LB  : {solved['ils_gap_from_lb'].mean():.1f}%")
        print(f"  Avg MibS opt. gap    : {solved['opt_gap_pct'].mean():.1f}%  (should be 0%)")
        print(f"  Avg Δ% (MibS vs ILS) : {solved['delta_pct'].mean():.1f}%")

    print("\nGenerating plots...")
    plot_all(df)
    print("Done.")


if __name__ == "__main__":
    main()
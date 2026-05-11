# scalability.py
# Scalability analysis: ILS vs MibS comparison.
# Parses MibS output files, loads ILS results CSV, computes gaps,
# saves a summary CSV, and generates scalability plots.
#
# Expected directory layout (relative to repo root):
#   results/mibs/mibs_<N>_sheet<S>.txt   ← MibS raw output files
#   results/ils_results.csv              ← ILS results (written by ils.py)
#   results/scalability_results.csv      ← output: merged comparison table
#   results/                             ← output: plots (.png)

import os
import re
import glob
import csv

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Paths ──────────────────────────────────────────────────────────────────────

# scalability.py lives inside src/, so go up two levels to reach the repo root
REPO_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIBS_DIR       = os.path.join(REPO_ROOT, "results", "mibs")
ILS_CSV        = os.path.join(REPO_ROOT, "results_full", "ils_results.csv")
OUTPUT_CSV     = os.path.join(REPO_ROOT, "results", "scalability_results.csv")
PLOTS_DIR      = os.path.join(REPO_ROOT, "results")

# ── MibS parsing ──────────────────────────────────────────────────────────────

def parse_mibs_file(path: str) -> dict:
    """
    Extract key metrics from a single MibS output text file.

    Returns a dict with:
      instance    : int   - file index (from filename)
      sheet       : int   - sheet index (from filename)
      ul_vars     : int   - number of upper-level variables (= hotels + nodes)
      ll_vars     : int   - number of lower-level variables
      best_obj    : float - best feasible objective found (None if no solution)
      wall_time_s : float - wall-clock time in seconds
      opt_gap_pct : float - relative optimality gap % (0.0 if proven optimal)
      status      : str   - "optimal" | "time_limit" | "no_solution"
      open_hotels : int   - number of hotels selected in best solution
    """
    with open(path) as f:
        text = f.read()

    row = {}

    # --- Instance index and sheet from filename (mibs_<N>_sheet<S>.txt) ---
    m = re.search(r"mibs_(\d+)_sheet(\d+)", os.path.basename(path))
    if m:
        row["instance"] = int(m.group(1))
        row["sheet"]    = int(m.group(2))

    # --- Problem size ---
    m = re.search(r"Number of UL Variables:\s*(\d+)", text)
    row["ul_vars"] = int(m.group(1)) if m else None

    # ll_vars appears twice: use the one inside "Analyzing problem structure"
    ll_matches = re.findall(r"Number of LL Variables:\s*(\d+)", text)
    row["ll_vars"] = int(ll_matches[-1]) if ll_matches else None

    # --- Solution quality ---
    m = re.search(r"Alps0260I Best solution found had quality\s*([\d.]+)", text)
    row["best_obj"] = float(m.group(1)) if m else None

    m = re.search(r"Blis0058I Relative optimality gap is\s*([\d.]+)%", text)
    row["opt_gap_pct"] = float(m.group(1)) if m else None

    # --- Timing ---
    m = re.search(r"Alps0278I Search wall-clock time:\s*([\d.]+)\s*seconds", text)
    row["wall_time_s"] = float(m.group(1)) if m else None

    # --- Status ---
    if "Alps0208I Search completed." in text:
        row["status"] = "optimal"
    elif "Alps0209I Search time limit exceeded." in text:
        row["status"] = "time_limit"
    else:
        row["status"] = "no_solution"

    # --- Open hotels: count "x[i] = 1" lines in the solution block ---
    row["open_hotels"] = len(re.findall(r"^x\[\d+\] = 1", text, re.MULTILINE))

    return row


def load_mibs_results(mibs_dir: str) -> pd.DataFrame:
    """Parse all MibS output files in mibs_dir and return a DataFrame."""
    files = sorted(glob.glob(os.path.join(mibs_dir, "mibs_*_sheet*.txt")))
    if not files:
        raise FileNotFoundError(f"No MibS output files found in: {mibs_dir}")
    rows = [parse_mibs_file(f) for f in files]
    df = pd.DataFrame(rows).sort_values(["instance", "sheet"]).reset_index(drop=True)
    return df


# ── ILS loading ───────────────────────────────────────────────────────────────

def load_ils_results(ils_csv: str) -> pd.DataFrame:
    """
    Load ILS results CSV (written by ils.py).
    Expected columns: file, sheet, objective, time_sec, ...
    Renames 'file' -> 'instance' and 'objective' -> 'ils_obj' for clarity.
    """
    df = pd.read_csv(ils_csv)
    df = df.rename(columns={"file": "instance", "objective": "ils_obj"})

    # Drop empty rows (trailing blank lines) and non-numeric objectives (e.g. "TOO_LARGE")
    df = df.dropna(subset=["instance"])
    df = df[pd.to_numeric(df["ils_obj"], errors="coerce").notna()].copy()
    df["ils_obj"]  = df["ils_obj"].astype(float)
    df["time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce")
    return df


# ── Comparison table ───────────────────────────────────────────────────────────

def build_comparison(mibs_df: pd.DataFrame, ils_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge ILS and MibS results on (instance, sheet) and compute the gap:

        delta_pct = (mibs_obj - ils_obj) / ils_obj * 100

    A positive delta_pct means MibS found a WORSE solution than ILS
    (higher objective = worse, since this is a minimisation problem).
    """
    # Average ILS over repetitions (ils.py already saves the average, but
    # if multiple rows exist per instance/sheet, we take the mean)
    ils_agg = (
        ils_df.groupby(["instance", "sheet"])
        .agg(ils_obj=("ils_obj", "mean"), ils_time_s=("time_sec", "mean"))
        .reset_index()
    )

    merged = pd.merge(
        mibs_df,
        ils_agg,
        on=["instance", "sheet"],
        how="inner",
    )

    # Gap: only meaningful when MibS found a feasible solution
    merged["delta_pct"] = merged.apply(
        lambda r: (
            round((r["best_obj"] - r["ils_obj"]) / r["ils_obj"] * 100, 2)
            if pd.notna(r["best_obj"]) and r["ils_obj"] != 0
            else None
        ),
        axis=1,
    )

    # Readable status label for plots
    merged["mibs_status_label"] = merged["status"].map({
        "optimal":    "MibS optimal",
        "time_limit": "MibS timeout",
        "no_solution":"MibS no sol.",
    })

    cols = [
        "instance", "sheet",
        "ul_vars", "ll_vars",
        "ils_obj", "best_obj", "delta_pct",
        "ils_time_s", "wall_time_s",
        "opt_gap_pct", "open_hotels",
        "status", "mibs_status_label",
    ]
    return merged[cols].sort_values(["instance", "sheet"]).reset_index(drop=True)


# ── Plotting ───────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str) -> None:
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_all(df: pd.DataFrame) -> None:
    """Generate all scalability plots from the comparison DataFrame."""

    # Use a label that combines instance and sheet for the x-axis
    df = df.copy()
    df["label"] = df["instance"].astype(str) + "-s" + df["sheet"].astype(str)

    x = range(len(df))
    labels = df["label"].tolist()

    # ── Plot 1: Solve time comparison (ILS vs MibS) ───────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, df["ils_time_s"],   marker="o", label="ILS",  color="steelblue")
    ax.plot(x, df["wall_time_s"],  marker="s", label="MibS", color="tomato")

    # Mark timeout instances
    timeouts = df["status"] == "time_limit"
    ax.scatter(
        [i for i, t in zip(x, timeouts) if t],
        df.loc[timeouts, "wall_time_s"],
        marker="x", s=120, color="darkred", zorder=5, label="MibS timeout"
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Scalability: Solve Time — ILS vs MibS")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "scalability_time.png")

    # ── Plot 2: Gap Δ% (MibS worse than ILS by how much) ─────────────────────
    gap_df = df.dropna(subset=["delta_pct"])
    x_gap  = range(len(gap_df))
    colors = ["tomato" if g >= 0 else "steelblue" for g in gap_df["delta_pct"]]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(x_gap, gap_df["delta_pct"], color=colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    ax.set_xticks(list(x_gap))
    ax.set_xticklabels(gap_df["label"].tolist(), rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Δ% = (MibS obj − ILS obj) / ILS obj × 100")
    ax.set_title("Solution Quality Gap: MibS vs ILS\n(positive = MibS is worse)")
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, "scalability_gap.png")

    # ── Plot 3: Objective values side by side ─────────────────────────────────
    has_mibs = df["best_obj"].notna()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, df["ils_obj"],  marker="o", label="ILS obj",  color="steelblue")
    ax.plot(
        [i for i, v in zip(x, has_mibs) if v],
        df.loc[has_mibs, "best_obj"],
        marker="s", label="MibS obj", color="tomato", linestyle="--"
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Objective value (lower is better)")
    ax.set_title("Objective Value: ILS vs MibS")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    _save(fig, "scalability_objective.png")

    # ── Plot 4: Time vs problem size (ul_vars) ────────────────────────────────
    size_df = df.dropna(subset=["ul_vars", "ils_time_s", "wall_time_s"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(size_df["ul_vars"], size_df["ils_time_s"],  label="ILS",  color="steelblue", alpha=0.8)
    ax.scatter(size_df["ul_vars"], size_df["wall_time_s"], label="MibS", color="tomato",    alpha=0.8)
    ax.set_xlabel("Number of upper-level variables (problem size)")
    ax.set_ylabel("Solve time (seconds)")
    ax.set_title("Scalability: Time vs Problem Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "scalability_time_vs_size.png")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading MibS results...")
    mibs_df = load_mibs_results(MIBS_DIR)
    print(f"  Found {len(mibs_df)} MibS instances.")

    print("Loading ILS results...")
    ils_df = load_ils_results(ILS_CSV)
    print(f"  Found {len(ils_df)} ILS rows.")

    print("Building comparison table...")
    df = build_comparison(mibs_df, ils_df)
    print(f"  Matched {len(df)} instances.\n")

    # Save CSV
    os.makedirs(PLOTS_DIR, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Results saved to: {OUTPUT_CSV}")

    # Print summary table
    print(f"\n{'inst':>4}  {'sh':>2}  {'ul_vars':>7}  {'ILS obj':>10}  "
          f"{'MibS obj':>10}  {'Δ%':>7}  {'ILS(s)':>7}  {'MibS(s)':>8}  status")
    print("-" * 80)
    for _, r in df.iterrows():
        mibs_obj_str = f"{r['best_obj']:>10.0f}" if pd.notna(r["best_obj"]) else "     N/A  "
        gap_str      = f"{r['delta_pct']:>7.1f}" if pd.notna(r["delta_pct"]) else "    N/A"
        mibs_time    = f"{r['wall_time_s']:>8.1f}" if pd.notna(r["wall_time_s"]) else "     N/A"
        print(
            f"{r['instance']:>4}  {r['sheet']:>2}  {r['ul_vars']:>7}  "
            f"{r['ils_obj']:>10.0f}  {mibs_obj_str}  {gap_str}  "
            f"{r['ils_time_s']:>7.1f}  {mibs_time}  {r['status']}"
        )

    solved = df[df["status"] == "optimal"]
    if len(solved) > 0:
        print(f"\nAverage Δ% (MibS vs ILS, solved instances): "
              f"{solved['delta_pct'].mean():.1f}%")
    print(f"MibS reached time limit on {(df['status'] == 'time_limit').sum()} instances.")
    print(f"MibS found no solution on  {(df['status'] == 'no_solution').sum()} instances.")

    print("\nGenerating plots...")
    plot_all(df)
    print("Done.")


if __name__ == "__main__":
    main()
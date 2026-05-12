"""
parse_mibs_results.py
---------------------
Extracts scalability metrics from MibS output files (mibs_<N>_sheet<S>.txt)
and writes them to a CSV file for plotting and analysis.

Paths (relative to repo root, i.e. the directory containing this script):
  input  : results/mibs/mibs_*_sheet*.txt
  output : results/mibs_final_result.csv
"""

import os
import re
import csv
import glob

# ── paths ──────────────────────────────────────────────────────────────────────

REPO_ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR         = os.path.join(REPO_ROOT, "results", "mibs")
DEFAULT_OUTPUT_CSV = os.path.join(REPO_ROOT, "results", "mibs_final_result.csv")

# ── regex patterns for each field ─────────────────────────────────────────────

PATTERNS = {
    # Problem structure
    "ul_vars":            re.compile(r"Number of UL Variables:\s*(\d+)"),
    "ll_vars":            re.compile(r"Number of LL Variables:\s*(\d+)\s*$", re.MULTILINE),
    "ul_rows":            re.compile(r"Number of UL Rows:\s*(\d+)"),
    "ll_rows":            re.compile(r"Number of LL Rows:\s*(\d+)"),
    "int_ul_vars":        re.compile(r"Number of integer UL Variables:\s*(\d+)"),
    "int_ll_vars":        re.compile(r"Number of integer LL Variables:\s*(\d+)"),

    # Search stats
    "nodes_processed":    re.compile(r"Alps0265I Number of nodes fully processed:\s*(\d+)"),
    "nodes_partial":      re.compile(r"Alps0266I Number of nodes partially processed:\s*(\d+)"),
    "nodes_branched":     re.compile(r"Alps0267I Number of nodes branched:\s*(\d+)"),
    "nodes_pruned":       re.compile(r"Alps0268I Number of nodes pruned before processing:\s*(\d+)"),
    "tree_depth":         re.compile(r"Alps0272I Tree depth:\s*(\d+)"),

    # Timing
    "cpu_time_s":         re.compile(r"Alps0274I Search CPU time:\s*([\d.]+)\s*seconds"),
    "wall_time_s":        re.compile(r"Alps0278I Search wall-clock time:\s*([\d.]+)\s*seconds"),
    "feasibility_time":   re.compile(r"Blis0040I Checking feasibility took\s*([\d.]+)\s*seconds"),
    "vf_time":            re.compile(r"Time for solving problem \(VF\)\s*=\s*([\d.]+)"),
    "ub_time":            re.compile(r"Time for solving problem \(UB\)\s*=\s*([\d.]+)"),

    # Solution quality
    "best_obj":           re.compile(r"Alps0260I Best solution found had quality\s*([\d.]+)"),
    "opt_gap_pct":        re.compile(r"Blis0058I Relative optimality gap is\s*([\d.]+)%"),

    # Cut generator
    "cut_gen_calls":      re.compile(r"Called MIBS cut generator\s*(\d+)\s*times"),
    "cuts_generated":     re.compile(r"generated\s*(\d+)\s*cuts"),

    # VF / UB solves
    "vf_solves":          re.compile(r"Number of problems \(VF\) solved\s*=\s*(\d+)"),
    "ub_solves":          re.compile(r"Number of problems \(UB\) solved\s*=\s*(\d+)"),

    # Status and branching
    "status":             re.compile(r"(Alps0208I Search completed\.|Alps0209I Search time limit exceeded\.)"),
    "branching_strategy": re.compile(r"Default branching strategy is (\S+)\."),
}

COLUMNS = [
    "instance", "sheet",
    "ul_vars", "ll_vars", "ul_rows", "ll_rows",
    "int_ul_vars", "int_ll_vars",
    "open_hotels",
    "branching_strategy", "status",
    "nodes_processed", "nodes_partial", "nodes_branched", "nodes_pruned",
    "tree_depth",
    "cpu_time_s", "wall_time_s", "feasibility_time",
    "vf_time", "ub_time", "vf_solves", "ub_solves",
    "cut_gen_calls", "cuts_generated",
    "best_obj", "opt_gap_pct",
    "file",
]


def parse_file(path: str) -> dict:
    """Parse a single MibS output file and return a dict of metrics."""
    with open(path) as f:
        text = f.read()

    row = {"file": os.path.basename(path)}

    # Instance index and sheet from filename (mibs_<N>_sheet<S>.txt)
    m = re.search(r"mibs_(\d+)_sheet(\d+)", os.path.basename(path))
    if m:
        row["instance"] = int(m.group(1))
        row["sheet"]    = int(m.group(2))

    for field, pattern in PATTERNS.items():
        m = pattern.search(text)
        if m:
            val = m.group(1)
            if field == "status":
                row[field] = "completed" if "0208" in val else "time_limit"
            elif field == "branching_strategy":
                row[field] = val.replace("MibSBranchingStrategy", "")
            else:
                try:
                    row[field] = float(val) if "." in val else int(val)
                except ValueError:
                    row[field] = val
        else:
            row[field] = None

    # ll_vars appears twice in the file (header + structure section); use the
    # second occurrence, which is inside "Analyzing problem structure"
    ll_matches = re.findall(r"Number of LL Variables:\s*(\d+)", text)
    if len(ll_matches) >= 2:
        row["ll_vars"] = int(ll_matches[1])
    elif ll_matches:
        row["ll_vars"] = int(ll_matches[0])

    # Count open hotels: lines starting with "x[" in the optimal solution section
    row["open_hotels"] = len(re.findall(r"^x\[", text, re.MULTILINE))

    return row


def parse_mibs_results(input_dir: str = INPUT_DIR, output_csv: str = DEFAULT_OUTPUT_CSV) -> list[dict]:
    """Parse all MibS output files, write a CSV, and return the list of result rows."""
    pattern = os.path.join(input_dir, "mibs_*_sheet*.txt")
    files   = sorted(glob.glob(pattern))

    if not files:
        print(f"No files found in: {input_dir}")
        return []

    rows = [parse_file(f) for f in files]
    rows.sort(key=lambda r: (r.get("instance", 0), r.get("sheet", 0)))

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {output_csv}")
    print(f"\n{'inst':>4}  {'UpperLevelVar':>5}  {'LowerLevelVar':>6}  {'hotels':>6}  {'nodes':>7}  {'cpu(s)':>8}  {'gap%':>6}  status")
    print("-" * 70)
    for r in rows:
        print(
            f"{r.get('instance', '?'):>4}  "
            f"{r.get('ul_vars', '?'):>5}  "
            f"{r.get('ll_vars', '?'):>6}  "
            f"{r.get('open_hotels', '?'):>6}  "
            f"{r.get('nodes_processed', '?'):>7}  "
            f"{r.get('cpu_time_s', '?'):>8.2f}  "
            f"{r.get('opt_gap_pct', '?'):>6}  "
            f"{r.get('status', '?')}"
        )

    return rows


if __name__ == "__main__":
    parse_mibs_results()
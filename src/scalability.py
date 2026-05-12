# scalability.py

import csv
import pandas as pd
import math
from pathlib import Path

_RESULTS_DIR = Path(__file__).parent.parent / "results"

FILE_ILS  = _RESULTS_DIR / "ils_results.csv"
FILE_MIBS = _RESULTS_DIR / "mibs_final_result.csv"

OUTPUT_FILE = _RESULTS_DIR / "scalability.csv"


def truncate_sig(x, sig=4):
    if pd.isna(x):
        return x
    x = float(x)
    if x == 0:
        return 0
    factor = 10 ** (sig - 1 - int(math.floor(math.log10(abs(x)))))
    return math.trunc(x * factor) / factor


def compute_outcomes():
    # 1) Load CSVs
    ils_raw  = pd.read_csv(FILE_ILS)
    mibs_raw = pd.read_csv(FILE_MIBS)

    # 2) Add instance column (integer)
    ils_raw["instance"] = ils_raw["file"].astype(int)
    mibs_raw["instance"] = mibs_raw["instance"].astype(int)

    ils = ils_raw[ils_raw["sheet"] == 0].copy()

    # 3) Parse hotels_selected ("11/12")
    ils[["ils_hotels", "tot_hotels"]] = ils["hotels_selected"].str.split("/", expand=True)
    ils["ils_hotels"] = ils["ils_hotels"].astype(int)
    ils["tot_hotels"] = ils["tot_hotels"].astype(int)

    # 4) ILS columns
    ils = ils.rename(columns={
        "objective": "ils_obj",
        "time_sec": "ils_cpu_time",
        "assignment_cost": "ils_assignment_cost",
        "misplacement_cost": "ils_misplacement_cost",
        "contract_cost": "ils_contract_cost"
    })

    ils = ils[[
        "instance",
        "ils_obj",
        "ils_cpu_time",
        "ils_assignment_cost",
        "ils_misplacement_cost",
        "ils_contract_cost",
        "ils_hotels",
        "tot_hotels"
    ]]

    # 5) MiBs columns
    mibs = mibs_raw[[
        "instance",
        "cpu_time_s",
        "open_hotels",
        "best_obj"
    ]].copy()

    mibs = mibs.rename(columns={"cpu_time_s": "mibs_cpu_time"})

    # 6) Merge on instance
    data = ils.merge(mibs, on="instance")
    data["instance"] = data["instance"].astype(int)

    # 7) Compute derived columns
    lb = data["best_obj"]

    data["ils_gap_from_lb"] = abs(data["ils_obj"] - lb)
    data["delta_pct"]       = (lb - data["ils_obj"]) / lb * 100
    data["cpu_ratio"]       = data["ils_cpu_time"] / data["mibs_cpu_time"]

    for col in data.columns:
        if col not in ["instance", "ils_hotels", "tot_hotels", "open_hotels"]:
            data[col] = data[col].apply(truncate_sig)

    # 8) Write output CSV
    with open(OUTPUT_FILE, "wt", newline="") as f:
        writer = csv.writer(f)
        header = list(data.columns)
        writer.writerow(header)

        for _, row in data.iterrows():
            writer.writerow([row[col] for col in header])

    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    compute_outcomes()
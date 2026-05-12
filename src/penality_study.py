# penality_study.py

import argparse
import random
import copy
import os
import sys
import time
import csv
from datetime import datetime
from zoneinfo import ZoneInfo

import gurobipy as gp
from gurobipy import GRB
from data_loader import get_data_from_file_excel, validate_dimensions
from utils import generate_plot_penality, format_time
from ils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--time-limit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop after 2 hours and save partial results (default: off)."
    )

    args = parser.parse_args()

    start_time = time.time()
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    results_file = os.path.join(results_dir, "penality_study.csv")

    instance_time_limit = 2 * 3600 if args.time_limit else None

    if os.path.exists(results_file):
        os.remove(results_file)
    for sheet_idx in range(0, 11):
        run_ils_instance("penality_study", sheet_idx, results_file, time_limit=instance_time_limit)

    if os.path.isfile(results_file):
        generate_plot_penality(results_file)

    elapsed = format_time(time.time() - start_time)
    print(f"Total time needed: {elapsed}")

    if sys.platform == "darwin":
        os.system(f'osascript -e \'display notification "Completed in {elapsed}" with title "Penality Study"\'')
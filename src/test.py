import argparse
import random
import copy
import os
import time
import csv
from datetime import datetime
from zoneinfo import ZoneInfo

import gurobipy as gp
from gurobipy import GRB
from data_loader import get_data_from_file_excel, validate_dimensions
from utils import generate_plots_ils, format_time
from ils import *
from run_mibs import *


if __name__ == "__main__":
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_results")
    results_ils_file = os.path.join(results_dir, "ils_results.csv")
    if os.path.exists(results_ils_file):
        os.remove(results_ils_file)

    # if you want, you can set a time limit in seconds and 
    # you need to add it as last parameter in the function call
    # instance_time_limit = 2 * 3600

    run_ils_instance(1, 0, results_ils_file)
    run_ils_instance(2, 0, results_ils_file)

    print(f"ILS concluded")


    ### MIBS
    results_mibs_file = os.path.join(results_dir, "mibs_results.csv")
    single_mibs_dir = os.path.join(results_dir, "single_mibs")
    run_mibs_instance(1, 0, results_dir=single_mibs_dir, output_csv=results_mibs_file)
    run_mibs_instance(2, 0, results_dir=single_mibs_dir, output_csv=results_mibs_file)

    print(f"MIBS concluded")




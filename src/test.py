import os
from ils import *
from run_mibs import *


if __name__ == "__main__":
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_results")
    os.makedirs(results_dir, exist_ok=True)

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
    if os.path.exists(results_mibs_file):
        os.remove(results_mibs_file)

    single_mibs_dir = os.path.join(results_dir, "single_mibs")
    os.makedirs(single_mibs_dir, exist_ok=True)
    for f in os.listdir(single_mibs_dir):
        if f.endswith(".txt") or f.endswith(".csv"):
            os.remove(os.path.join(single_mibs_dir, f))

    run_mibs_instance(1, 0, results_dir=single_mibs_dir, output_csv=results_mibs_file)
    run_mibs_instance(2, 0, results_dir=single_mibs_dir, output_csv=results_mibs_file)

    print(f"MIBS concluded")




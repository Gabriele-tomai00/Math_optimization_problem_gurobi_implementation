from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def generate_plots_flca(csv_file="flca_results.csv",
                        scalability_time_img="flca_scalability_time.png",
                        scalability_objective_img="flca_scalability_objective.png",
                        cost_breakdown_img="flca_cost_breakdown.png"):
    csv_file = RESULTS_DIR / Path(csv_file).name
    scalability_time_img = RESULTS_DIR / Path(scalability_time_img).name
    scalability_objective_img = RESULTS_DIR / Path(scalability_objective_img).name
    cost_breakdown_img = RESULTS_DIR / Path(cost_breakdown_img).name
    df = pd.read_csv(csv_file)

    # Plot 1: solve time
    plt.figure()
    plt.scatter(df["file"], df["time_sec"])
    plt.xlabel("File")
    plt.ylabel("Solve time (seconds)")
    plt.title("FLCA Scalability: Solve Time")
    plt.grid(True)
    plt.savefig(scalability_time_img)

    # Plot 2: objective
    plt.figure()
    plt.scatter(df["file"], df["objective"], color="green")
    plt.xlabel("File")
    plt.ylabel("Objective value")
    plt.title("FLCA Scalability: Objective Value")
    plt.grid(True)
    plt.savefig(scalability_objective_img)

    # Plot 3: cost breakdown
    plt.figure()
    plt.scatter(df["file"], df["assignment_cost"], label="Assignment cost")
    plt.scatter(df["file"], df["misplacement_cost"], label="Misplacement cost")
    plt.scatter(df["file"], df["contract_cost"], label="Contract cost")
    plt.xlabel("File")
    plt.ylabel("Cost")
    plt.title("FLCA Cost Breakdown")
    plt.legend()
    plt.grid(True)
    plt.savefig(cost_breakdown_img)

    print(f"\nAll instances processed. Results saved to {csv_file}")
    print(f"Plots saved in {scalability_time_img}, {scalability_objective_img}, {cost_breakdown_img}")

def generate_plots_ils(csv_file="ils_results.csv",
                       scalability_time_img="ils_scalability_time.png",
                       scalability_objective_img="ils_scalability_objective.png",
                       cost_breakdown_img="ils_cost_breakdown.png"):
    csv_file = RESULTS_DIR / Path(csv_file).name
    scalability_time_img = RESULTS_DIR / Path(scalability_time_img).name
    scalability_objective_img = RESULTS_DIR / Path(scalability_objective_img).name
    cost_breakdown_img = RESULTS_DIR / Path(cost_breakdown_img).name
    df = pd.read_csv(csv_file)

    # Plot 1: solve time
    plt.figure()
    plt.scatter(df["file"], df["time_sec"])
    plt.xlabel("File")
    plt.ylabel("Solve time (seconds)")
    plt.title("ILS Scalability: Solve Time")
    plt.grid(True)
    plt.savefig(scalability_time_img)

    # Plot 2: objective
    plt.figure()
    plt.scatter(df["file"], df["objective"], color="green")
    plt.xlabel("File")
    plt.ylabel("Objective value")
    plt.title("ILS Scalability: Objective Value")
    plt.grid(True)
    plt.savefig(scalability_objective_img)

    # Plot 3: cost breakdown
    plt.figure()
    plt.scatter(df["file"], df["assignment_cost"], label="Assignment cost")
    plt.scatter(df["file"], df["misplacement_cost"], label="Misplacement cost")
    plt.scatter(df["file"], df["contract_cost"], label="Contract cost")
    plt.xlabel("File")
    plt.ylabel("Cost")
    plt.title("ILS Cost Breakdown")
    plt.legend()
    plt.grid(True)
    plt.savefig(cost_breakdown_img)

    print(f"\nAll instances processed. Results saved to {csv_file}")
    print(f"Plot about scalability and cost breakdown saved in {scalability_time_img}, {scalability_objective_img}, {cost_breakdown_img}")


def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = round(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def generate_plot_penality(csv_path=None, out_path=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "..", "results")

    if csv_path is None:
        csv_path = os.path.join(results_dir, "penality_study.csv")
    if out_path is None:
        out_path = os.path.join(results_dir, "penality_study.png")

    df = pd.read_csv(csv_path)
    df = df.groupby("penalty")[["assignment_cost", "contract_cost", "misplacement_cost"]].mean().reset_index()

    penalties = df["penalty"].values
    assignment_cost = df["assignment_cost"].values
    contract_cost = df["contract_cost"].values
    misplacement_cost = df["misplacement_cost"].values

    x = np.arange(len(penalties))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_contract = ax.bar(x - width / 2, contract_cost, width, label="Contracting Cost", color="#5C3317")
    bars_assignment = ax.bar(x + width / 2, assignment_cost, width, label="Assignment Cost", color="#87CEEB")

    total_cost = assignment_cost + contract_cost
    line_total = ax.plot(x, total_cost, color="black", marker="s", linestyle="--", linewidth=1.5, markersize=5, label="Contracting Cost + Assignment Cost")[0]
    line_misplacement = ax.plot(x, misplacement_cost, color="#555555", marker="x", linestyle="-", linewidth=1.5, markersize=7, markeredgewidth=2, label="Misplaced Demand")[0]

    ax.set_xlabel("Penalty")
    ax.set_ylabel("Cost")
    ax.set_title("Assignment Cost vs Contract Cost by Penalty")
    ax.set_xticks(x)
    ax.set_xticklabels(penalties)
    ax.legend(handles=[bars_contract, bars_assignment, line_misplacement, line_total])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{int(val):,}"))
    fig.tight_layout()

    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {out_path}")
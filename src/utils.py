from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

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
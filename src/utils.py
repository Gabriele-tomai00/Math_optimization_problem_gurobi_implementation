import matplotlib.pyplot as plt
import pandas as pd

def generate_plots_flca(csv_file = "flca_results.csv", scalability_time_img="flca_scalability_time.png",
                        scalability_objective_img="flca_scalability_objective.png",
                        cost_breakdown_img="flca_cost_breakdown.png"):
    df = pd.read_csv(csv_file)

    # Plot 1: solve time
    plt.figure()
    plt.plot(df["file"], df["time_sec"], marker="o")
    plt.xlabel("File")
    plt.ylabel("Solve time (seconds)")
    plt.title("FLCA Scalability: Solve Time")
    plt.grid(True)
    plt.savefig(scalability_time_img)

    # Plot 2: objective
    plt.figure()
    plt.plot(df["file"], df["objective"], marker="o", color="green")
    plt.xlabel("File")
    plt.ylabel("Objective value")
    plt.title("FLCA Scalability: Objective Value")
    plt.grid(True)
    plt.savefig(scalability_objective_img)

    # Plot 3: cost breakdown
    plt.figure()
    plt.plot(df["file"], df["assignment_cost"], label="Assignment cost")
    plt.plot(df["file"], df["misplacement_cost"], label="Misplacement cost")
    plt.plot(df["file"], df["contract_cost"], label="Contract cost")
    plt.xlabel("File")
    plt.ylabel("Cost")
    plt.title("FLCA Cost Breakdown")
    plt.legend()
    plt.grid(True)
    plt.savefig(cost_breakdown_img)

    print("\nAll instances processed. Results saved to flca_results.csv")
    print(f"Plot about scalability and cost breakdown saved in flca_scalability_time.png, flca_scalability_objective.png, flca_cost_breakdown.png")

def generate_plots_ils(csv_file = "ils_results.csv", scalability_time_img="ils_scalability_time.png", scalability_objective_img="ils_scalability_objective.png", cost_breakdown_img="ils_cost_breakdown.png"):
    df = pd.read_csv(csv_file)

    # Plot 1: solve time
    plt.figure()
    plt.plot(df["file"], df["time_sec"], marker="o")
    plt.xlabel("File")
    plt.ylabel("Solve time (seconds)")
    plt.title("ILS Scalability: Solve Time")
    plt.grid(True)
    plt.savefig(scalability_time_img)

    # Plot 2: objective
    plt.figure()
    plt.plot(df["file"], df["objective"], marker="o", color="green")
    plt.xlabel("File")
    plt.ylabel("Objective value")
    plt.title("ILS Scalability: Objective Value")
    plt.grid(True)
    plt.savefig(scalability_objective_img)

    # Plot 3: cost breakdown
    plt.figure()
    plt.plot(df["file"], df["assignment_cost"], label="Assignment cost")
    plt.plot(df["file"], df["misplacement_cost"], label="Misplacement cost")
    plt.plot(df["file"], df["contract_cost"], label="Contract cost")
    plt.xlabel("File")
    plt.ylabel("Cost")
    plt.title("ILS Cost Breakdown")
    plt.legend()
    plt.grid(True)
    plt.savefig(cost_breakdown_img)

    print(f"\nAll instances processed. Results saved to {csv_file}")
    print(f"Plot about scalability and cost breakdown saved in {scalability_time_img}, {scalability_objective_img}, {cost_breakdown_img}")

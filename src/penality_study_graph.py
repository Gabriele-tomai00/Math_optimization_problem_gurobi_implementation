import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, "..", "results")

csv_path = os.path.join(results_dir, "penality_study.csv")
df = pd.read_csv(csv_path)

penalties = df["penalty"].values
assignment_cost = df["assignment_cost"].values
contract_cost = df["contract_cost"].values

x = np.arange(len(penalties))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars_assignment = ax.bar(x - width / 2, assignment_cost, width, label="Assignment Cost", color="#87CEEB")
bars_contract = ax.bar(x + width / 2, contract_cost, width, label="Contract Cost", color="#5C3317")

total_cost = assignment_cost + contract_cost
ax.plot(x, total_cost, color="black", marker="s", linestyle="--", linewidth=1.5, markersize=5, label="Total Cost (Assignment + Contract)")

misplacement_cost = df["misplacement_cost"].values
ax.plot(x, misplacement_cost, color="#555555", marker="x", linestyle="-", linewidth=1.5, markersize=7, markeredgewidth=2, label="Misplacement Cost")

ax.set_xlabel("Penalty")
ax.set_ylabel("Cost")
ax.set_title("Assignment Cost vs Contract Cost by Penalty")
ax.set_xticks(x)
ax.set_xticklabels(penalties)
ax.legend()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{int(val):,}"))
fig.tight_layout()

out_path = os.path.join(results_dir, "penality_study.png")
plt.savefig(out_path, dpi=150)
print(f"Plot saved to {out_path}")

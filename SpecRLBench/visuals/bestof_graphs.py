import numpy as np
import matplotlib.pyplot as plt


# Rescue rates (%)
# Rows: tasks
# Columns: algorithm types
data = {
    "PointLTL4MASAR1": {
        "Vanilla PPO": 96.0,
        "PPO Lagrangian": np.nan,
        "Vanilla TRPO": 94.0,
        "TRPO Lagrangian": np.nan,
    },
    "PointLTL4MASAR1WC": {
        "Vanilla PPO": 86.0,
        "PPO Lagrangian": 80.0,
        "Vanilla TRPO": 82.0,
        "TRPO Lagrangian": 88.0,
    },
    "PointLTL5MASAR1": {
        "Vanilla PPO": 94.0,
        "PPO Lagrangian": np.nan,
        "Vanilla TRPO": 90.0,
        "TRPO Lagrangian": np.nan,
    },
    "PointLTL5MASAR1WC": {
        "Vanilla PPO": 68.0,
        "PPO Lagrangian": 72.0,
        "Vanilla TRPO": 66.0,
        "TRPO Lagrangian": 72.0,
    },
    "PointLTL6MASAR1": {
        "Vanilla PPO": 90.0,
        "PPO Lagrangian": np.nan,
        "Vanilla TRPO": 84.0,
        "TRPO Lagrangian": np.nan,
    },
    "PointLTL6MASAR1WC": {
            "Vanilla PPO": 42.0,
            "PPO Lagrangian": np.nan,
            "Vanilla TRPO": np.nan,
            "TRPO Lagrangian": np.nan,
        },
}


algorithms = [
    "Vanilla PPO",
    "PPO Lagrangian",
    "Vanilla TRPO",
    "TRPO Lagrangian",
]

tasks = list(data.keys())

# Bar positioning
x = np.arange(len(tasks))
n_algorithms = len(algorithms)
bar_width = 0.18

fig, ax = plt.subplots(figsize=(12, 6))

for i, algorithm in enumerate(algorithms):
    values = [
        data[task][algorithm]
        for task in tasks
    ]

    # Center the group of bars around each task's x position
    offset = (i - (n_algorithms - 1) / 2) * bar_width

    bars = ax.bar(
        x + offset,
        values,
        width=bar_width,
        label=algorithm,
    )

    # Add rescue-rate labels above bars
    for bar, value in zip(bars, values):
        if not np.isnan(value):
            ax.annotate(
                f"{value:.0f}%",
                xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )


# Axis labels and title
ax.set_xlabel("Task")
ax.set_ylabel("Rescue Rate (%)")
ax.set_title("Rescue Rate by Task and Algorithm")

# Task names
ax.set_xticks(x)
ax.set_xticklabels(tasks, rotation=20, ha="right")

# Rescue rate is a percentage
ax.set_ylim(0, 105)

# Legend
ax.legend()

# Grid for readability
ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.4,
)

plt.tight_layout()
# plt.show()
# Save the figure to a file
plt.savefig(
    "bestof_rescue_rates.png",
    dpi=300,
    bbox_inches="tight",
)

print("Graph saved to: bestof_rescue_rates.png")
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# Data
# ============================================================

levels = ["Level 0", "Level 1"]
algorithms = ["PPO", "PPO-Lagrangian", "GenZ-LTL"]

data = {
    "success": {
        "Level 0": [
            [0.05, 0.00, 0.02, 0.03, 0.25],
            [0.14, 0.28, 0.14, 0.33, 0.47],
            [0.73, 0.85, 0.78, 0.75, 0.78]
        ],
        "Level 1": [
            [0.01, 0.01, 0.00, 0.00, 0.00],
            [0.00, 0.01, 0.03, 0.00, 0.00],
            [0.09, 0.20, 0.05, 0.14, 0.02]
        ]
    },

    "violation": {
        "Level 0": [
            [0.16, 0.64, 0.71, 0.10, 0.09],
            [0.19, 0.13, 0.07, 0.21, 0.22],
            [0.16, 0.05, 0.04, 0.16, 0.02]
        ],
        "Level 1": [
            [0.69, 0.74, 0.17, 0.19, 0.14],
            [0.57, 0.60, 0.57, 0.87, 0.51],
            [0.13, 0.04, 0.30, 0.11, 0.06]
        ]
    },

    "episode_length": {
        "Level 0": [
            [1028.8, 1426.0, 1107.3, 1069.2],
            [1005.4, 941.5, 810.4, 879.1, 892.5],
            [776.0, 766.718, 890.731, 765.8, 826.974]
        ],
        "Level 1": [
            [637.0, 625.0],
            [2259.0, 1169.7],
            [600.778, 826.75, 708.6, 834.571, 725.0]
        ]
    }
}


# ============================================================
# Plot function
# ============================================================

def make_plot(metric, ylabel, title, filename):

    x = np.arange(len(levels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    for i, algo in enumerate(algorithms):

        means = []
        stds = []

        for level in levels:

            values = data[metric][level][i]

            if len(values) > 0:
                means.append(np.mean(values))
                stds.append(np.std(values))
            else:
                means.append(np.nan)
                stds.append(0)

        ax.bar(
            x + (i-1)*width,
            means,
            width,
            yerr=stds,
            capsize=5,
            label=algo
        )


    ax.set_xticks(x)
    ax.set_xticklabels(levels, fontsize=12)

    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14)

    ax.legend(
        fontsize=10,
        frameon=False
    )

    ax.tick_params(axis="both", labelsize=11)

    save_dir = os.path.expanduser("~/RISE-2026/RISE-Training/visuals")

    os.makedirs(save_dir, exist_ok=True)

    path = os.path.join(save_dir, filename)

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()



# ============================================================
# Generate images
# ============================================================

make_plot(
    "success",
    "Success Rate ηₛ",
    "Success Rate Comparison",
    "ltl_success_rate.png"
)

make_plot(
    "violation",
    "Violation Rate ηᵥ",
    "Violation Rate Comparison",
    "ltl_violation_rate.png"
)

make_plot(
    "episode_length",
    "Episode Length μ",
    "Episode Length Comparison",
    "ltl_episode_length.png"
)
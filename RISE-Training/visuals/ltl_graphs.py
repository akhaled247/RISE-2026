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
            [0.01,0.00,0.08,0.20,0.06],
            [0.00,0.23,0.05,0.22,0.24],
            [0.44,0.46,0.45,0.44,0.45]
        ],
        "Level 1": [
            [0.01,0.00,0.00,0.00,0.00],
            [0.00,0.00,0.00,0.00,0.00],
            [0.07,0.1,0.08]
        ]
    },

    "violation": {
        "Level 0": [
            [0.10,0.13,0.18,0.15,0.02],
            [0.03,0.19,0.19,0.34,0.14],
            [0.22,0.18,0.10,0.19,0.08]
        ],
        "Level 1": [
            [0.03,0.01,0.111,0.05,0.28],
            [0.25,0.13,0.17,0.07,0.24],
            [0.12,0.07,0.13]
        ]
    },

    "episode_length": {
        "Level 0": [
            [1586,766,1779,1415,963],
            [662,837,1538,820,796],
            [774,712,922,737,837]
        ],
        "Level 1": [
            [721,467,1869,2319,2275],
            [922,881,576,787,1114],
            [567,797,694]
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
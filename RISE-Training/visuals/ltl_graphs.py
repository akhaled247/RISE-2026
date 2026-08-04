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
            [0.050, 0.000, 0.020, 0.030, 0.250],
            [0.140, 0.280, 0.140, 0.330, 0.470],
            [0.730, 0.850, 0.780, 0.750, 0.780]
        ],

        "Level 1": [
            [0.010, 0.010, 0.000, 0.000, 0.000],
            [0.000, 0.010, 0.030, 0.000, 0.000],
            [0.020, 0.050, 0.100, 0.010, 0.030]
        ]
    },


    "violation": {
        "Level 0": [
            [0.160, 0.640, 0.710, 0.100, 0.090],
            [0.190, 0.130, 0.070, 0.210, 0.220],
            [0.160, 0.050, 0.040, 0.160, 0.020]
        ],

        "Level 1": [
            [0.690, 0.740, 0.170, 0.190, 0.140],
            [0.570, 0.600, 0.570, 0.870, 0.510],
            [0.910, 0.910, 0.870, 0.950, 0.900]
        ]
    },


    "episode_length": {
        "Level 0": [
            [1028.8, np.nan, 1426.0, 1107.3, 1069.2],
            [1005.4, 941.5, 810.4, 879.1, 892.5],
            [776.0, 766.718, 890.731, 765.8, 826.974]
        ],

        "Level 1": [
            [637.0, 625.0, np.nan, np.nan, np.nan],
            [np.nan, 2259.0, 1169.7, np.nan, np.nan],
            [542.500, 652.600, 620.100, 716.000, 664.667]
        ]
    }
}

# ============================================================
# Plot function
# ============================================================

def make_plot(metric, ylabel, title, filename):

    x = np.arange(len(levels))
    width = 0.25

    fig, ax = plt.subplots(
        figsize=(8, 5),
        dpi=300
    )

    for i, algo in enumerate(algorithms):

        means = []
        stds = []

        for level in levels:

            values = np.array(
                data[metric][level][i],
                dtype=float
            )

            # Match R's mean(x, na.rm=TRUE) and sd(x, na.rm=TRUE)
            means.append(
                np.nanmean(values)
            )

            stds.append(
                np.nanstd(values, ddof=1)
            )


        ax.bar(
            x + (i-1)*width,
            means,
            width,
            yerr=stds,
            capsize=5,
            label=algo
        )


    ax.set_xticks(x)
    ax.set_xticklabels(
        levels,
        fontsize=12
    )

    ax.set_ylabel(
        ylabel,
        fontsize=13
    )

    ax.set_title(
        title,
        fontsize=14
    )

    ax.legend(
        fontsize=10,
        frameon=False
    )

    ax.tick_params(
        axis="both",
        labelsize=11
    )


    save_dir = os.path.expanduser(
        "~/RISE-2026/RISE-Training/visuals"
    )

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    path = os.path.join(
        save_dir,
        filename
    )

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
import time

import jax
import numpy as np
import matplotlib.pyplot as plt

from zone_jax import SafeMultiAgentMJXEnv
from zone_jax import COLOR_NAMES_BY_ID

config = {
    "agent_num": 2,
    "agent_name": "Point",
    "zone_colors": ["green", "yellow", "blue", "magenta"],
    "zone_size": [0.4, 0.4, 0.4, 0.4],
    "zone_num": [2, 2, 2, 2],
    "zone_keepout": 0.4,
    "agent_radius": 0.28,
    "placement_extents": [-2.5, -2.5, 2.5, 2.5],
    "max_steps": 1000,
    "frameskip": 10,
}


def labels_to_names(env, labels):
    out = {}
    for agent, v in labels.items():
        idx = int(jax.device_get(v))
        out[agent] = "none" if idx < 0 else env.lidar_zone_colors[idx]
    return out


def sample_actions(env, key):
    keys = jax.random.split(key, len(env.agents))
    actions = {"agent_0": jax.numpy.ones(env.action_spaces["agent_0"].shape),
               "agent_1": jax.numpy.zeros(env.action_spaces["agent_1"].shape)}
    return actions
    # return {
    #     # actions is all ones, which means "go forward" in our action space
    #     agent: jax.numpy.ones(env.action_spaces[agent].shape)
    #     for agent in env.agents
    # }
    # return {
    #     agent: jax.random.uniform(
    #         keys[i],
    #         shape=env.action_spaces[agent].shape,
    #         minval=-1.0,
    #         maxval=1.0,
    #     )
    #     for i, agent in enumerate(env.agents)
    # }


def draw_state(ax, env, state, obs, labels, ep, step, 
               lidar_color="blue", pause_sec=0.01):
    ax.clear()

    extents = np.asarray(env.spec.placement_extents, dtype=float)
    xpos = np.asarray(jax.device_get(state.data.xpos))
    xmat = np.asarray(jax.device_get(state.data.xmat)).reshape((-1, 3, 3))

    zone_xy = xpos[np.asarray(env.zone_body_ids), :2]
    agent_xy = xpos[np.asarray(env.agent_body_ids), :2]
    agent_theta = np.arctan2(
        xmat[np.asarray(env.agent_body_ids), 1, 0],
        xmat[np.asarray(env.agent_body_ids), 0, 0],
    )

    zone_size = np.asarray(jax.device_get(env.zone_sizes))
    zone_color_id = np.asarray(jax.device_get(env.zone_color_ids))

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(extents[0], extents[2])
    ax.set_ylim(extents[1], extents[3])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    ax.set_title(f"Episode {ep}, step {step}, lidar={lidar_color}")

    # zones
    for xy, size, color_id in zip(zone_xy, zone_size, zone_color_id):
        color = COLOR_NAMES_BY_ID[int(color_id)]
        ax.add_patch(
            plt.Circle(
                xy,
                float(size),
                color=color,
                alpha=0.18,
                ec=color,
                lw=1.5,
            )
        )
        ax.text(xy[0], xy[1], color, ha="center", va="center", fontsize=8)

    # selected lidar group
    if lidar_color in env.lidar_zone_colors:
        group_idx = env.lidar_zone_colors.index(lidar_color)

        bin_angles = (
            np.arange(env.lidar_bin_dim) + 0.5
        ) * (2.0 * np.pi / env.lidar_bin_dim)

        max_ray_length = 1.0

        for agent_idx, xy in enumerate(agent_xy):
            agent_obs = np.asarray(jax.device_get(obs[f"agent_{agent_idx}"]))

            start = env.sensor_obs_dim + group_idx * env.lidar_bin_dim
            end = env.sensor_obs_dim + (group_idx + 1) * env.lidar_bin_dim
            lidar = agent_obs[start:end]

            world_angles = agent_theta[agent_idx] + bin_angles
            ray_dirs = np.stack(
                [np.cos(world_angles), np.sin(world_angles)],
                axis=-1,
            )
            ray_ends = xy[None, :] + ray_dirs * lidar[:, None] * max_ray_length

            for value, ray_end in zip(lidar, ray_ends):
                if value <= 0:
                    continue

                ax.plot(
                    [xy[0], ray_end[0]],
                    [xy[1], ray_end[1]],
                    color="tab:red",
                    alpha=0.2 + 0.8 * float(value),
                    lw=1.2,
                    zorder=2,
                )
    else:
        print(
            f"[warning] lidar_color={lidar_color} not in "
            f"{env.lidar_zone_colors}; skip lidar rays."
        )

    # agents
    readable = labels_to_names(env, labels)

    for agent_idx, xy in enumerate(agent_xy):
        agent = f"agent_{agent_idx}"

        ax.add_patch(
            plt.Circle(
                xy,
                float(env.agent_radius),
                color="black",
                alpha=0.35,
                ec="black",
                lw=1.5,
                zorder=3,
            )
        )

        theta = agent_theta[agent_idx]
        arrow_len = 0.35
        ax.arrow(
            xy[0],
            xy[1],
            arrow_len * np.cos(theta),
            arrow_len * np.sin(theta),
            head_width=0.08,
            head_length=0.10,
            fc="black",
            ec="black",
            zorder=4,
        )

        ax.text(
            xy[0],
            xy[1] + 0.10,
            f"a{agent_idx}\n{readable[agent]}",
            ha="center",
            va="bottom",
            fontsize=9,
            zorder=5,
        )

    plt.pause(pause_sec)

def main():
    env = SafeMultiAgentMJXEnv(config)

    num_episodes = 1
    num_steps = 1000
    pause_sec = 0.1
    lidar_color = "yellow"

    rng = jax.random.PRNGKey(0)

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))

    for ep in range(num_episodes):
        rng, reset_key = jax.random.split(rng)
        obs, state = env.reset(reset_key)

        labels = env.label_f(state)
        print(f"\nEpisode {ep} reset propositions:", labels_to_names(env, labels))
        # draw_state(ax, env, state, labels, ep, 0)
        draw_state(ax, env, state, obs, labels, ep, 0, lidar_color=lidar_color, pause_sec=pause_sec)

        for step in range(1, num_steps + 1):
            rng, action_key, step_key = jax.random.split(rng, 3)
            actions = sample_actions(env, action_key)

            obs, state, rewards, dones, info = env.step_env(
                step_key,
                state,
                actions,
            )

            labels = env.label_f(state)
            readable = labels_to_names(env, labels)

            # obs_np = {k: np.asarray(jax.device_get(v)) for k, v in obs.items()}
            # print(f"Episode {ep}, step {step}, obs = {obs_np}, propositions:", readable)

            # draw_state(ax, env, state, labels, ep, step)
            draw_state(ax, env, state, obs, labels, ep, step, lidar_color=lidar_color, pause_sec=pause_sec)
            time.sleep(pause_sec)

            if bool(jax.device_get(dones["__all__"])):
                print(f"Episode {ep} terminated at step {step}")
                break

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
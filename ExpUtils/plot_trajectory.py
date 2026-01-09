import argparse
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import ast
import re


# -----------------------------
# Trajectory parsing utilities
# -----------------------------

def parse_trajectory(traj_str):
    """
    Convert trajectory string into a Python object.
    Handles np.float32(...) and array([...], dtype=float32).
    """
    cleaned = traj_str
    cleaned = re.sub(r'np\.float32\(([^)]+)\)', r'\1', cleaned)
    cleaned = re.sub(r'array\((\[[^\]]*\])[^)]*\)', r'\1', cleaned)
    return ast.literal_eval(cleaned)


# -----------------------------
# Environment plotting
# -----------------------------

def plot_env_config(ax, yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    env_height, env_width = config['Environment']['dimensions']
    pois = config['Environment']['pois']
    agents = config['Agents']

    ax.set_xlim(0, env_width)
    ax.set_ylim(0, env_height)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Environment + Joint Trajectory")
    ax.set_xlabel("Width")
    ax.set_ylabel("Height")

    ax.set_xticks(range(env_width + 1))
    ax.set_yticks(range(env_height + 1))
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # POIs
    for i, poi in enumerate(pois):
        x, y = poi["location"]
        radius = poi["radius"]

        ax.add_patch(
            plt.Circle((y, x), radius, color="blue", alpha=0.3,
                       label="POI" if i == 0 else "")
        )
        ax.plot(y, x, 'bo')

    # Agent starting positions
    for i, (loc, obs_radius) in enumerate(
        zip(agents['starting_locs'], agents['observation_radii'])
    ):
        x, y = loc

        ax.add_patch(
            plt.Circle((y, x), obs_radius, color="green", alpha=0.2,
                       label="Agent Obs Radius" if i == 0 else "")
        )
        ax.plot(y, x, 'go', label="Agent Start" if i == 0 else "")


# -----------------------------
# Trajectory overlay
# -----------------------------

def overlay_joint_trajectory(ax, trajectories):
    for agent_idx, agent_traj in enumerate(trajectories):
        xs = [step["position"][1] for step in agent_traj]
        ys = [step["position"][0] for step in agent_traj]

        ax.plot(xs, ys, linewidth=2, label=f"Agent {agent_idx}")
        ax.scatter(xs[0], ys[0], marker='s')   # start
        ax.scatter(xs[-1], ys[-1], marker='X') # end


# -----------------------------
# Main plotting function
# -----------------------------

def plot_env_and_rollout(env_yaml, csv_path, gen, indiv_id):
    df = pd.read_csv(csv_path)

    row = df[(df["gen"] == gen) & (df["id"] == indiv_id)]
    if row.empty:
        raise ValueError(f"No rollout found for gen={gen}, id={indiv_id}")

    trajectories = parse_trajectory(row.iloc[0]["trajectory"])

    fig, ax = plt.subplots(figsize=(12, 7))

    plot_env_config(ax, env_yaml)
    overlay_joint_trajectory(ax, trajectories)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    plt.show()


# -----------------------------
# CLI
# -----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Overlay NSGA-II rollout trajectories on a rover environment"
    )

    parser.add_argument(
        "--env",
        default="/home/santjami/repos/NoveltyMOMA/config/trap/2ag_trap_30kgens_MORoverEnvConfig.yaml",
        help="Path to environment YAML (default: 2ag_trap_30kgens_MORoverEnvConfig.yaml)"
    )
    parser.add_argument(
        "--csv",
        default="/home/santjami/repos/NoveltyMOMA/data/nsga2_rover_2024_2ag_30kgen_0p5h_2026-01-08_00-57-37_savedata.csv",
        help="Path to CSV results file (default: nsga2_rover_2024_2ag_30kgen_0p5h_2026-01-08_00-57-37_savedata.csv)"
    )
    parser.add_argument(
        "--gen",
        type=int,
        default=0,
        help="NSGA-II generation (default: 0)"
    )
    parser.add_argument(
        "--id",
        type=int,
        default=0,
        help="Individual ID (default: 0)"
    )

    args = parser.parse_args()

    plot_env_and_rollout(
        env_yaml=args.env,
        csv_path=args.csv,
        gen=args.gen,
        indiv_id=args.id
    )


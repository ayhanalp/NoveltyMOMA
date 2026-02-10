import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Trajectory parsing utilities
# -----------------------------
"""
Lightweight plotting utility for MORover experiment episodes.

Features:
- Render the environment instance from `env_instance.yaml` (POIs and agent starts).
- Load `data/<exp>/savedata.csv`, select a row by `gen` and `id`, and plot all agent
  trajectories for that rollout over the environment.

Usage examples:
  python scripts/plot_episode_trajectories.py --exp test_1 --gen 0 --id 0

This script prefers YAML/CSV files saved in the experiment run directory
(`data/<exp>/env_instance.yaml`, `data/<exp>/savedata.csv`).
"""

import os
import yaml
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_env_instance(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def parse_trajectory_field(field_str):
    """Parse the CSV trajectory field into a Python object.

    Tries ast.literal_eval first (safe). Falls back to eval with limited
    globals if the data contains numpy constructs.
    """
    if field_str is None or (isinstance(field_str, float) and np.isnan(field_str)):
        return None
    try:
        return ast.literal_eval(field_str)
    except Exception:
        # fallback for numpy arrays like array([...], dtype=float32)
        try:
            return eval(field_str, {"np": np, "array": np.array, "float32": np.float32})
        except Exception:
            raise


def plot_environment(ax, env_instance, show_poi_radii=True):
    dims = env_instance.get('dimensions', [100, 100])
    pois = env_instance.get('pois', [])
    agents_start = env_instance.get('agents_start', [])

    ax.set_xlim(0, dims[0])
    ax.set_ylim(0, dims[1])
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)

    # Draw POIs color-coded by objective
    cmap = plt.get_cmap('tab10')
    obj_to_color = {}
    poi_handles = {}
    for poi in pois:
        loc = poi.get('location', [0, 0])
        r = poi.get('radius', 1.0)
        obj = int(poi.get('obj', 0))
        if obj not in obj_to_color:
            obj_to_color[obj] = cmap(obj % 10)
        color = obj_to_color[obj]

        circ = plt.Circle((loc[0], loc[1]), r, alpha=0.25, color=color, ec='k')
        ax.add_patch(circ)
        h, = ax.plot(loc[0], loc[1], 'o', color=color)
        ax.text(loc[0], loc[1], f"obj {obj}", fontsize=8, color=color)

        # record a handle for the legend (one per objective)
        if obj not in poi_handles:
            poi_handles[obj] = h

    # Draw agent start locations
    for i, a in enumerate(agents_start):
        ax.plot(a[0], a[1], 'ks')
        ax.text(a[0], a[1], f"A{i}", fontsize=8, color='k')

    # Add a legend entry for POI objectives
    if poi_handles:
        sorted_objs = sorted(poi_handles.keys())
        handles = [poi_handles[o] for o in sorted_objs]
        labels = [f"POI obj {o}" for o in sorted_objs]
        ax.legend(handles=handles, labels=labels, fontsize=8, loc='upper right')


def extract_positions_from_traj(trajectory):
    """Convert trajectory (list per agent of dicts) into list of x,y sequences per agent."""
    positions_per_agent = []
    for agent_traj in trajectory:
        xs, ys = [], []
        for step in agent_traj:
            pos = step.get('position')
            # Accept numpy arrays or lists
            if hasattr(pos, 'tolist'):
                pos = pos.tolist()
            xs.append(float(pos[0]))
            ys.append(float(pos[1]))
        positions_per_agent.append((xs, ys))
    return positions_per_agent


def plot_episode(csv_path, env_yaml_path, gen, rollout_id, output_dir='episode_plots', show=True):
    df = pd.read_csv(csv_path)

    row = df[(df['gen'] == gen) & (df['id'] == rollout_id)]
    if row.empty:
        raise ValueError(f"No row found for gen={gen}, id={rollout_id} in {csv_path}")
    row = row.iloc[0]

    # Load env instance
    env_instance = None
    if os.path.exists(env_yaml_path):
        env_instance = load_env_instance(env_yaml_path)

    # Parse trajectory
    traj_field = row.get('trajectory')
    trajectory = parse_trajectory_field(traj_field)
    if trajectory is None:
        raise ValueError('Trajectory field is empty for requested row')

    positions = extract_positions_from_traj(trajectory)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"gen{gen}_id{rollout_id}_episode.png")

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot environment if available
    if env_instance is not None:
        plot_environment(ax, env_instance)

    cmap = plt.get_cmap('tab10')
    for agent_idx, (xs, ys) in enumerate(positions):
        color = cmap(agent_idx % 10)
        ax.plot(xs, ys, '-o', linewidth=2, markersize=3, color=color, label=f'agent {agent_idx}')
        ax.scatter(xs[0], ys[0], color=color, s=60, marker='o')
        ax.scatter(xs[-1], ys[-1], color=color, s=60, marker='X')

    ax.set_title(f"Gen {gen} | ID {rollout_id} | {len(positions)} agents")
    ax.set_aspect('equal')
    ax.legend(fontsize=8)

    plt.savefig(save_path, dpi=200, bbox_inches='tight')

    # Also save into the experiment folder (same folder as the CSV) if available
    try:
        exp_dir = os.path.dirname(csv_path)
        if exp_dir and os.path.isdir(exp_dir):
            exp_save_path = os.path.join(exp_dir, os.path.basename(save_path))
            plt.savefig(exp_save_path, dpi=200, bbox_inches='tight')
    except Exception:
        # Non-fatal if saving to experiment folder fails
        pass

    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot agent trajectories for a saved rollout')
    parser.add_argument('--exp', default='test_1', help='Experiment folder under data/')
    parser.add_argument('--gen', type=int, default=0, help='Generation number')
    parser.add_argument('--id', type=int, default=0, help='Rollout id')
    parser.add_argument('--outdir', default='episode_plots', help='Output directory for plots')
    parser.add_argument('--no-show', action='store_true', help='Do not display the plot')

    args = parser.parse_args()

    csv_path = os.path.join('data', args.exp, 'savedata.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find savedata.csv at {csv_path}")

    env_yaml_path = os.path.join('data', args.exp, 'env_instance.yaml')

    save_path = plot_episode(csv_path=csv_path, env_yaml_path=env_yaml_path, gen=args.gen, rollout_id=args.id, output_dir=args.outdir, show=not args.no_show)
    print(f"Saved episode plot to: {save_path}")
# -----------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot all agent trajectories from a single episode"
    )
    
    parser.add_argument(
        "--exp",
        default="test_1",
        help="Experiment name under data/ (e.g., test_1)",
    )

    parser.add_argument(
        "--env",
        default=None,
        help="Path to environment image (default: data/<exp>/env_instance.png)",
    )

    parser.add_argument(
        "--gen",
        type=int,
        default=0,
        help="Generation number (default: 0)",
    )
    parser.add_argument(
        "--id",
        type=int,
        default=0,
        help="Rollout ID (default: 0)",
    )
    parser.add_argument(
        "--outdir",
        default="episode_plots",
        help="Directory to save plots (default: episode_plots)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the plot (useful for headless runs)",
    )

    args = parser.parse_args()
    
    csv_path = os.path.join("data", args.exp, "savedata.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find savedata.csv at {csv_path}"
        )

    env_img_path = (
        args.env
        if args.env is not None
        else os.path.join("data", args.exp, "env_instance.png")
    )
    if not os.path.exists(env_img_path):
        raise FileNotFoundError(
            f"Could not find environment image at {env_img_path}"
        )
        
    save_path = plot_episode(
        csv_path=csv_path,
        env_img_path=env_img_path,
        gen=args.gen,
        rollout_id=args.id,
        output_dir=args.outdir,
        show=not args.no_show,
    )

    print(f"Saved episode plot to: {save_path}")


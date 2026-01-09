import pandas as pd
import ast
import re
import matplotlib.pyplot as plt

def parse_trajectory(traj_str):
    """
    Convert trajectory string into a Python list.
    Handles np.float32(...) and array(...) safely.
    """
    # Remove numpy wrappers so literal_eval works
    cleaned = traj_str

    # Replace np.float32(x) -> float(x)
    cleaned = re.sub(r'np\.float32\(([^)]+)\)', r'\1', cleaned)

    # Replace array([...], dtype=float32) -> [...]
    cleaned = re.sub(r'array\((\[[^\]]*\])[^)]*\)', r'\1', cleaned)

    return ast.literal_eval(cleaned)


def plot_joint_trajectory(csv_path, gen, indiv_id):
    df = pd.read_csv(csv_path)

    row = df[(df["gen"] == gen) & (df["id"] == indiv_id)].iloc[0]
    trajectories = parse_trajectory(row["trajectory"])

    plt.figure(figsize=(6, 6))

    for agent_idx, agent_traj in enumerate(trajectories):
        xs = [step["position"][0] for step in agent_traj]
        ys = [step["position"][1] for step in agent_traj]

        plt.plot(xs, ys, marker='o', linewidth=2, label=f"Agent {agent_idx}")
        plt.scatter(xs[0], ys[0], marker='s')   # start
        plt.scatter(xs[-1], ys[-1], marker='X') # end

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Joint Trajectory | Gen {gen}, ID {indiv_id}")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()


# Example usage
plot_joint_trajectory(
    csv_path="/home/santjami/repos/NoveltyMOMA/data/nsga2_rover_2024_2ag_30kgen_0h_2026-01-07_17-34-50_savedata.csv",
    gen=0,
    indiv_id=9
)

#!/usr/bin/env python3

import pandas as pd
import ast
import glob
import os
import re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


def count_unique_raw_fitness(csv_path):
    df = pd.read_csv(csv_path)

    final_gen = df["gen"].max()
    final_df = df[df["gen"] == final_gen]

    raw_vectors = final_df["raw_fitness"].apply(ast.literal_eval)
    raw_vectors = raw_vectors.apply(tuple)

    return len(set(raw_vectors))


def parse_beta(folder_name):
    match = re.search(r'B([0-9p]+)', folder_name)
    if not match:
        raise ValueError(f"Could not parse beta from {folder_name}")

    beta_str = match.group(1)

    if "p" in beta_str:
        return float(beta_str.replace("p", "."))
    else:
        return float(beta_str)


def main():

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(repo_root, "data")

    run_dirs = glob.glob(os.path.join(data_path, "A2_P4_B*_*"))

    beta_results = defaultdict(list)

    print("\n=== UNIQUE SOLUTIONS PER RUN ===\n")

    for run_dir in sorted(run_dirs):

        csv_path = os.path.join(run_dir, "savedata.csv")

        if not os.path.exists(csv_path):
            print(f"{run_dir} --> savedata.csv not found")
            continue

        try:
            beta = parse_beta(os.path.basename(run_dir))
            count = count_unique_raw_fitness(csv_path)

            beta_results[beta].append(count)

            print(f"{os.path.basename(run_dir)} --> {count}")

        except Exception as e:
            print(f"{run_dir} --> FAILED ({e})")

    # Aggregate statistics
    betas = sorted(beta_results.keys())
    means = []
    stds = []

    print("\n=== LATEX TABLE ===\n")
    print("\\begin{tabular}{c c}")
    print("\\toprule")
    print("Beta & Average Number of Unique Solutions \\\\")
    print("\\midrule")

    for beta in betas:
        mean = np.mean(beta_results[beta])
        std = np.std(beta_results[beta])

        means.append(mean)
        stds.append(std)

        print(f"{beta} & {mean:.2f} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")

    # ---- Plot ----
    plt.figure()
    plt.errorbar(betas, means, yerr=stds, marker='o')
    plt.xlabel("Beta")
    plt.ylabel("Average Number of Unique Solutions")
    plt.title("Effect of Beta on Final Population Diversity")

    plot_path = os.path.join(repo_root, "beta_unique_plot.png")
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)

    print(f"\nPlot saved to: {plot_path}")


if __name__ == "__main__":
    main()
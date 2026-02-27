#!/usr/bin/env python3

import pandas as pd
import ast
import glob
import os
import re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


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

        rounded_mean = int(round(mean))

        print(f"{beta} & {rounded_mean} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    
    # ==========================
    # Statistical Significance vs B=0
    # ==========================
    print("\n=== WELCH T-TESTS vs B=0 ===\n")

    if 0.0 not in beta_results:
        print("B0 not found. Cannot perform statistical tests.")
    else:
        baseline = beta_results[0.0]

        for beta in betas:
            if beta == 0.0:
                continue

            comparison = beta_results[beta]

            t_stat, p_val = ttest_ind(comparison, baseline, equal_var=False)

            mean_diff = np.mean(comparison) - np.mean(baseline)

            print(f"B{beta} vs B0")
            print(f"  Mean Difference: {mean_diff:.2f}")
            print(f"  t-statistic: {t_stat:.3f}")
            print(f"  p-value: {p_val:.5f}")

            if p_val < 0.001:
                print("  Result: *** Highly Significant (p < 0.001)")
            elif p_val < 0.01:
                print("  Result: ** Significant (p < 0.01)")
            elif p_val < 0.05:
                print("  Result: * Significant (p < 0.05)")
            else:
                print("  Result: Not Significant")

            print("")

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
#!/usr/bin/env python3

import pandas as pd
import ast
import glob
import os
import re
from collections import defaultdict
import numpy as np


def count_unique_raw_fitness(csv_path):
    """
    Count unique raw_fitness vectors in the final generation
    of a saved save_data.csv file.
    """
    df = pd.read_csv(csv_path)

    final_gen = df["gen"].max()
    final_df = df[df["gen"] == final_gen]

    # Parse string "[a, b]" into tuple (a, b)
    raw_vectors = final_df["raw_fitness"].apply(ast.literal_eval)
    raw_vectors = raw_vectors.apply(tuple)

    return len(set(raw_vectors))


def parse_beta(folder_name):
    """
    Converts:
        B0     -> 0
        B0p1   -> 0.1
        B0p25  -> 0.25
        B0p5   -> 0.5
        B1     -> 1
    """
    match = re.search(r'B([0-9p]+)', folder_name)
    if not match:
        raise ValueError(f"Could not parse beta from {folder_name}")

    beta_str = match.group(1)

    if "p" in beta_str:
        return float(beta_str.replace("p", "."))
    else:
        return float(beta_str)


def main():

    # Resolve repo root robustly
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

    print("\n=== LATEX TABLE ===\n")

    print("\\begin{tabular}{c c}")
    print("\\toprule")
    print("Beta & Average Number of Unique Solutions \\\\")
    print("\\midrule")

    for beta in sorted(beta_results.keys()):
        avg = np.mean(beta_results[beta])
        print(f"{beta} & {avg:.2f} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")


if __name__ == "__main__":
    main()
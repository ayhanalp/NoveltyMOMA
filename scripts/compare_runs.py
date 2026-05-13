import pandas as pd
import ast
import numpy as np
from glob import glob

DATA_ROOT = "data"
PREFIX = "A2_P4_B0_"
SEEDS = [1]

def load_final_gen(path):
    df = pd.read_csv(path, usecols=["gen", "raw_fitness"])
    df["fitness"] = df["raw_fitness"].apply(ast.literal_eval)

    max_gen = df["gen"].max()
    final = df[df["gen"] == max_gen]["fitness"].tolist()

    # sort for consistent comparison
    final = np.array(final)
    final = final[np.lexsort(final.T[::-1])]

    return final, max_gen

def compare_runs(seed):
    local_path = f"{DATA_ROOT}/{PREFIX}{seed}/savedata.csv"
    hpc_path   = f"{DATA_ROOT}/hpc/{PREFIX}{seed}/savedata.csv"

    local_F, gen_local = load_final_gen(local_path)
    hpc_F,   gen_hpc   = load_final_gen(hpc_path)

    print(f"\nSeed {seed}")
    print(f"Local gen: {gen_local}, HPC gen: {gen_hpc}")

    if len(local_F) != len(hpc_F):
        print(f"Different population sizes: {len(local_F)} vs {len(hpc_F)}")

    # Compare elementwise
    min_len = min(len(local_F), len(hpc_F))

    diffs = np.abs(local_F[:min_len] - hpc_F[:min_len])
    max_diff = np.max(diffs)

    print(f"Max absolute difference: {max_diff}")

    # Show where differences occur
    idx = np.where(np.any(diffs > 1e-8, axis=1))[0]

    print(f"Number of differing individuals: {len(idx)}")

    for i in idx[:10]:  # limit output
        print(f"\nIndex {i}")
        print("Local:", local_F[i])
        print("HPC  :", hpc_F[i])
        print("Diff :", diffs[i])

def main():
    for s in SEEDS:
        compare_runs(s)

if __name__ == "__main__":
    main()
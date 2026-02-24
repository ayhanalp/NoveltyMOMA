import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import pygmo as pg

DATA_ROOT = "data"

experiments = {
    "B=0.0"  : ["A4_P8_B0_1"],
    "B=0.15" : ["A4_P8_B0p15_1"],
    "B=0.3"  : ["A4_P8_B0p3_1"],
    "B=0.5"  : ["A4_P8_B0p5_1"],
}

# ----------------------------
# Pareto front extraction
# ----------------------------
def get_final_front(csv):

    df = pd.read_csv(csv, usecols=["gen","raw_fitness"])
    df["fitness"] = df["raw_fitness"].apply(ast.literal_eval)

    final_gen = df.gen.max()
    pts = df[df.gen == final_gen]["fitness"].tolist()

    ndf,_,_,_ = pg.fast_non_dominated_sorting(points=pts)
    F = np.unique(np.array([pts[i] for i in ndf[0]]), axis=0)

    return F

# ----------------------------
# Plot
# ----------------------------
def main():

    plt.figure(figsize=(8,6))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i,(label,prefixes) in enumerate(experiments.items()):

        files = []
        for p in prefixes:
            files += glob(f"{DATA_ROOT}/{p}*/savedata.csv")

        all_fronts = []

        for f in files:
            F = get_final_front(f)
            all_fronts.append(F)

        F = np.vstack(all_fronts)
        c = colors[i % len(colors)]
        
        F = -1 * F  # Negate if you want to plot in the original objective space (assuming minimization)

        plt.scatter(F[:,0], F[:,1], label=label, alpha=0.7)

    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.title("Final Generation Pareto Fronts")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
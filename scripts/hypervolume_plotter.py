import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.stats import sem
from pymoo.indicators.hv import HV
import pygmo as pg

from concurrent.futures import ProcessPoolExecutor

# ----------------------------
# 1. Specify experiments here
# ----------------------------
DATA_ROOT = "data"
#SEEDS = {1, 2, 3, 4, 5, 6, 7}   # or "all"
SEEDS = "1 2 3 4 5 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 "

experiments = {
    "no entropy"        : ["A2_P4_B0_"],
    "beta = 0.05"       : ["A2_P4_B0p05_"],
}


#     "beta = 0.3"        : ["A2_P4_B0p3_"]
#hv_indicator = HV(ref_point=np.array([1,1]))
hv_indicator = HV(ref_point=np.array([0,0]))
# ----------------------------
# 2. HV computation
# ----------------------------
def compute_hv(csv):
    df = pd.read_csv(csv, usecols=["gen","raw_fitness"])
    df["fitness"] = df["raw_fitness"].apply(ast.literal_eval)

    hv = {}

    for g in df.gen.unique():
        pts = df[df.gen==g]["fitness"].tolist()

        ndf,_,_,_ = pg.fast_non_dominated_sorting(points=pts)
        F = np.unique(np.array([pts[i] for i in ndf[0]]), axis=0)

        hv[g] = hv_indicator(F)

    return hv

# ----------------------------
# 3. Plot
# ----------------------------
def main():
    plt.figure(figsize=(10,6))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i,(label,prefixes) in enumerate(experiments.items()):

        files = []
        for p in prefixes:
            # if plotting all seeds
            if SEEDS == "all":
                pattern = f"{DATA_ROOT}/{p}*/savedata.csv"
                files += glob(pattern)

            # if plotting specific seeds
            else:
                for s in SEEDS:
                    pattern = f"{DATA_ROOT}/{p}{s}/savedata.csv"
                    files += glob(pattern)

        hv_per_gen = {}

        with ProcessPoolExecutor() as ex:
            results = list(ex.map(compute_hv, files))

        for hv in results:
            for g,v in hv.items():
                hv_per_gen.setdefault(g,[]).append(v)

        gens = sorted(hv_per_gen.keys())
        mean = np.array([np.mean(hv_per_gen[g]) for g in gens])
        err  = np.array([sem(hv_per_gen[g])  for g in gens])

        c = colors[i%len(colors)]
        plt.plot(gens,mean,label=label)
        plt.fill_between(gens,mean-err,mean+err,alpha=0.2)

    plt.xlabel("Generation")
    plt.ylabel("Hypervolume")
    plt.title("Pareto Front Hypervolume Over Time")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()


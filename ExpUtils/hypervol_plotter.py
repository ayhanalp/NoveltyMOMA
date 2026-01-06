import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt

# Import HV from pymoo
from pymoo.indicators.hv import HV
# Import pygmo for finding the NDF
import pygmo as pg
# For SEM (standard error of the mean)
from scipy.stats import sem

# Normalize by the absolute value of the minimum NSGA-II fitness (pygmo convention)
NORMALIZE = False

# ----------------------------
# 1. Define Multiple Methods and Their CSV Files
# ----------------------------
'''
methods_data = {
    "nsga2": [
        "/home/raghav/Research/GECCO25/DMO/experiments/data/nsga2_rover_2024_sanity_1ag_2025-01-07_22-16-57_savedata.csv",
        "/home/raghav/Research/GECCO25/DMO/experiments/data/nsga2_rover_2025_sanity_1ag_2025-01-07_22-16-57_savedata.csv",
        "/home/raghav/Research/GECCO25/DMO/experiments/data/nsga2_rover_2026_sanity_1ag_2025-01-07_22-16-57_savedata.csv",
        "/home/raghav/Research/GECCO25/DMO/experiments/data/nsga2_rover_2027_sanity_1ag_2025-01-07_22-16-57_savedata.csv",
        "/home/raghav/Research/GECCO25/DMO/experiments/data/nsga2_rover_2028_sanity_1ag_2025-01-07_22-16-57_savedata.csv",
    ],
    "dmo": [
        "/home/raghav/Research/GECCO25/DMO/experiments/data/dmo_rover_2024_sanity_1ag_2025-01-07_22-16-57_savedata.csv",
        "/home/raghav/Research/GECCO25/DMO/experiments/data/dmo_rover_2025_sanity_1ag_2025-01-07_22-16-57_savedata.csv",
        "/home/raghav/Research/GECCO25/DMO/experiments/data/dmo_rover_2026_sanity_1ag_2025-01-07_22-16-57_savedata.csv",
        "/home/raghav/Research/GECCO25/DMO/experiments/data/dmo_rover_2027_sanity_1ag_2025-01-07_22-16-57_savedata.csv",
        "/home/raghav/Research/GECCO25/DMO/experiments/data/dmo_rover_2028_sanity_1ag_2025-01-07_22-16-57_savedata.csv",
    ],
    "kpnsga2": [
        "/home/raghav/Research/GECCO25/DMO/experiments/data/kpnsga2_rover_2024_sanity_1ag_2025-01-07_22-16-57_savedata.csv",
        "/home/raghav/Research/GECCO25/DMO/experiments/data/kpnsga2_rover_2025_sanity_1ag_2025-01-07_22-16-57_savedata.csv",
        "/home/raghav/Research/GECCO25/DMO/experiments/data/kpnsga2_rover_2026_sanity_1ag_2025-01-07_22-16-57_savedata.csv",
        "/home/raghav/Research/GECCO25/DMO/experiments/data/kpnsga2_rover_2027_sanity_1ag_2025-01-07_22-16-57_savedata.csv",
        "/home/raghav/Research/GECCO25/DMO/experiments/data/kpnsga2_rover_2028_sanity_1ag_2025-01-07_22-16-57_savedata.csv",
    ],
    # "nsga2+d": [
    #     "/home/raghav/Research/GECCO25/DMO/experiments/data/nsga2+d_rover_2024_sanity_1ag_2025-01-07_22-16-57_savedata.csv",
    #     "/home/raghav/Research/GECCO25/DMO/experiments/data/nsga2+d_rover_2025_sanity_1ag_2025-01-07_22-17-05_savedata.csv",
    #     "/home/raghav/Research/GECCO25/DMO/experiments/data/nsga2+d_rover_2026_sanity_1ag_2025-01-07_22-17-16_savedata.csv",
    #     "/home/raghav/Research/GECCO25/DMO/experiments/data/nsga2+d_rover_2027_sanity_1ag_2025-01-07_22-17-12_savedata.csv",
    #     "/home/raghav/Research/GECCO25/DMO/experiments/data/nsga2+d_rover_2028_sanity_1ag_2025-01-07_22-17-14_savedata.csv",
    # ],
    # Add more methods as needed
}
'''

methods_data = {
    "nsga2": [
        "/home/santjami/repos/NoveltyMOMA/data/nsga2_rover_2024_test1ag_2025-12-08_09-20-27_savedata.csv",
    ],
    "nsga2+entropy": [
        "/home/santjami/repos/NoveltyMOMA/data/nsga2_rover_2024_negh1ag_01beta_10k_2025-12-08_21-15-09_savedata.csv",
    ],
}
#"/home/santjami/repos/NoveltyMOMA/data/nsga2_rover_2024_negh1ag_1beta_2025-12-08_12-45-07_savedata.csv",
# "/home/santjami/repos/NoveltyMOMA/data/nsga2_rover_2024_h1ag_1beta_2025-12-08_11-54-21_savedata.csv",

# --------------------------------------------------
# 2. Initialize HV with a reference point
#    Choose the ref_point carefully for your data!
# --------------------------------------------------
hv_indicator = HV(ref_point=np.array([0, 0]))

if NORMALIZE:

    baseline_min = +np.inf

    for file in methods_data["nsga2"]:
        df = pd.read_csv(file, usecols=["fitness"])
        fitness_lists = df["fitness"].apply(ast.literal_eval)
        for fl in fitness_lists:
            baseline_min = min(baseline_min, *fl)

        baseline_scale = abs(baseline_min)
else:
    baseline_scale = 1.0

# ----------------------------
# 3. Helper Function
# ----------------------------
def compute_hv_for_file(csv_file, baseline_scale):
    """
    Reads a CSV file, parses the 'fitness' column,
    computes the Non-dominated Front for each generation,
    and returns a dictionary: gen -> hypervolume.
    """
    df = pd.read_csv(csv_file, usecols=["gen", "fitness"])
    df["fitness_list"] = df["fitness"].apply(lambda x: ast.literal_eval(x))
    
    hv_dict = {}
    for g in sorted(df["gen"].unique()):
        raw = df[df["gen"] == g]["fitness_list"].tolist()

        if NORMALIZE:
            # Normalize each objective using baseline constant
            gen_data = [[f / baseline_scale for f in fl] for fl in raw]
        else:
            #gen_data = df[df["gen"] == g]["fitness_list"].tolist()
            gen_data = raw

        # Compute non-dominated front
        ndf, _, _, _ = pg.fast_non_dominated_sorting(points=gen_data)
        ndf_fitnesses = np.array([gen_data[i] for i in ndf[0]])
        # Remove duplicates
        F = np.unique(ndf_fitnesses, axis=0)
        hv_val = hv_indicator(F)
        hv_dict[g] = hv_val
    return hv_dict

# ----------------------------
# 4. Plot Settings
# ----------------------------
plt.figure(figsize=(10, 6))

# Define a colormap or cycle through default colors
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

method_idx = 0

# ----------------------------
# 5. Loop Over Methods, Aggregate, and Plot
# ----------------------------
for method_name, csv_files in methods_data.items():
    
    if not csv_files:
        # Skip if there are no files for this method
        continue
    
    # Dictionary to collect generation -> [HV values across files]
    hv_per_generation = {}
    
    # Process each file for this method
    for file_path in csv_files:
        hv_dict = compute_hv_for_file(file_path, baseline_scale)
        for gen, hv_val in hv_dict.items():
            if gen not in hv_per_generation:
                hv_per_generation[gen] = []
            hv_per_generation[gen].append(hv_val)
    
    # Compute mean and SEM
    sorted_gens = sorted(hv_per_generation.keys())
    mean_hvs, sem_hvs = [], []
    for gen in sorted_gens:
        values = hv_per_generation[gen]
        mean_hv = np.mean(values)
        sem_hv = sem(values)
        mean_hvs.append(mean_hv)
        sem_hvs.append(sem_hv)
    
    mean_hvs = np.array(mean_hvs)
    sem_hvs = np.array(sem_hvs)
    
    color = colors[method_idx % len(colors)]
    method_idx += 1
    
    # Plot the mean hypervolume
    plt.plot(sorted_gens, mean_hvs, label=method_name, color=color)
    
    # Plot the shaded region for ±SEM
    plt.fill_between(sorted_gens,
                     mean_hvs - sem_hvs,
                     mean_hvs + sem_hvs,
                     alpha=0.2,
                     color=color)

# ----------------------------
# 6. Final Touches
# ----------------------------
plt.xlabel("Generation")
plt.ylabel("Hypervolume")    # <- Updated label
plt.title("Comparison of HV Across Multiple Methods")  # <- Updated title
plt.grid(True)
plt.legend()
plt.show()

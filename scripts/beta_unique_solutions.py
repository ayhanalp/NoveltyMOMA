#!/usr/bin/env python3

"""
Compute per-beta stats for unique final raw-fitness vectors and unique Pareto-front solutions.

This script uses pygmo.fast_non_dominated_sorting() (same as NSGAII) to extract the Pareto front.
It expects run directories like: data/hpc_time_obj_1/A2_P12_B0p05_1/ with a savedata.csv inside.
"""

import ast
import glob
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

import pygmo as pg


def parse_beta(folder_name):
    m = re.search(r'B([0-9p]+)', folder_name)
    if not m:
        raise ValueError(f"Could not parse beta from {folder_name}")
    s = m.group(1)
    return float(s.replace('p', '.')) if 'p' in s else float(s)


def parse_seed(folder_name):
    m = re.search(r'_(\d+)$', folder_name)
    if not m:
        raise ValueError(f"Could not parse seed from {folder_name}")
    return int(m.group(1))


def _normalize_raw(raw_entry):
    """Normalize the raw_fitness field to a tuple (ordered by objective index).
    raw_entry may be a list/tuple or a dict like {0: val0, 1: val1}.
    """
    if isinstance(raw_entry, str):
        raw = ast.literal_eval(raw_entry)
    else:
        raw = raw_entry

    if isinstance(raw, dict):
        # Use numeric keys sorted
        keys = sorted(raw.keys())
        return tuple(raw[k] for k in keys)
    elif isinstance(raw, (list, tuple)):
        return tuple(raw)
    else:
        # Single-value fitness
        return (raw,)


def count_unique_raw_fitness(csv_path):
    df = pd.read_csv(csv_path)
    final_gen = df['gen'].max()
    final_df = df[df['gen'] == final_gen]

    raw_cols = final_df['raw_fitness'].apply(_normalize_raw)
    unique = set(raw_cols.tolist())
    return len(unique)


def count_unique_pareto(csv_path):
    df = pd.read_csv(csv_path)
    final_gen = df['gen'].max()
    final_df = df[df['gen'] == final_gen]

    raw_parsed = final_df['raw_fitness'].apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else s).tolist()
    if len(raw_parsed) == 0:
        return 0

    pts = []
    for r in raw_parsed:
        if isinstance(r, dict):
            keys = sorted(r.keys())
            pts.append(tuple(r[k] for k in keys))
        elif isinstance(r, (list, tuple)):
            pts.append(tuple(r))
        else:
            pts.append((r,))

    ndf, _, _, _ = pg.fast_non_dominated_sorting(points=pts)
    pareto_idxs = ndf[0] if ndf else []
    pareto_pts = [pts[i] for i in pareto_idxs]
    return len(set(pareto_pts))


def aggregate_and_report(results_dict, title, out_png, xlabel='Beta', ylabel='Value'):
    betas = sorted(results_dict.keys())
    means = [np.mean(results_dict[b]) for b in betas]
    stds = [np.std(results_dict[b]) for b in betas]

    print(f"\n=== {title} ===\n")
    print("\\begin{tabular}{c c}")
    print("\\toprule")
    # Use .format to avoid backslash escaping issues in f-strings
    print("{} & Average {} \\\\".format(xlabel, ylabel))
    print("\\midrule")
    for b, m in zip(betas, means):
        print("{} & {} \\\\".format(b, int(round(m))))
    print("\\bottomrule")
    print("\\end{tabular}\n")

    if 0.0 in results_dict:
        base = results_dict[0.0]
        print(f"--- Welch t-tests vs B=0 for {title} ---")
        for b in betas:
            if b == 0.0:
                continue
            comp = results_dict[b]
            t_stat, p_val = ttest_ind(comp, base, equal_var=False)
            md = np.mean(comp) - np.mean(base)
            print(f"B{b} vs B0: mean_diff={md:.2f}, t={t_stat:.3f}, p={p_val:.4f}")
        print("")

    plt.figure()
    plt.errorbar(betas, means, yerr=stds, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"Saved plot to {out_png}")


def aggregate_and_report_float(results_dict, title, out_png,
                               xlabel='Beta', ylabel='Value'):
    """Same as aggregate_and_report but prints float means with std dev."""

    betas = sorted(results_dict.keys())
    means = [np.mean(results_dict[b]) for b in betas]
    stds = [np.std(results_dict[b]) for b in betas]

    print(f"\n=== {title} ===\n")

    # Pretty console output
    for b, m, s in zip(betas, means, stds):
        print(f"Beta={b:<5} mean={m:.4f} std={s:.4f} n={len(results_dict[b])}")

    # LaTeX table
    print("\n\\begin{tabular}{c c}")
    print("\\toprule")
    print("{} & Average {} \\\\".format(xlabel, ylabel))
    print("\\midrule")

    for b, m in zip(betas, means):
        print("{} & {:.4f} \\\\".format(b, m))

    print("\\bottomrule")
    print("\\end{tabular}\n")

    # Welch t-tests vs beta=0
    if 0.0 in results_dict:
        base = results_dict[0.0]

        print(f"--- Welch t-tests vs B=0 for {title} ---")

        for b in betas:
            if b == 0.0:
                continue

            comp = results_dict[b]

            t_stat, p_val = ttest_ind(comp, base, equal_var=False)
            md = np.mean(comp) - np.mean(base)

            print(
                f"B{b} vs B0: "
                f"mean_diff={md:.4f}, "
                f"t={t_stat:.3f}, "
                f"p={p_val:.4f}"
            )

        print("")

    # Plot
    plt.figure()
    plt.errorbar(betas, means, yerr=stds, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)

    print(f"Saved plot to {out_png}")


from pymoo.indicators.hv import HV

# SAME reference point as plotting script
hv_indicator = HV(ref_point=np.array([0, 0]))


def compute_final_hypervolume(csv_path):
    """
    Compute hypervolume of the final generation Pareto front.

    Matches the behavior of the hypervolume plotting script exactly.
    """

    df = pd.read_csv(csv_path)

    final_gen = df['gen'].max()
    final_df = df[df['gen'] == final_gen]

    raw_parsed = final_df['raw_fitness'].apply(
        lambda s: ast.literal_eval(s) if isinstance(s, str) else s
    ).tolist()

    if len(raw_parsed) == 0:
        return 0.0

    pts = []

    for r in raw_parsed:
        if isinstance(r, dict):
            keys = sorted(r.keys())
            pts.append(tuple(r[k] for k in keys))

        elif isinstance(r, (list, tuple)):
            pts.append(tuple(r))

        else:
            pts.append((r,))

    # Extract Pareto front exactly like plotter
    ndf, _, _, _ = pg.fast_non_dominated_sorting(points=pts)

    pareto = np.unique(
        np.array([pts[i] for i in ndf[0]], dtype=float),
        axis=0
    )

    # Compute HV exactly like plotter
    hv = hv_indicator(pareto)

    return float(hv)


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # Use the HPC aggregated dataset for time objective experiments
    data_root = os.path.join(repo_root, 'data', 'hpc_time_obj1_full', 'time_obj_1')
    #data_root = os.path.join(repo_root, 'data', 'wall_objectives')
    run_dirs = glob.glob(os.path.join(data_root, 'A2_P12_B*_*'))
    run_dirs = sorted(run_dirs)

    allowed_seeds = None

    raw_results = defaultdict(list)
    pareto_results = defaultdict(list)
    hv_results = defaultdict(list)

    print(f"Found {len(run_dirs)} run directories under {data_root}")

    for rd in run_dirs:
        csv_path = os.path.join(rd, 'savedata.csv')
        if not os.path.exists(csv_path):
            print(f"Missing savedata.csv in {rd}")
            continue
        try:
            folder = os.path.basename(rd)
            beta = parse_beta(folder)
            seed = parse_seed(folder)
            if allowed_seeds is not None and seed not in allowed_seeds:
                continue
            u = count_unique_raw_fitness(csv_path)
            #up = count_unique_pareto(csv_path)
            up = 0
            hv = compute_final_hypervolume(csv_path)
            raw_results[beta].append(u)
            pareto_results[beta].append(up)
            # collect hypervolume results
            hv_results[beta].append(hv)
            #print(f"{folder}: unique_raw={u}, unique_pareto={up}")
            print(
                f"{folder}: "
                f"seed={seed}, "
                f"beta={beta}, "
                f"unique_raw={u}, "
                f"unique_pareto={up}, "
                f"hypervolume={hv:.4f}"
            )
        except Exception as e:
            print(f"Failed processing {rd}: {e}")

    aggregate_and_report(raw_results,
                         title='Average Unique Final Raw-Fitness Solutions per Beta',
                         out_png=os.path.join(repo_root, 'beta_unique_plot.png'),
                         xlabel='Beta', ylabel='Unique Solutions')

    aggregate_and_report(pareto_results,
                         title='Average Unique Pareto-Front Solutions per Beta',
                         out_png=os.path.join(repo_root, 'beta_pareto_unique_plot.png'),
                         xlabel='Beta', ylabel='Unique Pareto Solutions')

    # Aggregate and report hypervolumes (float values)
    aggregate_and_report_float(
        hv_results,
        title='Average Final Hypervolume per Beta',
        out_png=os.path.join(repo_root, 'beta_hypervolume_plot.png'),
        xlabel='Beta',
        ylabel='Hypervolume'
    )


if __name__ == '__main__':
    main()
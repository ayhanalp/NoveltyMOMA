"""
Resume script for NSGA-II experiments using the per-run latest_checkpoint.pth file.

Usage:
  python scripts/resume_nsga2.py /absolute/path/to/data/<run_dir>

This will load data/<run>/latest_checkpoint.pth, reconstruct the NSGAII algorithm's
population (for NSGAII only), and continue evolution for remaining generations.
"""
import os
import sys
import torch

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

from algorithms.NSGAII import NSGAII
from Individual import Individual

def resume(run_dir):
    latest = os.path.join(run_dir, 'latest_checkpoint.pth')
    if not os.path.exists(latest):
        raise FileNotFoundError(f"No checkpoint found at {latest}")

    ckpt = torch.load(latest, map_location='cpu')

    # Configs and data paths saved in run_dir
    data_filename = os.path.join(run_dir, 'savedata.csv')
    alg_config = os.path.join(run_dir, 'algconfig.yaml')
    env_config = os.path.join(run_dir, 'envconfig.yaml')

    # Instantiate algorithm (this will create an initial population we will overwrite)
    alg = NSGAII(alg_config_filename=alg_config,
                 domain_name='rover',
                 rover_config_filename=env_config,
                 data_filename=data_filename)

    # Reconstruct population from checkpoint
    new_pop = []
    for ind_entry in ckpt.get('population', []):
        ind = Individual(config_filename=alg.config_filename,
                         num_agents=alg.team_size,
                         input_size=alg.interface.get_state_size(),
                         output_size=alg.interface.get_action_size(),
                         id=ind_entry.get('id', -1),
                         num_objs=alg.num_objs)
        # Load policy state_dicts
        for p, sd in zip(ind.joint_policy, ind_entry.get('policies', [])):
            try:
                p.load_state_dict(sd)
            except Exception:
                # continue if loading fails; best-effort
                pass
        ind.fitness = ind_entry.get('fitness', ind.fitness)
        ind.raw_fitness = ind_entry.get('raw_fitness', ind.raw_fitness)
        new_pop.append(ind)

    alg.pop = new_pop
    alg.glob_ind_counter = ckpt.get('glob_ind_counter', alg.glob_ind_counter)

    start_gen = ckpt.get('gen', 0) + 1
    traj_write_freq = ckpt.get('traj_write_freq', 100)

    print(f"Resuming run {run_dir} from gen {start_gen} (num_gens={alg.num_gens})")

    for gen in range(start_gen, alg.num_gens):
        print(f"Running gen {gen}/{alg.num_gens-1}")
        alg.evolve(gen=gen, traj_write_freq=traj_write_freq)

    print("Resume finished")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python scripts/resume_nsga2.py /abs/path/to/data/<run_dir>")
        sys.exit(1)
    run_dir = sys.argv[1]
    if not os.path.isabs(run_dir):
        run_dir = os.path.join(repo_root, run_dir)
    resume(run_dir)

import random
import torch
import numpy
import sys
import datetime
import shutil # for file management
import os
import re

import algorithms.NSGAII as NSGAII
import algorithms.KParentNSGAII as KParentNSGAII
import algorithms.DMO as DMO
import algorithms.NSGAII_D as NSGAII_D

# Absolute path to the repo root (directory containing main.py)
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    assert len(sys.argv) == 10, "Correct usage: python alg_name domain_name data_dirpath alg_config env_config seed label traj_write_freq"
   
    # Process the command line args
    alg_name = sys.argv[1]
    assert alg_name in ['nsga2', 'kpnsga2', 'dmo', 'nsga2+d'], "Unrecognised alg_name"
    domain_name = sys.argv[2]
    assert domain_name in ['rover'], 'Uncrecognised domain_name'
    data_dir = sys.argv[3]
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(REPO_ROOT, data_dir)
    data_dir = data_dir+'/' if data_dir[-1]!='/' else data_dir # Add a directory '/' at the end
    src_alg_config_filename = sys.argv[4]
    if not os.path.isabs(src_alg_config_filename):
        src_alg_config_filename = os.path.join(REPO_ROOT, src_alg_config_filename)
    src_env_config_filename = sys.argv[5]
    if not os.path.isabs(src_env_config_filename):
        src_env_config_filename = os.path.join(REPO_ROOT, src_env_config_filename)
    seed_val = int(sys.argv[6])
    seed_val_str = str(seed_val)
    label = sys.argv[7]
    traj_write_freq = int(sys.argv[8])
    beta = float(sys.argv[9])

    # Datetime for file naming
    datetime_now_string = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Sanitize label for filesystem safety
    safe_label = label.replace(' ', '_').replace('/', '_')
    
    # Find existing runs with the same label
    existing = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
        and re.match(rf"^{re.escape(safe_label)}_\d+$", d)
    ]

    # Extract run numbers
    run_nums = [
        int(d.split('_')[-1]) for d in existing
    ]

    next_run_num = max(run_nums) + 1 if run_nums else 1
    
    # Create a run-specific subdirectory
    run_id = f"{safe_label}_{next_run_num}"
    run_dir = os.path.join(data_dir, run_id)
    os.makedirs(run_dir, exist_ok=False)
    print(f"Data will be saved to directory: {run_dir}")
    
    # Save the metadata
    metadata_path = os.path.join(run_dir, 'metadata.txt')

    with open(metadata_path, 'w') as f:
        f.write(f"algorithm: {alg_name}\n")
        f.write(f"domain: {domain_name}\n")
        f.write(f"seed: {seed_val}\n")
        f.write(f"label: {label}\n")
        f.write(f"run_number: {next_run_num}\n")
        f.write(f"traj_write_freq: {traj_write_freq}\n")
        f.write(f"beta: {beta}\n")
        f.write(f"datetime: {datetime_now_string}\n")
        f.write(f"alg_config_source: {src_alg_config_filename}\n")
        f.write(f"env_config_source: {src_env_config_filename}\n")
        f.write(f"beta: {beta}\n")

    # Save data filename
    data_filename = os.path.join(run_dir, 'savedata.csv')
    
    # Create copy of configs at save data location
    dest_alg_config_filename = os.path.join(run_dir, 'algconfig.yaml')
    shutil.copyfile(src_alg_config_filename, dest_alg_config_filename)
    dest_env_config_filename = os.path.join(run_dir, 'envconfig.yaml')
    shutil.copyfile(src_env_config_filename, dest_env_config_filename)

    # Set the seed value for all libraries
    random.seed(seed_val)
    torch.manual_seed(seed_val)
    numpy.random.seed(seed_val)

    # Initialise the algorithm based on alg name
    if alg_name == 'nsga2':
        alg = NSGAII.NSGAII(alg_config_filename=dest_alg_config_filename,
                            domain_name=domain_name,
                            data_filename=data_filename,
                            rover_config_filename=dest_env_config_filename,
                            beta=beta)
    elif alg_name == 'kpnsga2':
        alg = KParentNSGAII.KParentNSGAII(alg_config_filename=dest_alg_config_filename,
                                          domain_name=domain_name,
                                          data_filename=data_filename,
                                          rover_config_filename=dest_env_config_filename)
    elif alg_name == 'dmo':
        alg = DMO.DMO(alg_config_filename=dest_alg_config_filename,
                      domain_name=domain_name,
                      data_filename=data_filename,
                      rover_config_filename=dest_env_config_filename)
    elif alg_name == 'nsga2+d':
        alg = NSGAII_D.NSGAII_D(alg_config_filename=dest_alg_config_filename,
                                domain_name=domain_name,
                                data_filename=data_filename,
                                rover_config_filename=dest_env_config_filename)
    
    # Run the algorithm
    next_print = 0.0  # percent
    num_gens = alg.num_gens

    for gen in range(num_gens):
        percent = (gen / num_gens) * 100

        if percent >= next_print:
            print(f"Progress: {int(next_print)}% ({gen}/{num_gens})")
            next_print += 1.0
        alg.evolve(gen=gen, traj_write_freq=traj_write_freq)

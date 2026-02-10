HPC run notes for NoveltyMOMA

Quick steps to run on a typical SLURM cluster:

1. Create or load a Python environment:

   - Using conda (recommended):
     conda env create -f environment.yml -n noveltymoma
     conda activate noveltymoma

   - Or use system modules as required by your HPC (Python, CUDA, etc.).

2. Prepare a data directory for your run (absolute paths are safer on HPC):

   mkdir -p /scratch/$USER/noveltymoma/data

3. Copy config files into the repo or reference absolute config paths.

4. Use the provided SLURM template at `scripts/hpc_job_template.sh`.
   - Edit the SBATCH headers (time, partition, mem, cpus, gres) to match your cluster.
   - Update the python command at the bottom with absolute paths for `main.py`, data dir and config files.

5. Submit the job:

   sbatch scripts/hpc_job_template.sh

Notes and best practices
- Use absolute paths for data and config files to avoid issues with the working directory.
- The script writes `env_instance.yaml` at interface initialization and saves `env_instance.png` on the first rollout. Both are written to the run directory (e.g., `/scratch/$USER/noveltymoma/data/<label>_1/`).
- Headless plotting is supported; use the `--no-show` flag when running the plotting script under a batch job.
- Ensure random seeds are set (the CLI already sets seeds for random, torch, and numpy).

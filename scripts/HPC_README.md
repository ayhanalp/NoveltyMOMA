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

Resuming interrupted runs (quick steps)
1. Find the run directory created by the job, e.g. `/scratch/$USER/noveltymoma/data/<label>_1`.
2. Confirm a checkpoint exists: `ls -l <run_dir>/latest_checkpoint.pth` and that `metadata.txt` and `env_instance.yaml` are present.
3. Backup the CSV before resuming: `cp <run_dir>/savedata.csv <run_dir>/savedata.csv.bak`.
4. Activate the same Python environment you used for the run (conda/environment.yml):
   ```bash
   conda activate noveltymoma
   ```
5. Resume NSGA-II runs with the helper script (it loads the checkpoint and continues):
   ```bash
   python3 scripts/resume_nsga2.py /absolute/path/to/<run_dir>
   ```
   - To run under the scheduler, create an sbatch script that calls the same resume command and submit it.
6. After restart, monitor the run output and `savedata.csv` to confirm new generations are appended.

Notes:
- If `latest_checkpoint.pth` is missing the run cannot be resumed; consider re-running from scratch with the same seed.
- The minimal resume helper currently supports NSGA-II. If you need resume for other algorithms, add checkpointing for them or ask for help.

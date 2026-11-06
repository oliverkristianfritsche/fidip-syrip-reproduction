# SKILL: cluster workflow for smalldata-project

This file tells an agent how to run this project on the Explorer cluster.

## Connection

| Item | Value |
| --- | --- |
| Login | `ssh explorer` (key login, no password) |
| User | `fritsche.o` |
| Transfer host | `ssh -o HostName=xfer.discovery.neu.edu explorer` |
| Project folder | `/scratch/fritsche.o/smalldata-project` |
| Environment | `module load anaconda3/2024.06`, then `conda activate <project folder>/env/smalldata-project` |
| Internet | The cluster has no GitHub access. Code arrives over ssh |

## Cluster layout

```text
code/<commit>/, code/current                   pushed code; jobs run code/current
upstream/Infant-Pose-Estimation/               pinned upstream code, never edited
env/smalldata-project/                         conda environment
data/syrip/, data/models/, data/_zips/         SyRIP, the weights, and the SyRIP archives
raw/train_<config>_<tag>/                      one folder per training run; a resubmit resumes it
raw/smalldata-project_baseline_run_<job id>/   the baseline job
logs/                                          Slurm logs; secret, never pull
```

## Workflow

1. Push the code: `bash scripts/push_code.sh`. The push and pull scripts name the cluster folder after the repository folder. In a clone with another name, run `export PROJECT=smalldata-project` first. The script ships `src`, `scripts`, `jobs`, `configs`, and `requirements.txt` with a hash file. No commit is needed.
2. Push the upstream code once: `bash scripts/push_upstream.sh`. Push the data once: `bash scripts/push_data.sh`.
3. Check the queue: `ssh explorer 'squeue -u $USER'`.
4. Submit from the project folder: `ssh explorer 'cd /scratch/$USER/smalldata-project && timeout 60 sbatch <flags> code/current/jobs/<job>.sbatch'`.
5. Watch: `ssh explorer 'sacct -X -n -j <ids> -o JobID,JobName%40,State,Elapsed'`.
6. Pull the results: `bash scripts/pull_results.sh raw/<run folder> results/training/<name>`. Checkpoints stay on the cluster.
7. Update the table: `python scripts/summarize_training.py`.
8. Copy the loss logs of the training runs over the transfer host (`MANIFEST.md` gives the command), then run `python scripts/extract_loss_curves.py`.

## Jobs

| Job file in `jobs/` | What it does | How we submit it |
| --- | --- | --- |
| `build_env.sbatch` | Builds the conda environment | Header: `sharing`, no GPU, 1 h |
| `prepare_data.sbatch` | Checks the counts, the md5 sums, and the image sizes | Header: `short`, no GPU, 20 min |
| `train_model.sbatch` | Trains one config with the upstream trainer, then evaluates the final and the best epoch | `-p gpu-short --gres=gpu:h200:1 --time=01:00:00`; the paper-settings config needs `--time=01:15:00` |
| `reproduce_baseline.sbatch` | Runs the environment check, the released models with each `test11.jpg`, and the demo | Header: `gpu-short`, `h200:1`, 45 min |
| `make_clean_split.sbatch`, `make_select_split.sbatch` | Build the cleaned data and the selection split | Header: `short`, no GPU, 15 min |
| `train_with_validation.sbatch` | Trains with a seed and a freeze depth, with checkpoint selection on the selection split | `-p gpu-short --gres=gpu:h200:1 --time=01:00:00` |

- The headers of `train_model.sbatch` and `train_with_validation.sbatch` ask for an A100. Always override them with `--gres=gpu:h200:1`, so every number comes from one GPU type.
- Name jobs `smalldata-project_<step>_<smoke|run>`.
- Run a smoke test first: `-p sharing --time=00:30:00 --export=ALL,EPOCHS=1,PRE_EPOCHS=1,MAX_SAMPLES=16`. Its numbers are not results.

## Exact commands for this project

Run these from the project folder on the cluster: `cd /scratch/fritsche.o/smalldata-project`.

```bash
timeout 60 sbatch -J smalldata-project_build_run code/current/jobs/build_env.sbatch       # the conda environment
H200='-p gpu-short --gres=gpu:h200:1'
for n in 1 2 3; do                                                                        # three configs, three runs each
  timeout 60 sbatch -J smalldata-project_train_run $H200 --time=01:00:00 --export=ALL,CFG=configs/hrnet_w48_384x288_infant_baseline.yaml,TAG=rep$n code/current/jobs/train_model.sbatch
  timeout 60 sbatch -J smalldata-project_train_run $H200 --time=01:00:00 --export=ALL,CFG=configs/hrnet_w48_384x288_infant_fidip.yaml,TAG=rep$n code/current/jobs/train_model.sbatch
  timeout 60 sbatch -J smalldata-project_train_run $H200 --time=01:15:00 --export=ALL,CFG=configs/hrnet_w48_384x288_infant_fidip_paper.yaml,TAG=rep$n code/current/jobs/train_model.sbatch
done
timeout 60 sbatch code/current/jobs/reproduce_baseline.sbatch                             # environment check, released models, demo

# training changes: build the two splits once; start the loop after build_env and both split jobs finish
timeout 60 sbatch -J smalldata-project_clean_run code/current/jobs/make_clean_split.sbatch
timeout 60 sbatch -J smalldata-project_select_run code/current/jobs/make_select_split.sbatch
for n in 1 2 3; do                                                                        # SEED = n - 1
  timeout 60 sbatch -J smalldata-project_train_run $H200 --time=01:00:00 --export=ALL,CFG=configs/hrnet_w48_384x288_infant_fidip.yaml,TAG=clean_rep$n,BASE=variants/clean code/current/jobs/train_model.sbatch
  timeout 60 sbatch -J smalldata-project_train_run $H200 --time=01:00:00 --export=ALL,CFG=configs/hrnet_w48_384x288_infant_fidip_lr1e-4.yaml,TAG=rep$n code/current/jobs/train_model.sbatch
  timeout 60 sbatch -J smalldata-project_trainval_run $H200 --time=01:00:00 --export=ALL,CFG=configs/hrnet_w48_384x288_infant_fidip_lr1e-4.yaml,TAG=rep$n,BASE=variants/clean,SEED=$((n-1)) code/current/jobs/train_with_validation.sbatch
  timeout 60 sbatch -J smalldata-project_trainval_run $H200 --time=01:00:00 --export=ALL,CFG=configs/hrnet_w48_384x288_infant_fidip_lr1e-4.yaml,TAG=freeze0_rep$n,BASE=variants/clean,SEED=$((n-1)),FREEZE=0 code/current/jobs/train_with_validation.sbatch
  timeout 60 sbatch -J smalldata-project_trainval_run $H200 --time=01:00:00 --export=ALL,CFG=configs/hrnet_w48_384x288_infant_fidip_lr1e-4.yaml,TAG=freeze484_rep$n,BASE=variants/clean,SEED=$((n-1)),FREEZE=484 code/current/jobs/train_with_validation.sbatch
  timeout 60 sbatch -J smalldata-project_trainval_run $H200 --time=01:00:00 --export=ALL,CFG=configs/hrnet_w48_384x288_infant_fidip_lr1e-4_lambda5e-3.yaml,TAG=rep$n,BASE=variants/clean,SEED=$((n-1)) code/current/jobs/train_with_validation.sbatch
  timeout 60 sbatch -J smalldata-project_trainval_run $H200 --time=01:00:00 --export=ALL,CFG=configs/hrnet_w48_384x288_infant_fidip_lr1e-4_lambda5e-2.yaml,TAG=rep$n,BASE=variants/clean,SEED=$((n-1)) code/current/jobs/train_with_validation.sbatch
done

squeue -u $USER -o "%i %j %P %T %M %l %R"
sacct -X -j <ids> -o JobID,JobName%40,Partition,State,Elapsed,ExitCode
```

- Compare two configs image by image, locally: `python scripts/compare_runs.py --treatment hrnet_w48_384x288_infant_fidip_clean --control hrnet_w48_384x288_infant_fidip`.
- A training run resumes from its `checkpoint.pth`. If a run hits its time limit, submit the same command again with `--dependency=afterany:<job id>`.
- A finished run skips itself, so a repeated command is safe.
- Ask for the time a job needs, not the maximum. The scheduler fits short jobs into gaps, so they start sooner.

## Local and cluster

| Work | Machine |
| --- | --- |
| Every GPU job: environment check, training, evaluation, demo | Cluster, one H200 |
| Code edits, figures, report | Local, no GPU |

## Local runs

- Run local work in WSL with `wsl.exe -d Ubuntu -- bash <script file>`.
- Write a script file. The wrapper expands `$` and does not keep nested quotes.
- The wrapper always returns 0. Print OK or FAILED and read that line.
- Make the report figures: `python scripts/make_report_figures.py` in the `smalldata-project` environment.
- Build the report: `cd report && tectonic report.tex` in the `tectonic` environment.

## Active runs

None. Every run of this submission is complete. `MANIFEST.md` lists each job.

## Rules

- Ask the owner before any `scancel`.
- Never touch a job or a file of another project. Other projects share this user account and its submit limits.
- Never pull `logs/`. Never print the environment in a job.
- Never edit files under `code/` or `upstream/` on the cluster. Edit locally and push.
- Do not pass `--qos`.
- Wrap `sbatch` and `scancel` in `timeout 60`. Keep each ssh call short.
- Submit to one partition. A job on `-p gpu-short,gpu` meets both submit limits and can fail.
- After an empty `sbatch` reply, check `squeue` and `sacct` by job name. Never resubmit before that check.
- Run `md5sum` and `unzip` on large files on the transfer host, never on the login node.
- When a loader fails on a missing image, count the files under `data/syrip/`. Restore them with `scripts/push_data.sh --only syrip`.
- Hold a queued job with `scontrol hold` instead of cancelling it.
- Log each submit and result in `AGENT.md` at the time it happens.

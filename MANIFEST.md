# MANIFEST: inventory for smalldata-project

This file lists the data, weights, code, runs, and final numbers. `<root>` is the data root in `configs/paths.yaml`: `/scratch/fritsche.o/smalldata-project/` on the cluster. Data and weights never enter the repository.

## Final numbers

"Paper" is the number in the FiDIP paper. "Ours" is our number, made in one of two ways:
- our evaluation of the authors' released checkpoint;
- our training runs from the COCO weights: the final epoch, mean of three runs.

| Model | Paper AP | Our AP | How we got our AP |
| --- | ---: | ---: | --- |
| Released HRNet-W48 checkpoint | 93.6 | 93.60 | Our evaluation, `test11.jpg` of the 2023 archive |
| Released HRNet-W48 checkpoint | 93.6 | 92.15 | Our evaluation, `test11.jpg` of the 2024 archive |
| Released MobileNetV2 checkpoint | 79.2 | 79.18 | Our evaluation, `test11.jpg` of the 2023 archive |
| Guide config | 92.7 | 91.50 | Our training runs |
| FiDIP config | 93.6 | 90.43 | Our training runs |
| Paper-settings config | 93.6 | 91.81 | Our training runs |
| Best training change: every layer trained, cleaned data, LR 1e-4 | 93.6 | 91.88 | Our training runs |

- Our evaluations come from the baseline job: `results/baseline/results/`.
- Our training runs are in `results/training/summary_by_config.csv`.
- Each results folder holds a `COMMIT` file: the version of the development code that made it. The development repository is private, so these versions do not resolve here.

## Runs

Every job ran on one NVIDIA H200 of the Explorer cluster.

| Run | Jobs | Results |
| --- | --- | --- |
| Baseline: environment check, released models, demo | 11633082 | `results/baseline/` |
| Guide config, runs 1 to 3 | 11601046, 11601199, 11601334 | `results/training/hrnet_w48_384x288_infant_baseline_rep<n>/` |
| FiDIP config, runs 1 to 3 | 11582243, 11598032, 11598631 | `results/training/hrnet_w48_384x288_infant_fidip_rep<n>/` |
| Paper-settings config, runs 1 to 3 | 11634837, 11634838, 11634839 | `results/training/hrnet_w48_384x288_infant_fidip_paper_rep<n>/` |
| Training change: cleaned data | 11601486, 11601730, 11601852 | `results/training/hrnet_w48_384x288_infant_fidip_clean_rep<n>/` |
| Training change: LR 1e-4 | 11601896, 11602035, 11602819 | `results/training/hrnet_w48_384x288_infant_fidip_lr1e-4_rep<n>/` |
| Training change: both, seeds 0 to 2 | 11607297, 11607298, 11607305 | `results/training/hrnet_w48_384x288_infant_fidip_lr1e-4_heldout_rep<n>/` |
| Training change: both, every layer trained | 11610307, 11610309, 11610311 | `results/training/hrnet_w48_384x288_infant_fidip_lr1e-4_heldout_freeze0_rep<n>/` |
| Training change: both, stage 3 frozen too | 11615390 (11610511 failed), 11610960, 11611339 | `results/training/hrnet_w48_384x288_infant_fidip_lr1e-4_heldout_freeze484_rep<n>/` |
| Training change: both, `LAMBDA` 0.005 | 11607848, 11607913, 11608820 | `results/training/hrnet_w48_384x288_infant_fidip_lr1e-4_lambda5e-3_heldout_rep<n>/` |
| Training change: both, `LAMBDA` 0.05 | 11608963, 11609760, 11610029 | `results/training/hrnet_w48_384x288_infant_fidip_lr1e-4_lambda5e-2_heldout_rep<n>/` |

## Reproduce the numbers

`SKILL.md` gives the setup: the environment, the code push, and the data.

```bash
# on the cluster, from the project folder; repeat the training lines with TAG=rep2 and TAG=rep3
sbatch code/current/jobs/reproduce_baseline.sbatch     # environment check, released models, demo
sbatch -p gpu-short --gres=gpu:h200:1 --time=01:00:00 --export=ALL,CFG=configs/hrnet_w48_384x288_infant_baseline.yaml,TAG=rep1 code/current/jobs/train_model.sbatch
sbatch -p gpu-short --gres=gpu:h200:1 --time=01:00:00 --export=ALL,CFG=configs/hrnet_w48_384x288_infant_fidip.yaml,TAG=rep1 code/current/jobs/train_model.sbatch
sbatch -p gpu-short --gres=gpu:h200:1 --time=01:15:00 --export=ALL,CFG=configs/hrnet_w48_384x288_infant_fidip_paper.yaml,TAG=rep1 code/current/jobs/train_model.sbatch

# locally, from the repository root
bash scripts/pull_results.sh raw/smalldata-project_baseline_run_<job id> results/baseline
bash scripts/pull_results.sh raw/train_<config>_rep<n> results/training/<config>_rep<n>   # each training run
python scripts/summarize_training.py                    # results/training/summary.csv
python scripts/extract_loss_curves.py                   # needs the loss logs, see below
python scripts/make_report_figures.py                   # report figures
```

`SKILL.md` gives the commands of the training changes.

`scripts/extract_loss_curves.py` reads `losses_P.npy`, `losses_D.npy`, and the TensorBoard file of each training run. Copy them from the cluster over the transfer host into the local data root:

```bash
ssh -o HostName=xfer.discovery.neu.edu explorer 'cd /scratch/fritsche.o/smalldata-project/raw && tar -c train_*_rep*/train/*.npy train_*_rep*/train/upstream_log' \
  | tar -x -C ../_data/smalldata-project/raw
```

## Data

All splits sit under `<root>/data/syrip/`.

| Split | Images | Content |
| --- | ---: | --- |
| `train_pre_infant` | 4057 | Adults: 2029 real COCO, 2028 synthetic SURREAL |
| `train_infant` | 1200 | Infants: 1000 synthetic, 200 real |
| `validate_infant` (Test100) | 100 | Real infants |
| Cleaned data, `<root>/variants/clean/` | 1154 | `train_infant` without the 46 mislabeled images. `scripts/make_clean_split.py` makes it |
| Selection split, `<root>/variants/select/` | 398 | The real Test500 images outside Test100. `scripts/make_select_split.py` makes it |

- Put both archives in `<root>/data/_zips/`. Unpack `syrip_for_train.zip` into `<root>/data/`: it gives `data/syrip/`.
- Copy `SyRIP/images/validate_infant/test11.jpg` from `SyRIP.zip` over `data/syrip/images/validate_infant/test11.jpg`.
- For the selection split, unpack `SyRIP.zip` and rename its folder `SyRIP/` to `data/syrip_original/`.
- The source is `https://coe.northeastern.edu/Research/AClab/SyRIP/`: `syrip_for_train.zip` (2024, md5 `06c2d03e09c5561c4f598227c61c4fcc`) and `SyRIP.zip` (2023, md5 `70d2bf2f160e1f04d01e3b61885434c2`).
- We use the 2024 archive. In both archives, Test100 has the same images and labels, except `test11.jpg`.
- In the 2024 archive, `test11.jpg` is a different photo whose label belongs to the 2023 photo. We use the 2023 file, md5 `0161fbedd7ce17b4ebfdae1e10c30074`.
- 46 synthetic `train_infant` images have labels for a 640 by 480 frame. The authors trained on them, so we keep them. `results/data_check/image_size_check.csv` lists them.
- The `train_infant` folder holds 300 more real files that the annotation file does not list. We do not use them.

## Weights

| File under `<root>/data/models/` | md5 | Role | Source |
| --- | --- | --- | --- |
| `hrnet_fidip.pth` | `c1b4994799f08e6cbc44942f44472790` | Released HRNet-W48 checkpoint | Upstream README, "FiDIP_models" folder |
| `mobile_fidip.pth` | `049dbf22c3131bb72872bc744a1eefe5` | Released MobileNetV2 checkpoint (Pose-MobileNet) | Upstream README, "FiDIP_models" folder |
| `coco/w48_384x288.pth` | `403962abcb38f9a85891c1a7a029dcdc` | COCO DarkPose start weights | Upstream README, "TGA_models" folder |
| `coco/posemobile.pth` | `f3a3f265d26dd8fdc1d57897809e7ed8` | COCO Pose-MobileNet start weights. `scripts/prepare_data.py` needs it | Upstream README, "FiDIP_models" folder |

- "FiDIP_models": `https://drive.google.com/drive/folders/108P-1SnTqaj3xNtjYZ1o7T8z6UvUYuiC`.
- "TGA_models": `https://drive.google.com/drive/folders/14kAA1zXuKODYgrRiQmKnVcipbY7RedVV`.
- Google Drive refuses scripted downloads of these folders. Download them in a browser.

## Trained checkpoints

The checkpoints of the nine training runs stay on the cluster:

```text
<root>/raw/train_<config>_rep<n>/train/upstream_output/syrip/adaptive_pose_hrnet/<config>/
    final_state.pth    the last epoch: the reported number
    model_best.pth     the best epoch, chosen on Test100: for comparison only
    checkpoint.pth     the resume point
```

`<config>` is `hrnet_w48_384x288_infant_baseline` (the guide config), `hrnet_w48_384x288_infant_fidip` (the FiDIP config), or `hrnet_w48_384x288_infant_fidip_paper` (the paper-settings config). `<n>` is 1, 2, or 3.

The training changes keep `final_state.pth` in `<root>/raw/train_<config>_<tag>/train/`. The runs of `scripts/train_with_validation.py` use `train_<config>_heldout_<tag>`.

## Configs

| File in `configs/` | Use |
| --- | --- |
| `hrnet_w48_384x288_infant_baseline.yaml` | The guide config, and the test config. Upstream `experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml`, unchanged |
| `hrnet_w48_384x288_infant_fidip.yaml` | The FiDIP config: the guide config with five training values from upstream `experiments/coco/hrnet/w48_384x288_adam_lr1e-3_custom.yaml` |
| `hrnet_w48_384x288_infant_fidip_paper.yaml` | The paper-settings config: the FiDIP config with the paper's 100 epochs at batch 64 and 1-epoch warm-up at batch 128 |
| `hrnet_w48_384x288_infant_fidip_lr1e-4.yaml` | The FiDIP config at LR 1e-4 |
| `hrnet_w48_384x288_infant_fidip_lr1e-4_lambda5e-3.yaml`, `..._lambda5e-2.yaml` | LR 1e-4 with `LAMBDA` 0.005 and 0.05 |
| `mobile_224x224_infant_baseline.yaml` | Test of the released MobileNetV2 checkpoint. Upstream verbatim |
| `paths.yaml` | Roots, paths, and expected counts |

## Code map

| Path | Purpose |
| --- | --- |
| `scripts/check_env.py` | Check the imports, one forward pass, and the paths |
| `scripts/prepare_data.py` | Check the counts, the md5 sums, and the image sizes |
| `scripts/test_model.py` | Evaluate one checkpoint with the upstream test script |
| `scripts/train_model.py` | Train with the upstream trainer, then evaluate the final and best checkpoints |
| `scripts/summarize_training.py` | Write `results/training/summary.csv` |
| `scripts/extract_loss_curves.py` | Write the per-epoch training and Test100 losses of each run |
| `scripts/demo.py` | Predict the pose on one image |
| `scripts/make_clean_split.py`, `scripts/make_select_split.py` | Make the cleaned data and the selection split |
| `scripts/train_with_validation.py` | Train with a seed and a freeze depth, with checkpoint selection on the selection split |
| `scripts/compare_runs.py` | Compare two training configs image by image |
| `scripts/find_near_duplicates.py` | Find training images that copy or nearly copy a Test100 image |
| `scripts/make_report_figures.py` | Make the report figures |
| `scripts/push_code.sh`, `push_data.sh`, `push_upstream.sh`, `pull_results.sh` | Move code, data, and results to and from the cluster |
| `src/smalldata/` | Shared code: evaluation, OKS, and paths |
| `jobs/*.sbatch` | Slurm jobs. `SKILL.md` lists them |

## Environments

| Name | Location | Content |
| --- | --- | --- |
| `smalldata-project` | Cluster `<root>/env/` | Python 3.12, torch 2.3.1+cu121, `requirements.txt` |
| `tectonic` | Local | Tectonic 0.17.0, the LaTeX engine for the report |

## Result folders

| Folder | Content |
| --- | --- |
| `results/baseline/` | Environment check, the released models, and the demo, from the baseline job |
| `results/data_check/` | Counts, md5 sums, and the image-size check |
| `results/training/` | Per-run metrics, per-epoch AP and losses, `summary.csv`, and the paired comparisons |
| `results/analysis/leakage_review_test100.csv` | The Test100 overlap review: per Test100 image, a verdict and the related training image |
| `results/figures/` | Report figures and `per_image_oks.csv` |
| `report/` | `report.tex`, `references.bib`, and `report.pdf` |

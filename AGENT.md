# AGENT: lab notebook for smalldata-project

This file logs what the owner asked, what the agent did, and why the plan changed. The newest entry is last.

This submission covers guide parts 0 to 6, and five training changes of part 7.

## Rules

- AP is COCO OKS AP on Test100. It uses ground-truth boxes, the flip test, and DARK decoding, as upstream does.
- The number of a training run is its final-epoch AP. Upstream picks `model_best.pth` by Test100 AP, so that AP is for comparison only.
- Each training config gets three runs, because the upstream trainer sets no seed.
- No choice uses Test100. One exception: later training changes kept the settings that won on Test100.
- Every reported number comes from one NVIDIA H200 of the Explorer cluster.
- The upstream code stays unmodified. Our code lives in `src/`, `scripts/`, and `jobs/`.

## Status

| Guide part | Result |
| --- | --- |
| 0 Agentic workflow | This log, `SKILL.md`, and `MANIFEST.md` |
| 1 Environment | The environment check passes |
| 2 Data | The counts match. `MANIFEST.md` lists the md5 sums |
| 3 Model | Notes in the log below |
| 4 Training | Our runs: guide config 91.50 AP (paper 92.7), FiDIP config 90.43 AP (paper 93.6), paper-settings config 91.81 AP (paper 93.6) |
| 5 Evaluation | Our evaluation of the released checkpoint: 93.60 AP (paper 93.6) |
| 6 Demo | Test image 27 runs end to end |
| 7 Improvement attempts | Five training changes. The best gives 91.88 AP (paper 93.6) |

## Log

### 2026-11-02: environment and data

- Asked: set up the project, the environment, and the data.
- Built the conda environment on the cluster. The environment check passes.
- Pinned the upstream code at `a76b4f4`.
- Downloaded SyRIP and four weight files. `scripts/prepare_data.py` checks the counts and the image sizes, and records the md5 sums.

### 2026-11-02: model notes

- Asked: understand the model before any training (guide part 3).
- The pose network is HRNet-W48 with 63.6 M parameters. It gives 17 heatmaps of 96 by 72.
- The domain classifier has 0.75 M parameters and reads the last 48-channel feature map.
- Upstream freezes the first 109 parameter tensors, up to HRNet stage 2. It uses Adam and sets no seed.
- The guide config file drops the learning rate tenfold at epoch 40. The 20-epoch configs never reach that drop.
- Upstream evaluates Test100 after each epoch and saves the best epoch as `model_best.pth`.
- Finding: the guide's training config turns FiDIP off. It has no warm-up and `LAMBDA` 0, and despite its name it sets LR 1e-4. It trains plain fine-tuning, the paper's 92.7 row.
- Finding: the FiDIP settings sit in an upstream config for another dataset.
- Dead end: an empty `MODEL.PRETRAINED` stops the upstream trainer with an error. Our trainer loads the start weights with `TRAIN.CHECKPOINT` instead.
- Decision: train two configs from the COCO weights, three runs each.
  - The guide config. Published: 92.7 AP.
  - The FiDIP config: the guide config with the five FiDIP values. These are LR 1e-3, 10 warm-up epochs, `LAMBDA` 0.0005, and two batch sizes. Published: 93.6 AP.

### 2026-11-03: the released checkpoint reproduces after a data fix

- Asked: evaluate the released checkpoint before any training (guide part 5).
- First result: 92.15 AP, 1.45 below the published 93.6.
- Hypotheses tested:
  - Wrong test settings: excluded. The settings match the upstream config.
  - Library or GPU differences: excluded. A second GPU gives the same per-image OKS to four decimals.
  - Changed annotations: excluded. The 2023 and 2024 annotation files hold the same labels and image sizes.
  - A bad test image: confirmed. `test11.jpg` scores OKS 0.09. In the 2024 archive it is a different photo, at 2048 by 2048: another scene, other clothes, another pose. Its label belongs to the 1536 by 2048 photo of the 2023 archive.
- In both archives, Test100 has the same 100 images and labels, except `test11.jpg`.
- The test crops a region around the labeled person box, padded to 3:4 and 1.25 times larger (`lib/dataset/syrip.py`). On the 2024 file, that region holds a different scene.
- With the 2023 file, the released checkpoint scores 93.61 AP, 98.52 AP50, 98.52 AP75, and 94.60 AR. These equal the published numbers.
- The released MobileNetV2 checkpoint scores 79.16 AP against the published 79.2.
- A size scan found 46 synthetic training images with labels for a 640 by 480 frame. The authors trained on them, so the reproduction keeps them.
- Dead end: before the `test11.jpg` fix, one idea was a second reference model. Upstream also releases a model trained on the expanded SyRIP. Google Drive refused the scripted download. The fix made the check unnecessary: that model uses other data, and the README gives no number for it.
- Decision: use the 2024 archive with the 2023 `test11.jpg`.

### 2026-11-03: the training falls short

- Asked: reproduce the training (guide part 4).
- Ran smoke tests first, then six training jobs, 6 to 12 minutes each.

| Config | Final-epoch AP, mean (sd) | Best epoch, chosen on Test100 | Published |
| --- | ---: | ---: | ---: |
| Guide | 91.50 (0.57) | 92.02 | 92.7 |
| FiDIP | 90.43 (1.25) | 92.07 | 93.6 |

- The guide config falls 1.2 AP short, and the FiDIP config falls 3.2 AP short.
- With test selection, the two configs tie. The paper's gain of FiDIP over fine-tuning does not appear.
- At LR 1e-3, the AP of the FiDIP config swings by up to 2 AP between epochs.
- The upstream files do not say what setup produced the released checkpoint.

### 2026-11-03: demo

- Asked: run the single-image demo (guide part 6).
- The guide's `tools/demo.py` and Streamlit demo are missing in the upstream code, also at its latest commit. `scripts/demo.py` replaces them.
- The demo runs test image 27 end to end. With the annotation box, the keypoints match the evaluation within 0.01 pixels.

### 2026-11-03 to 2026-11-04: training changes

- Asked: try to beat the published baseline (guide part 7). We registered each change here before its first run.
- Control: the FiDIP config, 90.43 AP. Each change gets three runs. Its number is the final-epoch AP.
- New `scripts/make_clean_split.py` drops the 46 mislabeled images.
- New `scripts/train_with_validation.py` takes a seed and a freeze depth. It picks a checkpoint on a selection split: the 398 real Test500 images outside Test100 (`scripts/make_select_split.py`).

| Change | AP | SD | Result |
| --- | ---: | ---: | --- |
| Drop the 46 mislabeled images | 91.60 | 0.41 | Better than the FiDIP config, p = 0.043 |
| LR 1e-4 | 91.74 | 0.25 | Better than the FiDIP config, p = 0.007 |
| Both changes | 91.63 | 0.15 | Same as the first change alone, p = 0.87 |
| Both, checkpoint chosen on the selection split | 90.63 | 0.43 | 1.0 AP below the final epoch |
| Both, every layer trained, checkpoint chosen on the selection split | 92.30 | 0.21 | 0.4 AP above the final epoch |
| Both, stage 3 frozen too, checkpoint chosen on the selection split | 91.12 | 0.49 | 0.2 AP above the final epoch |
| Both, every layer trained | 91.88 | 0.64 | Same as both, p = 0.55 |
| Both, stage 3 frozen too | 90.92 | 0.64 | Worse than both, p = 0.023 |
| Both, `LAMBDA` 0.005 | 58.55 | 5.51 | Worse than both, by 33 AP |
| Both, `LAMBDA` 0.05 | 0.48 | 0.50 | Worse than both, by 91 AP |

- p: exact sign test on the per-image OKS (`scripts/compare_runs.py`).
- Each of the first two changes gains 1.2 to 1.3 AP and removes most of the run-to-run noise. Together, they gain nothing more.
- With a larger `LAMBDA`, the selection split picks epoch 1 in all six runs: Test100 AP 83.0 to 85.3. After epoch 1, the AP falls.
- Checkpoint choice on the selection split gives mixed results: 1.0 AP lower, then 0.4 and 0.2 AP higher.
- The selection split is too easy to rank checkpoints. The released checkpoint scores 98.91 AP on it.
- The selection split shares a video or an infant with 32 Test100 images (`results/analysis/leakage_review_test100.csv`). So the final epoch stays the number of record.
- No change reaches 93.6. The best gives 91.88 AP.
- Decision: later changes kept the settings that won on Test100. So these numbers are not free of test information.
- Every run used one H200, for 6 to 12 minutes.
- Job 11610511, stage 3 frozen, run 1, failed: the cluster data folder was empty at its start. After a restore of the data, job 11615390 ran it again.

### 2026-11-04: failure analysis and data overlap

- Asked: analyze the successes and failures.
- The released checkpoint has a mean OKS of 0.953. 91 of 100 images score above 0.9.
- Weakest joints: ears, hips, and wrists. Hardest posture: prone.
- On the hardest images, the released checkpoint and our best run miss the same joints.
- `scripts/find_near_duplicates.py` and a review by eye compared each Test100 image with the training images.
- 56 Test100 images share a copy, a video, or an infant with them. `results/analysis/leakage_review_test100.csv` holds the verdicts.
- The owner set the scope: no new split. The overlap is a limitation of the benchmark.

### 2026-11-04: every number on one GPU type

- Asked: run every GPU step on one GPU type.
- The training runs ran on H200 GPUs. The evaluations, the demo, and the environment check had run on other GPUs.
- New job `jobs/reproduce_baseline.sbatch` reruns these on one H200:
  - the environment check;
  - the released checkpoint with each `test11.jpg`;
  - the MobileNetV2 checkpoint and the demo.
- The job builds a second data root that differs from the main root in the 2024 `test11.jpg` only.

### 2026-11-04: training and Test100 loss curves

- Asked: show the training and validation loss of each epoch, as guide part 4.5 asks. Never choose an epoch on the test set.
- Upstream uses Test100 as its validation set. So these curves only diagnose training. The reported number stays the final epoch.
- New `scripts/extract_loss_curves.py` writes the training heatmap loss and the Test100 loss of each epoch.
- The FiDIP config logs the heatmap loss minus `LAMBDA` times the classifier loss. The script adds that term back.
- The guide config does not overfit: its Test100 loss falls until about epoch 10 and then stays flat.
- The FiDIP config overfits: after epoch 12 its mean Test100 loss rises from about 3.9e-4 to 4.5e-4. Its training loss still falls.

### 2026-11-05: the paper's training settings differ from the upstream configs

- Asked: why our training stays below the published numbers, and whether the paper states the settings.
- The paper (Section V-B) trains for 100 epochs with batch 64. It warms the classifier up for 1 epoch with batch 128, and uses LR 1e-3 and `LAMBDA` 0.0005.
- The upstream configs train for 20 epochs with batches of 18 to 20. The FiDIP config warms up for 10 epochs with batch 64.
- The paper does not say how it chose the reported epoch, or which HRNet layers it freezes.
- This difference is the likeliest cause of the gap.

### 2026-11-05: training with the paper's settings, registered before the runs

- Asked: train with the paper's settings, the one untested part of the training reproduction.
- New config `configs/hrnet_w48_384x288_infant_fidip_paper.yaml`: the FiDIP config with 100 epochs at batch 64 and a 1-epoch warm-up at batch 128.
- The paper names no learning-rate schedule. The upstream one stays: the rate drops tenfold at epoch 40.
- A local smoke test (1 epoch, 16 images) passed. Its numbers are not results.
- Three runs on H200 GPUs. The number of record is the final-epoch AP, mean of the three runs, against the paper's 93.6.

### 2026-11-05: the baseline job on the H200

- The baseline job ran on one H200 in under 3 minutes. Its results are in `results/baseline/`.
- Our evaluation of the released HRNet checkpoint: 93.60 AP, 98.52 AP50, 98.52 AP75, 94.60 AR. The paper gives 93.6, 98.5, 98.5, 94.6.
- With the 2024 `test11.jpg`: 92.15 AP, the same as before.
- The released MobileNetV2 checkpoint: 79.18 AP. The paper gives 79.2.
- The earlier runs on other GPUs gave 93.61 and 79.16. GPU types differ in the last decimal. The H200 numbers are the numbers of record.
- The environment check and the demo pass. The per-image OKS of the released checkpoint is the same to four decimals, so the analysis numbers stay.

### 2026-11-05: the paper's settings close part of the gap

- The three runs waited 13 hours with a 2 h limit. The scheduler fits short jobs into gaps, so the limit went down to 1 h 15 min. The runs then started within an hour.
- Each run took 24 to 28 minutes on one H200.
- Final-epoch AP: 92.21, 92.01, and 91.21. Mean 91.81 (sd 0.53), against the paper's 93.6.
- The best epoch, chosen on Test100, gives 92.69 on average. It is for comparison only.
- The paper's settings raise the FiDIP config by 1.4 AP. A 1.8 AP gap stays open.
- The paper does not state its epoch choice, its frozen layers, or its learning-rate schedule. These unknowns can explain the remaining gap.
- The paper-settings config does not overfit. Its Test100 loss falls until about epoch 80, then stays flat. The learning-rate drop at epoch 40 steadies the loss and the AP.
- Our best reproduction run is now paper-settings run 1 (92.21), the best of the three configs. On the worst images, it misses the same joints as the released checkpoint.
- The two models differ most on test29, a prone infant: 0.91 OKS for the released checkpoint, 0.83 for our run.
- The longer runs write one TensorBoard subfolder per AP metric. `scripts/extract_loss_curves.py` now reads the one file that holds `valid_loss`.

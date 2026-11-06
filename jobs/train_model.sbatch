#!/bin/bash
#SBATCH --job-name=smalldata-project_train_run
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out

# Part 4: train with the upstream trainer from the COCO DarkPose weights, then evaluate.
# Submit from the project root on the cluster, so logs/ and raw/ resolve:
#
#   cd /scratch/$USER/smalldata-project
#   sbatch -J smalldata-project_train_run code/current/jobs/train_model.sbatch
#
# Variables through --export=ALL,...:
#   CFG=configs/hrnet_w48_384x288_infant_fidip.yaml   (default; the FiDIP config)
#   CFG=configs/hrnet_w48_384x288_infant_baseline.yaml (the guide config)
#   CFG=configs/hrnet_w48_384x288_infant_fidip_paper.yaml (the paper-settings config; --time=01:15:00)
#   TAG=rep1          suffix of the result and run folders, one per repeat
#   BASE=<folder>     data root for the run, relative to the project folder (default: the project folder)
#   EPOCHS=1 PRE_EPOCHS=1 MAX_SAMPLES=16   smoke test; submit with -J smalldata-project_train_smoke -p sharing --time=00:30:00
#
# The run folder is raw/train_<config>_<TAG>, not raw/<job>_<id>, so a resubmission with the same CFG and TAG
# resumes from the upstream checkpoint.pth and the driver skips a finished run. Chain links for a short
# partition with --dependency=afterany:<previous job id>. The JOBS file in the run folder lists every job.
#
# Check a new header first with `sbatch --test-only`. Do not add --qos.
set -euo pipefail

ROOT="$SLURM_SUBMIT_DIR"
[ -d "$ROOT/code/current" ] || { echo "submit from the project root" >&2; exit 2; }
# Resolve the link once. A later push must not change this job.
CODE="$(readlink -f "$ROOT/code/current")"
COMMIT="$(cat "$CODE/COMMIT")"

export TMPDIR="$ROOT/tmp"
export PYTHONNOUSERSITE=1
export SMALLDATA_ROOT="$ROOT"
export FIDIP_UPSTREAM="$ROOT/upstream/Infant-Pose-Estimation"

module load anaconda3/2024.06
eval "$(conda shell.bash hook)"
# The conda shell hook reads unbound variables, so relax -u around the activation only.
set +u
conda activate "$ROOT/env/smalldata-project"
set -u

CFG="${CFG:-configs/hrnet_w48_384x288_infant_fidip.yaml}"
TAG="${TAG:-rep1}"
BASE="$ROOT/${BASE:-.}"
[ -d "$BASE/data/syrip" ] || { echo "no data root at $BASE" >&2; exit 2; }
EPOCHS="${EPOCHS:-0}"
PRE_EPOCHS="${PRE_EPOCHS:--1}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
NAME="$(basename "$CFG" .yaml)_${TAG}"

RUN="$ROOT/raw/train_${NAME}"
[ "$MAX_SAMPLES" != 0 ] && RUN="$ROOT/raw/train_${NAME}_smoke"
mkdir -p "$RUN"
[ -f "$RUN/COMMIT" ] || echo "$COMMIT" > "$RUN/COMMIT"
echo "$SLURM_JOB_ID $(date -u +%Y-%m-%dT%H:%M:%SZ) $COMMIT $(hostname)" >> "$RUN/JOBS"
echo "job=$SLURM_JOB_ID node=$(hostname) commit=$COMMIT run=$RUN cfg=$CFG tag=$TAG"
nvidia-smi -L

OUT="$RUN/results"

cd "$CODE"
# Node-local temp for Python's multiprocessing folders; the networked tmp/ refuses their removal at exit.
TMPDIR=/tmp python scripts/train_model.py --cfg "$CFG" --model "$ROOT/data/models/coco/w48_384x288.pth" --data validate_infant \
  --base-path "$BASE" --out "$OUT" --work "$RUN/train" --workers 8 \
  --epochs "$EPOCHS" --pre-epochs "$PRE_EPOCHS" --max-samples "$MAX_SAMPLES"

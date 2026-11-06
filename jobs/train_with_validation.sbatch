#!/bin/bash
#SBATCH --job-name=smalldata-project_trainval_run
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out

# Train with a seed and a freeze depth, with checkpoint selection on the selection split (scripts/train_with_validation.py),
# then evaluate final_state.pth and model_best.pth once each on the test split.
# Submit from the project root on the cluster, so logs/ and raw/ resolve:
#
#   cd /scratch/$USER/smalldata-project
#   sbatch -p gpu-short --gres=gpu:h200:1 --time=01:00:00 \
#     --export=ALL,CFG=configs/hrnet_w48_384x288_infant_fidip_lr1e-4.yaml,TAG=rep1,BASE=variants/clean,SEED=0 \
#     -J smalldata-project_trainval_run code/current/jobs/train_with_validation.sbatch
#
# Variables through --export=ALL,...:
#   CFG=configs/<config>.yaml   (default: the FiDIP config)
#   TAG=rep1                    suffix of the result and run folders, one per repeat
#   BASE=variants/clean         data root for training, relative to the project folder (default: the project folder)
#   SELECT=variants/select      root of the selection split (default; jobs/make_select_split.sbatch builds it)
#   MODEL=data/models/coco/w48_384x288.pth   start weights, relative to the project folder (default: the COCO weights)
#   SEED=0  FREEZE=109  DETERMINISTIC=0
#   EPOCHS=1 PRE_EPOCHS=1 MAX_SAMPLES=16 BATCH_SIZE=4   smoke test; submit with -J smalldata-project_trainval_smoke -p sharing --time=00:30:00
#
# The run folder is raw/train_<config>_heldout_<TAG>, so a resubmission with the same CFG and TAG resumes
# from its checkpoint.pth and the trainer skips a finished run. The JOBS file in the run folder lists every job.
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
SELECT_DIR="$ROOT/${SELECT:-variants/select}"
[ -d "$BASE/data/syrip" ] || { echo "no data root at $BASE" >&2; exit 2; }
[ -d "$SELECT_DIR/data/syrip" ] || { echo "no selection split at $SELECT_DIR" >&2; exit 2; }
MODEL_FILE="$ROOT/${MODEL:-data/models/coco/w48_384x288.pth}"
[ -f "$MODEL_FILE" ] || { echo "no start weights at $MODEL_FILE" >&2; exit 2; }
SEED="${SEED:-0}"
FREEZE="${FREEZE:-109}"
EPOCHS="${EPOCHS:-0}"
PRE_EPOCHS="${PRE_EPOCHS:--1}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
BATCH_SIZE="${BATCH_SIZE:-0}"
EXTRA=()
[ "${DETERMINISTIC:-0}" != 0 ] && EXTRA+=(--deterministic)
[ "$BATCH_SIZE" != 0 ] && EXTRA+=(--batch-size "$BATCH_SIZE")
NAME="$(basename "$CFG" .yaml)_heldout_${TAG}"

RUN="$ROOT/raw/train_${NAME}"
[ "$MAX_SAMPLES" != 0 ] && RUN="$ROOT/raw/train_${NAME}_smoke"
mkdir -p "$RUN"
[ -f "$RUN/COMMIT" ] || echo "$COMMIT" > "$RUN/COMMIT"
echo "$SLURM_JOB_ID $(date -u +%Y-%m-%dT%H:%M:%SZ) $COMMIT $(hostname)" >> "$RUN/JOBS"
echo "job=$SLURM_JOB_ID node=$(hostname) commit=$COMMIT run=$RUN cfg=$CFG tag=$TAG seed=$SEED freeze=$FREEZE select=$SELECT_DIR start=$MODEL_FILE"
nvidia-smi -L

OUT="$RUN/results"

cd "$CODE"
# Node-local temp for Python's multiprocessing folders; the networked tmp/ refuses their removal at exit.
TMPDIR=/tmp python scripts/train_with_validation.py --cfg "$CFG" --model "$MODEL_FILE" \
  --data validate_infant --base-path "$BASE" --select-root "$SELECT_DIR" --out "$OUT" --work "$RUN/train" --workers 8 \
  --seed "$SEED" --freeze-tensors "$FREEZE" --epochs "$EPOCHS" --pre-epochs "$PRE_EPOCHS" --max-samples "$MAX_SAMPLES" \
  ${EXTRA[@]+"${EXTRA[@]}"}

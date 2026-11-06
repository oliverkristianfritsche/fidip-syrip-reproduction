#!/bin/bash
#SBATCH --job-name=smalldata-project_clean_run
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out

# Build variants/clean, the training variant without the 46 synthetic images whose labels sit in
# another frame. Submit from the project root on the cluster, so logs/ and raw/ resolve:
#
#   cd /scratch/$USER/smalldata-project
#   sbatch -J smalldata-project_clean_run code/current/jobs/make_clean_split.sbatch
#
# A batch job, because the login node kills the builder (it opens 1200 image headers). Train on the
# variant with --export=ALL,BASE=variants/clean,TAG=clean_rep<n> in jobs/train_model.sbatch.
set -euo pipefail

ROOT="$SLURM_SUBMIT_DIR"
[ -d "$ROOT/code/current" ] || { echo "submit from the project root" >&2; exit 2; }
# Resolve the link once. A later push must not change this job.
CODE="$(readlink -f "$ROOT/code/current")"
COMMIT="$(cat "$CODE/COMMIT")"

export PYTHONNOUSERSITE=1
export SMALLDATA_ROOT="$ROOT"
export FIDIP_UPSTREAM="$ROOT/upstream/Infant-Pose-Estimation"

RUN="$ROOT/raw/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$RUN"
echo "$COMMIT" > "$RUN/COMMIT"
echo "job=$SLURM_JOB_ID node=$(hostname) commit=$COMMIT run=$RUN"

cd "$CODE"
TMPDIR=/tmp "$ROOT/env/smalldata-project/bin/python" scripts/make_clean_split.py --out "$RUN/clean_split_dropped.csv"
ls -la "$ROOT/variants/clean/data/syrip/annotations" "$ROOT/variants/clean/data/syrip/images"

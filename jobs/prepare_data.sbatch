#!/bin/bash
#SBATCH --job-name=smalldata-project_data_run
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out

# Part 2 check on the cluster copy of the data: counts, every image file, weight md5, stored image
# sizes against the annotations. Submit from the project root on the cluster, so logs/ and raw/ resolve:
#
#   cd /scratch/$USER/smalldata-project
#   sbatch -J smalldata-project_data_run code/current/jobs/prepare_data.sbatch
#
# A batch job, because the login node kills the check (it opens 5357 images and hashes 540 MB)
# and the transfer host cannot run the environment's Python. No GPU needed.
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
TMPDIR=/tmp "$ROOT/env/smalldata-project/bin/python" scripts/prepare_data.py --out "$RUN"

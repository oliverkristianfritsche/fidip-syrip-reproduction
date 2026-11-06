#!/bin/bash
#SBATCH --job-name=smalldata-project_select_run
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out

# Build variants/select, the selection split for checkpoint selection: the real Test500 images
# outside Test100. Submit from the project root on the cluster, so logs/ and raw/ resolve:
#
#   cd /scratch/$USER/smalldata-project
#   sbatch -J smalldata-project_select_run code/current/jobs/make_select_split.sbatch
#
# A batch job, because the login node kills work of this size (md5 of 600 images, 500 image headers).
# Train with selection on the variant through jobs/train_with_validation.sbatch.
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
TMPDIR=/tmp "$ROOT/env/smalldata-project/bin/python" scripts/make_select_split.py --out "$RUN/select_split.csv"
ls -la "$ROOT/variants/select/data/syrip/annotations"
ls "$ROOT/variants/select/data/syrip/images/validate_infant" | wc -l

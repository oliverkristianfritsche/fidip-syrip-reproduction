#!/bin/bash
#SBATCH --job-name=smalldata-project_baseline_run
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.out

# Reproduce every baseline number on one H200, the GPU type of the training runs:
#   1. the environment check;
#   2. the released HRNet checkpoint on Test100 with the corrected test11.jpg (2023 archive);
#   3. the same checkpoint with the 2024 archive's test11.jpg, from a second data root that differs in that file only;
#   4. the released mobile model on Test100;
#   5. the single-image demo on test image 27 with its annotation box.
# Submit from the project root on the cluster, so logs/ and raw/ resolve:
#
#   cd /scratch/$USER/smalldata-project
#   sbatch code/current/jobs/reproduce_baseline.sbatch
#
# Pull the results:  bash scripts/pull_results.sh raw/smalldata-project_baseline_run_<job id> results/baseline
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

RUN="$ROOT/raw/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
OUT="$RUN/results"
mkdir -p "$OUT"
echo "$COMMIT" > "$RUN/COMMIT"
echo "job=$SLURM_JOB_ID node=$(hostname) commit=$COMMIT run=$RUN"
nvidia-smi -L | tee "$OUT/gpu.txt"

TEST11="$ROOT/data/syrip/images/validate_infant/test11.jpg"
[ "$(md5sum < "$TEST11" | cut -d' ' -f1)" = 0161fbedd7ce17b4ebfdae1e10c30074 ] || { echo "test11.jpg is not the 2023 file" >&2; exit 3; }

cd "$CODE"
# Node-local temp for Python's multiprocessing folders; the networked tmp/ refuses their removal at exit.
echo "== environment check"
TMPDIR=/tmp python scripts/check_env.py --out "$OUT/environment"

echo "== released HRNet checkpoint, 2023 test11.jpg"
TMPDIR=/tmp python scripts/test_model.py --model hrnet_fidip --data validate_infant \
  --out "$OUT/hrnet_fidip" --work "$RUN/upstream_hrnet_fidip" --workers 4

echo "== released HRNet checkpoint, 2024 test11.jpg"
R24="$RUN/root_images2024"
mkdir -p "$R24/data/syrip/images/validate_infant"
ln -s "$ROOT/data/models" "$R24/data/models"
ln -s "$ROOT/data/syrip/annotations" "$R24/data/syrip/annotations"
for d in train_pre_infant train_infant; do ln -s "$ROOT/data/syrip/images/$d" "$R24/data/syrip/images/$d"; done
for f in "$ROOT"/data/syrip/images/validate_infant/*; do ln -s "$f" "$R24/data/syrip/images/validate_infant/"; done
rm "$R24/data/syrip/images/validate_infant/test11.jpg"
unzip -p "$ROOT/data/_zips/syrip_for_train.zip" syrip/images/validate_infant/test11.jpg \
  > "$R24/data/syrip/images/validate_infant/test11.jpg"
[ "$(md5sum < "$R24/data/syrip/images/validate_infant/test11.jpg" | cut -d' ' -f1)" = c642c6573896388d5c8cf1592d573958 ] \
  || { echo "the 2024 test11.jpg has the wrong md5" >&2; exit 3; }
TMPDIR=/tmp python scripts/test_model.py --model hrnet_fidip --data validate_infant --base-path "$R24" \
  --out "$OUT/hrnet_fidip_images2024" --work "$RUN/upstream_hrnet_fidip_images2024" --workers 4

echo "== released mobile model"
TMPDIR=/tmp python scripts/test_model.py --model mobile_fidip --cfg configs/mobile_224x224_infant_baseline.yaml \
  --data validate_infant --out "$OUT/mobile_fidip" --work "$RUN/upstream_mobile_fidip" --workers 4

echo "== demo, test image 27 with its annotation box"
TMPDIR=/tmp python scripts/demo.py --image "$ROOT/data/syrip/images/validate_infant/test27.jpg" \
  --annotation validate_infant --out "$OUT/demo/test27_hrnet_fidip_gtbox"

echo "DONE"

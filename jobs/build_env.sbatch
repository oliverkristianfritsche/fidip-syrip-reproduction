#!/bin/bash
#SBATCH --job-name=smalldata-project_build_run
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

# Build the conda environment for smalldata-project. Run it once, and again after a change
# to requirements.txt. Submit from the project root on the cluster, so logs/ resolves:
#
#   cd /scratch/$USER/smalldata-project
#   sbatch -J smalldata-project_build_run code/current/jobs/build_env.sbatch
#
# The job needs no GPU. A rerun updates the existing environment instead of recreating it.
# torch 2.3.1 with CUDA 12.1 matches third_party/Infant-Pose-Estimation/fidip_env.yml.
set -euo pipefail

ROOT="$SLURM_SUBMIT_DIR"
[ -d "$ROOT/code/current" ] || { echo "submit from the project root" >&2; exit 2; }
# Resolve the link once. A later push must not change this job.
CODE="$(readlink -f "$ROOT/code/current")"
COMMIT="$(cat "$CODE/COMMIT")"
ENV="$ROOT/env/smalldata-project"

export TMPDIR="$ROOT/tmp"
export PIP_CACHE_DIR="$ROOT/tmp/pip"
export PYTHONNOUSERSITE=1
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$ROOT/env"

module load anaconda3/2024.06
eval "$(conda shell.bash hook)"
echo "job=$SLURM_JOB_ID node=$(hostname) commit=$COMMIT env=$ENV"

if [ ! -x "$ENV/bin/python" ]; then
  conda create -y -p "$ENV" python=3.12
fi
# The conda shell hook reads unbound variables, so relax -u around the activation only.
set +u
conda activate "$ENV"
set -u

python -m pip install --upgrade pip
python -m pip install "torch==2.3.1" "torchvision==0.18.1" --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r "$CODE/requirements.txt"

python - <<'EOF'
import numpy, torch, torchvision
print("torch", torch.__version__, "torchvision", torchvision.__version__,
      "numpy", numpy.__version__, "cuda build", torch.version.cuda)
EOF
python -m pip freeze > "$ROOT/env/smalldata-project.freeze.txt"
echo "wrote $ROOT/env/smalldata-project.freeze.txt"

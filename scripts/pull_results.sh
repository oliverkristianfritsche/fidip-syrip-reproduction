#!/usr/bin/env bash
# Copy the small result files of one cluster run into the repository. Run from the project root.
#
#   bash scripts/pull_results.sh raw/train_hrnet_w48_384x288_infant_fidip_rep1 results/training/hrnet_w48_384x288_infant_fidip_rep1
#   bash scripts/pull_results.sh raw/smalldata-project_baseline_run_<job id> results/baseline
#
# Only the results/ subtree of the run comes down: metrics.csv, metrics.json, epochs.csv,
# summary.json, predictions_*.json, and the COMMIT and JOBS files of the run. Checkpoints, the
# upstream feature dumps, and logs stay on the cluster. A run folder holds hundreds of megabytes.
#
#   HOST=alias     overrides the ssh host (default: explorer)
#   PROJECT=name   overrides the project folder (default: name of the repository folder)
set -euo pipefail

HOST="${HOST:-explorer}"
TOP="$(git rev-parse --show-toplevel)"
PROJECT="${PROJECT:-$(basename "$TOP")}"
[ $# -eq 2 ] || { echo "usage: $0 <run folder under the cluster project root> <local folder>" >&2; exit 2; }
REMOTE="$1"
DEST="$TOP/$2"
case "$REMOTE" in
  ..*|/*) echo "the run folder is relative to the cluster project root" >&2; exit 2 ;;
esac
mkdir -p "$DEST"

ssh "$HOST" "cd /scratch/\$USER/$PROJECT/$REMOTE && tar -c \
  --exclude='*.pth' --exclude='*.npy' --exclude='upstream_output' --exclude='upstream_log' --exclude='subset_root' \
  \$(ls -d results results_smoke COMMIT JOBS 2>/dev/null)" |
  tar -x --no-same-owner -C "$DEST"
echo "pulled $REMOTE -> $DEST"
find "$DEST" -type f | sed "s#^$TOP/##" | sort

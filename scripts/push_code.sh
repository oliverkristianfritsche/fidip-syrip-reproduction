#!/usr/bin/env bash
# Copy the code on disk to the cluster project folder, with no commit required. Run from the project root.
#
#   bash scripts/push_code.sh            # send src scripts jobs configs requirements.txt as they are on disk
#   bash scripts/push_code.sh --dry-run  # print the plan, send nothing
#
# A push of a commit would refuse every uncommitted edit, and the three Markdown logs change after each
# cluster action. This script ships the files on disk instead and keeps the trace another way:
#   - When the shipped paths equal HEAD, the folder is code/<commit>.
#   - Otherwise the folder is code/<commit>-wip-<utc time>, and a HASHES file lists the sha256 of
#     every shipped file, so a run still points at exact code.
# A folder is never overwritten. code/current moves to the new folder. Older folders stay, so a push
# cannot change a running job. Line endings are normalized to LF on the cluster.
#
#   HOST=alias     overrides the ssh host (default: explorer)
#   PROJECT=name   overrides the project folder (default: name of the repository folder)
set -euo pipefail

HOST="${HOST:-explorer}"
PATHS=(src scripts jobs configs requirements.txt)
TOP="$(git rev-parse --show-toplevel)"
PROJECT="${PROJECT:-$(basename "$TOP")}"
DRY=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

HEAD_COMMIT="$(git -C "$TOP" rev-parse --short=12 HEAD)"
# Tracked edits and untracked files inside the shipped paths make the push a work-in-progress push.
if [ -n "$(git -C "$TOP" status --porcelain -- "${PATHS[@]}" | grep -v '__pycache__' || true)" ]; then
  NAME="$HEAD_COMMIT-wip-$(date -u +%Y%m%dT%H%M%SZ)"
  STATE="work in progress: shipped paths differ from HEAD"
else
  NAME="$HEAD_COMMIT"
  STATE="shipped paths equal HEAD"
fi

echo "project: $PROJECT"
echo "code:    $NAME ($STATE)"
echo "paths:   ${PATHS[*]}"
echo "target:  $HOST:/scratch/\$USER/$PROJECT/code/$NAME"
[ "$DRY" = 1 ] && exit 0

tar -c --exclude='__pycache__' --exclude='*.pyc' -C "$TOP" "${PATHS[@]}" |
  ssh "$HOST" "PROJECT='$PROJECT' NAME='$NAME' bash -c '
    set -euo pipefail
    root=/scratch/\$USER/\$PROJECT
    mkdir -p \$root/code \$root/env \$root/containers \$root/hf \$root/data \$root/raw \$root/logs \$root/tmp
    if [ -e \$root/code/\$NAME ]; then
      cat > /dev/null
      echo already on the cluster: \$NAME
    else
      stage=\$(mktemp -d \$root/code/.incoming.XXXXXX)
      tar -x --no-same-owner -C \$stage
      find \$stage -type f \( -name \"*.py\" -o -name \"*.sh\" -o -name \"*.sbatch\" -o -name \"*.yaml\" -o -name \"*.txt\" \) -exec sed -i \"s/\r\$//\" {} +
      echo \$NAME > \$stage/COMMIT
      (cd \$stage && find . -type f ! -name HASHES ! -name COMMIT | sort | xargs sha256sum > HASHES)
      mv \$stage \$root/code/\$NAME
    fi
    ln -sfn \$NAME \$root/code/current
    echo current: \$(cat \$root/code/current/COMMIT), \$(wc -l < \$root/code/current/HASHES) files hashed
  '"

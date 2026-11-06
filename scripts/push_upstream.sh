#!/usr/bin/env bash
# Send the pinned upstream FiDIP clone to the cluster project folder. Run from the project root.
#
#   bash scripts/push_upstream.sh            # send third_party/Infant-Pose-Estimation
#   bash scripts/push_upstream.sh --dry-run  # print the plan, send nothing
#
# Git ignores the clone, so cluster_push.sh cannot ship it. This script archives the tracked
# files at the clone's HEAD, so LF endings survive and .git stays local. The copy lands in
# /scratch/$USER/<project>/upstream/Infant-Pose-Estimation with a COMMIT file.
#
# The script stops when the clone's HEAD differs from the pin in configs/paths.yaml, when the
# clone has local edits, and when the cluster holds a copy with another commit. The owner decides.
#
#   HOST=alias     overrides the ssh host (default: explorer)
#   PROJECT=name   overrides the project folder (default: name of the repository folder)
set -euo pipefail

HOST="${HOST:-explorer}"
TOP="$(git rev-parse --show-toplevel)"
PROJECT="${PROJECT:-$(basename "$TOP")}"
CLONE="$TOP/third_party/Infant-Pose-Estimation"
PIN="$(sed -n 's/^  commit: *\([0-9a-f]\{40\}\).*/\1/p' "$TOP/configs/paths.yaml")"
DRY=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

[ -d "$CLONE/.git" ] || { echo "no clone at $CLONE. Run: git clone https://github.com/ostadabbas/Infant-Pose-Estimation.git $CLONE" >&2; exit 1; }
[ -n "$PIN" ] || { echo "configs/paths.yaml has no upstream.commit" >&2; exit 1; }
HEAD_COMMIT="$(git -C "$CLONE" rev-parse HEAD)"
if [ "$HEAD_COMMIT" != "$PIN" ]; then
  echo "clone HEAD $HEAD_COMMIT differs from the pin $PIN. Check out the pin, or move the pin." >&2
  exit 1
fi
if [ -n "$(git -C "$CLONE" status --porcelain)" ]; then
  echo "the clone has local edits. Upstream stays unmodified. Put changes in src/." >&2
  exit 1
fi

echo "project: $PROJECT"
echo "commit:  $PIN"
echo "target:  $HOST:/scratch/\$USER/$PROJECT/upstream/Infant-Pose-Estimation"
[ "$DRY" = 1 ] && exit 0

# The remote side drains stdin before an early exit, so the local archive never sees a broken pipe.
git -C "$CLONE" -c core.autocrlf=false -c core.eol=lf archive --format=tar HEAD |
  ssh "$HOST" "PROJECT='$PROJECT' COMMIT='$PIN' bash -c '
    set -euo pipefail
    dest=/scratch/\$USER/\$PROJECT/upstream/Infant-Pose-Estimation
    if [ -f \$dest/COMMIT ]; then
      have=\$(cat \$dest/COMMIT)
      cat > /dev/null
      if [ \"\$have\" = \"\$COMMIT\" ]; then echo already on the cluster: \$have; exit 0; fi
      echo \"cluster copy is at \$have, the pin is \$COMMIT. Move the copy aside by hand first.\" >&2
      exit 1
    fi
    mkdir -p \$(dirname \$dest)
    stage=\$(mktemp -d \$(dirname \$dest)/.incoming.XXXXXX)
    tar -x -C \$stage
    echo \$COMMIT > \$stage/COMMIT
    mv \$stage \$dest
    echo upstream: \$(cat \$dest/COMMIT)
  '"

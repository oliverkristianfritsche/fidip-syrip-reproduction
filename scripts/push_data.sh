#!/usr/bin/env bash
# Send data/ from the local data root to the cluster project folder. Run from the project root.
#
#   bash scripts/push_data.sh                 # send ../_data/<project>/data/ to $ROOT/data/
#   bash scripts/push_data.sh --only models   # one subfolder of data/, such as models or syrip
#   bash scripts/push_data.sh --dry-run       # print the plan and the local size, send nothing
#
# The transfer host carries the copy, because login nodes kill long processes. tar over ssh
# restarts from zero on a dropped link, so send one subfolder at a time for a large set.
# Files with the same path on the cluster are overwritten. Nothing is deleted.
# After the copy, the script compares the md5 of every .pth file on both sides.
#
#   HOST=alias     overrides the ssh host (default: explorer)
#   PROJECT=name   overrides the project folder (default: name of the repository folder)
set -euo pipefail

HOST="${HOST:-explorer}"
SSH_OPTS=(-o HostName=xfer.discovery.neu.edu -o ServerAliveInterval=20 -o ServerAliveCountMax=3)
TOP="$(git rev-parse --show-toplevel)"
PROJECT="${PROJECT:-$(basename "$TOP")}"
SRC="$(dirname "$TOP")/_data/$PROJECT/data"
ONLY=""
DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --only)    ONLY="${2:?--only needs a folder name under data/}"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done
case "$ONLY" in
  */*|..) echo "--only takes one folder name under data/" >&2; exit 2 ;;
esac

[ -d "$SRC" ] || { echo "no local data root: $SRC" >&2; exit 1; }
if [ -n "$ONLY" ] && [ ! -d "$SRC/$ONLY" ]; then
  echo "no such folder: $SRC/$ONLY" >&2
  exit 1
fi

echo "source:      $SRC${ONLY:+/$ONLY}"
echo "destination: $HOST:/scratch/\$USER/$PROJECT/data${ONLY:+/$ONLY}"
du -sh "$SRC${ONLY:+/$ONLY}"
[ "$DRY" = 1 ] && exit 0

tar -C "$SRC" -c "${ONLY:-.}" |
  ssh "${SSH_OPTS[@]}" "$HOST" "dest=/scratch/\$USER/$PROJECT/data; mkdir -p \$dest; tar -x --no-same-owner -C \$dest; echo copied to \$dest"

echo "md5 check of .pth files"
# Windows md5sum prints "hash *path" for binary files, Linux prints "hash  path". Normalize the local side.
LOCAL="$(cd "$SRC" && find "${ONLY:-.}" -name '*.pth' -type f | sort | xargs -r md5sum | sed 's/ \*/  /')"
if [ -z "$LOCAL" ]; then
  echo "no .pth files in the sent set"
  exit 0
fi
REMOTE="$(ssh "${SSH_OPTS[@]}" "$HOST" "cd /scratch/\$USER/$PROJECT/data && find ${ONLY:-.} -name '*.pth' -type f | sort | xargs -r md5sum")"
if [ "$LOCAL" = "$REMOTE" ]; then
  echo "md5 match"
  echo "$REMOTE"
else
  echo "md5 differ between local and cluster:" >&2
  diff <(echo "$LOCAL") <(echo "$REMOTE") >&2 || true
  exit 3
fi

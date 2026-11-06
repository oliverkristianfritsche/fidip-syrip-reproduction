#!/usr/bin/env python3
"""Resolve project paths from configs/paths.yaml for the host that runs the script.

    from smalldata.paths import Paths
    p = Paths.load()
    p.root                      # /scratch/fritsche.o/smalldata-project on Explorer
    p.upstream                  # the pinned FiDIP clone
    p.abs("data.syrip")         # root / data/syrip
    p.add_upstream_lib()        # puts <upstream>/lib on sys.path, so `import config` works

Host selection: SMALLDATA_ROOT wins. Otherwise the cluster root from paths.yaml is used when
it exists, and the local root otherwise. FIDIP_UPSTREAM overrides the upstream location.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "paths.yaml"


def _get(tree, dotted):
    node = tree
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            sys.exit(f"paths.yaml has no key: {dotted}")
        node = node[key]
    return node


class Paths:
    def __init__(self, tree, root, host):
        self.tree = tree
        self.root = Path(root)
        self.host = host
        override = os.environ.get("FIDIP_UPSTREAM")
        if override:
            self.upstream = Path(override)
        elif host == "cluster":
            self.upstream = self.root / _get(tree, "upstream.cluster_path")
        else:
            self.upstream = REPO / _get(tree, "upstream.local_path")

    @classmethod
    def load(cls, config=CONFIG):
        import yaml  # heavy import stays inside the function

        with open(config, encoding="utf-8") as f:
            tree = yaml.safe_load(f)
        cluster = Path(_get(tree, "roots.cluster"))
        override = os.environ.get("SMALLDATA_ROOT")
        if override:
            host = "cluster" if Path(override) == cluster else "local"
            return cls(tree, override, host)
        if cluster.is_dir():
            return cls(tree, cluster, "cluster")
        return cls(tree, (REPO / _get(tree, "roots.local")).resolve(), "local")

    def get(self, dotted):
        return _get(self.tree, dotted)

    def abs(self, dotted):
        return self.root / _get(self.tree, dotted)

    def pinned_commit(self):
        return str(_get(self.tree, "upstream.commit"))

    def upstream_commit(self):
        """Commit of the upstream copy: the COMMIT file on the cluster, git or .git/HEAD locally."""
        marker = self.upstream / "COMMIT"
        if marker.is_file():
            return marker.read_text(encoding="utf-8").strip()
        try:
            # safe.directory: WSL git refuses a checkout on a Windows mount that another user owns.
            return subprocess.check_output(
                ["git", "-c", "safe.directory=*", "-C", str(self.upstream), "rev-parse", "HEAD"],
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
        head = self.upstream / ".git" / "HEAD"
        if not head.is_file():
            return ""
        ref = head.read_text(encoding="utf-8").strip()
        if not ref.startswith("ref: "):
            return ref
        ref = ref[5:]
        loose = self.upstream / ".git" / ref
        if loose.is_file():
            return loose.read_text(encoding="utf-8").strip()
        packed = self.upstream / ".git" / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0]
        return ""

    def add_upstream_lib(self):
        lib = self.upstream / "lib"
        if not lib.is_dir():
            sys.exit(f"upstream lib folder is missing: {lib}")
        if str(lib) not in sys.path:
            sys.path.insert(0, str(lib))
        return lib

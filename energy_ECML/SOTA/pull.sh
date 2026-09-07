#!/usr/bin/env bash
# Clone the repositories listed in sources.toml at their pinned commits.
#
#   bash energy_ECML/SOTA/pull.sh            # all of them
#   bash energy_ECML/SOTA/pull.sh reactnet   # just one
#
# Clones land in energy_ECML/SOTA/repos/<name>/, which is gitignored.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
want="${1:-}"
python3 - "$here" "$want" <<'PY'
import os, subprocess, sys, tomllib
here, want = sys.argv[1], sys.argv[2]
src = os.path.join(here, "sources.toml")
with open(src, "rb") as f:
    cfg = tomllib.load(f)
if not cfg:
    print(f"{src}: no repositories listed yet"); raise SystemExit(0)
for name, e in cfg.items():
    if want and name != want:
        continue
    dest = os.path.join(here, "repos", name)
    commit = e.get("commit") or ""
    if not commit:
        print(f"{name}: no commit pinned in sources.toml -- refusing to clone a moving target")
        continue
    if not os.path.isdir(dest):
        subprocess.run(["git", "clone", e["url"], dest], check=True)
    subprocess.run(["git", "-C", dest, "fetch", "--all"], check=True)
    subprocess.run(["git", "-C", dest, "checkout", commit], check=True)
    print(f"{name}: {e['url']} @ {commit}")
PY

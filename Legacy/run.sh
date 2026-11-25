#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [[ ! -f .venv/Scripts/python.exe ]]; then
    echo 'Missing .venv/Scripts/python.exe. Keep run.sh in CVFingerprint.' >&2
    exit 1
fi

# Run 50 edits on the GPU; forward optional benchmark arguments.
exec ./.venv/Scripts/python.exe -u benchmark.py --runs 50 "$@"

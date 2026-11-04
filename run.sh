#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source scripts/env.sh
if [[ $# == 0 ]]; then set -- all; fi
exec .venv/bin/python -m scripts.reproduce "$@"

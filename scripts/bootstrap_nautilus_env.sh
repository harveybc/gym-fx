#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${NAUTILUS_VENV:-$HOME/.venvs/gymfx-nautilus}"
CONTRACTS="${TRADING_CONTRACTS_ROOT:-$ROOT/../trading-contracts}"

command -v uv >/dev/null 2>&1 || {
  echo "uv is required: https://docs.astral.sh/uv/" >&2
  exit 1
}

uv venv "$VENV" --python 3.12
uv pip install --python "$VENV/bin/python" -e "$ROOT[nautilus]" "pytest>=8,<9" psutil
if [[ -d "$CONTRACTS" ]]; then
  uv pip install --python "$VENV/bin/python" -e "$CONTRACTS"
fi

echo "Nautilus environment ready: $VENV/bin/python"

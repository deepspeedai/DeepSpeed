#!/usr/bin/env bash
# PR 7916: venv at .venvs/pr7916, PyTorch 2.8 + cu128, then repro on current branch vs master.
#
# Venv: reuses $VENV_DIR if bin/activate exists (no pip). --force-install always recreates.
#       --skip-install reuses only and errors if the venv is missing.
#
# Usage: ./scripts/setup_pr7916.sh [--force-install] [--skip-install]
# Env:   PR7916_VENV_DIR, PR7916_MAIN_REF (default master), PR7916_FORCE_INSTALL, PR7916_SKIP_INSTALL
#
# --- Recorded test environment (original bug report / CI reference) ---
# OS:              Ubuntu 22.04
# GPU:             NVIDIA H100 80GB PCIe
# Python:          3.11
# PyTorch:         2.8.0+cu128  (torch.version.cuda: 12.8)
# DeepSpeed:       0.16.4 wheel (issue); PR validates against editable install + this script's venv
# CUDA (driver):   12.8 (via PyTorch cu128 wheels; nvcc optional / often N/A)
# Launcher:        deepspeed CLI (repro uses torchrun --standalone --nproc_per_node=1)
# -------------------------------------------------------------------------
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VENV_DIR="${PR7916_VENV_DIR:-$ROOT/.venvs/pr7916}"
MAIN_REF="${PR7916_MAIN_REF:-master}"
VENV_SH="$VENV_DIR/bin/activate"

truthy() { case "${1:-}" in 1|true|yes|on) return 0;; *) return 1;; esac; }

force=0
skip_only=0
truthy "${PR7916_FORCE_INSTALL:-}" && force=1
truthy "${PR7916_SKIP_INSTALL:-}" && skip_only=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force-install) force=1 ;;
    --skip-install) skip_only=1 ;;
    *) echo "error: unknown argument: $1" >&2; exit 1 ;;
  esac
  shift
done

print_runtime_env() {
  python <<'PY'
import platform
import sys

import deepspeed
import torch

print("==> Runtime environment (this session)")
print(f"  python:          {sys.version.split()[0]} ({platform.system()} {platform.release()})")
print(f"  torch:           {torch.__version__}")
print(f"  torch.version.cuda: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"  cuda available:  yes ({torch.cuda.device_count()} device(s))")
    print(f"  cuda device 0:   {torch.cuda.get_device_name(0)}")
else:
    print("  cuda available:  no")
print(f"  deepspeed:       {deepspeed.__version__}")
print(f"  deepspeed path:  {deepspeed.__file__}")
PY
}

# Sets: full=1 → wipe + venv + pip; full=0 → activate existing only
decide_full_setup() {
  if [[ "$force" -eq 1 ]]; then
    echo 1
  elif [[ "$skip_only" -eq 1 ]]; then
    echo 0
  elif [[ -f "$VENV_SH" ]]; then
    echo 0
  else
    echo 1
  fi
}

setup_venv() {
  local full
  full="$(decide_full_setup)"

  if [[ "$full" -eq 0 ]]; then
    [[ -f "$VENV_SH" ]] || {
      echo "error: no venv at $VENV_DIR (drop --skip-install or run once without it)" >&2
      exit 1
    }
    echo "==> Reusing venv $VENV_DIR (use --force-install to reinstall)"
  else
    echo "==> Creating venv at $VENV_DIR"
    rm -rf "$VENV_DIR"
    mkdir -p "$(dirname "$VENV_DIR")"
    python3 -m venv "$VENV_DIR"
  fi

  # shellcheck source=/dev/null
  . "$VENV_SH"

  if [[ "$full" -eq 1 ]]; then
    python -c 'import sys; assert sys.version_info[:2] == (3, 11), "Use Python 3.11 to match the bug report"' || {
      echo "Warning: expected Python 3.11; found $(python -V)" >&2
    }
    pip install -U pip setuptools wheel
    pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128
    pip install -r requirements/requirements.txt
    pip install -e .
    pip install pytest
  fi

  print_runtime_env
}

run_repro_compare() {
  local REPRO_SRC="$ROOT/scripts/repro_pr7916.py" REPRO_TMP FIX_BRANCH STASHED=0 MAIN_EC

  [[ -f "$REPRO_SRC" ]] || {
    echo "error: missing $REPRO_SRC (need this file on the current branch)" >&2
    exit 1
  }

  REPRO_TMP="$(mktemp /tmp/repro_pr7916_XXXXXX.py)"
  cp "$REPRO_SRC" "$REPRO_TMP"
  trap 'rm -f "$REPRO_TMP"' EXIT

  FIX_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
  local -a run=(torchrun --standalone --nproc_per_node=1)

  echo ""
  echo "==> [1/2] Repro on $FIX_BRANCH (expect OK)"
  "${run[@]}" "$REPRO_TMP"

  echo ""
  echo "==> [2/2] Repro on $MAIN_REF (expect setup_context RuntimeError on unfixed tree)"
  if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
    echo "==> Stashing local changes for checkout..."
    git stash push -m "pr7916-setup: temp stash before main repro"
    STASHED=1
  fi
  if ! git checkout "$MAIN_REF"; then
    echo "error: checkout $MAIN_REF failed" >&2
    [[ "$STASHED" -eq 1 ]] && git stash pop || true
    exit 1
  fi

  set +e
  "${run[@]}" "$REPRO_TMP"
  MAIN_EC=$?
  set -e

  if [[ "$MAIN_EC" -eq 0 ]]; then
    echo "warning: main-branch repro exited 0 (expected failure on unfixed tree)." >&2
  else
    echo "main-branch repro exited $MAIN_EC (non-zero expected for unfixed tree)."
  fi

  echo ""
  echo "==> Restoring $FIX_BRANCH"
  git checkout "$FIX_BRANCH"
  if [[ "$STASHED" -eq 1 ]]; then
    git stash pop || echo "warning: stash pop failed — see git stash list" >&2
  fi
}

setup_venv
run_repro_compare

echo ""
echo "Done. Activate: source $VENV_DIR/bin/activate"

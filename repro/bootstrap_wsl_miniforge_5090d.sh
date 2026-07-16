#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-minionerec-5090d}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
MINIFORGE_HOME="${MINIFORGE_HOME:-$HOME/miniforge3}"
MINIFORGE_URL="${MINIFORGE_URL:-https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This bootstrap script is intended for WSL/Linux."
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "Warning: nvidia-smi was not found in WSL. Install/update the Windows NVIDIA driver with WSL support first."
else
  nvidia-smi
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "Installing Miniforge to ${MINIFORGE_HOME}"
  installer="/tmp/Miniforge3-Linux-x86_64.sh"
  if command -v curl >/dev/null 2>&1; then
    curl -L "${MINIFORGE_URL}" -o "${installer}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${installer}" "${MINIFORGE_URL}"
  else
    echo "Neither curl nor wget is installed. Run: sudo apt update && sudo apt install -y curl ca-certificates"
    exit 1
  fi
  bash "${installer}" -b -p "${MINIFORGE_HOME}"
fi

source "${MINIFORGE_HOME}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

conda activate "${ENV_NAME}"
python -m pip install -U pip setuptools wheel

python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r repro/requirements-5090d-cu128.txt
python -m pip uninstall -y deepspeed >/dev/null 2>&1 || true

python repro/check_5090d_env.py --skip-model-load

echo
echo "Environment is ready."
echo "Next shell activation command:"
echo "source \"${MINIFORGE_HOME}/etc/profile.d/conda.sh\" && conda activate ${ENV_NAME}"

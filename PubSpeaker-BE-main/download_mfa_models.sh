#!/usr/bin/env bash
# Download English MFA models in an existing conda MFA environment.

set -euo pipefail

ENV_NAME="${ENV_NAME:-mfa}"

log() {
    printf "[models] %s\n" "$*"
}

if command -v conda >/dev/null 2>&1; then
    CONDA_EXE="$(command -v conda)"
elif [ -x "${HOME}/miniconda3/bin/conda" ]; then
    CONDA_EXE="${HOME}/miniconda3/bin/conda"
else
    echo "[models][error] Conda not found. Run ./install_mfa.sh first." >&2
    exit 1
fi

if ! "${CONDA_EXE}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[models][error] Conda environment '${ENV_NAME}' does not exist." >&2
    echo "[models][error] Run ./install_mfa.sh to create it." >&2
    exit 1
fi

log "Downloading dictionary model: english_us_arpa"
"${CONDA_EXE}" run -n "${ENV_NAME}" mfa model download dictionary english_us_arpa

log "Downloading acoustic model: english_us_arpa"
"${CONDA_EXE}" run -n "${ENV_NAME}" mfa model download acoustic english_us_arpa

log "Installed dictionary models"
"${CONDA_EXE}" run -n "${ENV_NAME}" mfa model list dictionary

log "Installed acoustic models"
"${CONDA_EXE}" run -n "${ENV_NAME}" mfa model list acoustic

log "Done"

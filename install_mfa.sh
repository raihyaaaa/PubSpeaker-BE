#!/usr/bin/env bash
# Full installation script for PubSpeaker on macOS and Linux.
# Installs/uses Miniconda, creates MFA environment, installs app deps,
# downloads MFA English models, and verifies setup.

set -euo pipefail

ENV_NAME="${ENV_NAME:-mfa}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="${PROJECT_DIR}/requirements.txt"

log() {
    printf "[install] %s\n" "$*"
}

fail() {
    printf "[install][error] %s\n" "$*" >&2
    exit 1
}

detect_installer_name() {
    local os
    local arch
    os="$(uname -s)"
    arch="$(uname -m)"

    case "${os}" in
        Darwin)
            case "${arch}" in
                arm64) echo "Miniconda3-latest-MacOSX-arm64.sh" ;;
                x86_64) echo "Miniconda3-latest-MacOSX-x86_64.sh" ;;
                *) fail "Unsupported macOS architecture: ${arch}" ;;
            esac
            ;;
        Linux)
            case "${arch}" in
                aarch64|arm64) echo "Miniconda3-latest-Linux-aarch64.sh" ;;
                x86_64) echo "Miniconda3-latest-Linux-x86_64.sh" ;;
                *) fail "Unsupported Linux architecture: ${arch}" ;;
            esac
            ;;
        *)
            fail "Unsupported OS: ${os}. Use Windows script for Windows."
            ;;
    esac
}

ensure_conda() {
    if command -v conda >/dev/null 2>&1; then
        CONDA_EXE="$(command -v conda)"
        return
    fi

    local installer
    local url
    local tmp_installer

    installer="$(detect_installer_name)"
    url="https://repo.anaconda.com/miniconda/${installer}"
    tmp_installer="/tmp/${installer}"

    if ! command -v curl >/dev/null 2>&1; then
        fail "curl is required to install Miniconda"
    fi

    log "Conda not found. Installing Miniconda (${installer})"
    curl -fsSL "${url}" -o "${tmp_installer}"
    bash "${tmp_installer}" -b -p "${HOME}/miniconda3"
    rm -f "${tmp_installer}"

    CONDA_EXE="${HOME}/miniconda3/bin/conda"
    [ -x "${CONDA_EXE}" ] || fail "Conda installation failed"

    log "Miniconda installed at ${HOME}/miniconda3"
}

create_or_update_env() {
    if "${CONDA_EXE}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
        log "Conda environment '${ENV_NAME}' already exists"
    else
        log "Creating conda environment '${ENV_NAME}' (python=${PYTHON_VERSION})"
        "${CONDA_EXE}" create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
    fi

    log "Installing MFA and ffmpeg from conda-forge"
    "${CONDA_EXE}" install -n "${ENV_NAME}" -c conda-forge montreal-forced-aligner ffmpeg -y
}

install_python_deps() {
    [ -f "${REQ_FILE}" ] || fail "requirements.txt not found at ${REQ_FILE}"

    log "Upgrading pip in '${ENV_NAME}'"
    "${CONDA_EXE}" run -n "${ENV_NAME}" python -m pip install --upgrade pip

    # MFA is installed via conda to avoid binary dependency issues.
    if grep -Eiq '^Montreal_Forced_Aligner(==.*)?$' "${REQ_FILE}"; then
        local tmp_req
        tmp_req="$(mktemp)"
        grep -Eiv '^Montreal_Forced_Aligner(==.*)?$' "${REQ_FILE}" > "${tmp_req}"
        log "Installing pip dependencies (excluding Montreal_Forced_Aligner)"
        "${CONDA_EXE}" run -n "${ENV_NAME}" python -m pip install -r "${tmp_req}"
        rm -f "${tmp_req}"
    else
        log "Installing pip dependencies"
        "${CONDA_EXE}" run -n "${ENV_NAME}" python -m pip install -r "${REQ_FILE}"
    fi
}

download_models() {
    log "Downloading MFA dictionary model: english_us_arpa"
    "${CONDA_EXE}" run -n "${ENV_NAME}" mfa model download dictionary english_us_arpa

    log "Downloading MFA acoustic model: english_us_arpa"
    "${CONDA_EXE}" run -n "${ENV_NAME}" mfa model download acoustic english_us_arpa
}

verify_installation() {
    log "Verifying MFA and core app imports"
    "${CONDA_EXE}" run -n "${ENV_NAME}" mfa version
    "${CONDA_EXE}" run -n "${ENV_NAME}" python -c "import fastapi, whisper; print('python deps ok')"
}

main() {
    log "Starting PubSpeaker full installation"
    ensure_conda
    create_or_update_env
    install_python_deps
    download_models
    verify_installation

    cat <<EOF

Installation complete.

Use the environment:
  conda activate ${ENV_NAME}

Run the API from project root:
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload

If conda is not initialized in your shell yet:
  ${CONDA_EXE} init
  # then restart terminal
EOF
}

main "$@"

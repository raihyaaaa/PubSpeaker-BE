# PubSpeaker Setup Guide

This guide installs everything needed for PubSpeaker:

- Python dependencies from `requirements.txt`
- Montreal Forced Aligner (MFA) in a dedicated Miniconda environment
- MFA English US dictionary and acoustic models

The project uses a single conda environment named `mfa` by default.

## 1) System Requirements

- OS: Linux, macOS, or Windows 10/11 (64-bit)
- Disk space: at least 8 GB free (models + Python packages)
- RAM: 8 GB minimum, 16 GB recommended
- Internet connection for downloading models and packages

## 2) Quick Install (Recommended)

From the project root directory:

### macOS / Linux

```bash
chmod +x install_mfa.sh download_mfa_models.sh
./install_mfa.sh
```

### Windows (PowerShell)

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\install_full_windows.ps1
```

This performs full installation end-to-end.

## 3) What the Installer Does

1. Installs Miniconda if `conda` is missing
2. Creates (or reuses) conda env: `mfa`
3. Installs MFA + ffmpeg from `conda-forge`
4. Installs Python packages from `requirements.txt`
5. Downloads MFA models:
   - `english_us_arpa` dictionary
   - `english_us_arpa` acoustic model
6. Verifies MFA and core imports (`fastapi`, `whisper`)

## 4) Run the API

After installation:

```bash
conda activate mfa
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

API docs:

- Swagger UI: http://127.0.0.1:8000/docs

## 5) Re-download MFA Models Only

If dependencies are already installed and you only need models:

### macOS / Linux

```bash
./download_mfa_models.sh
```

### Windows (PowerShell)

```powershell
conda run -n mfa mfa model download dictionary english_us_arpa
conda run -n mfa mfa model download acoustic english_us_arpa
```

## 6) OS-Specific Notes

### Linux

- If `curl` is missing, install it first using your package manager.
- If conda command is not recognized after install, run:

```bash
~/miniconda3/bin/conda init
```

Then restart the terminal.

### macOS

- Works on Apple Silicon (`arm64`) and Intel (`x86_64`).
- If conda command is not recognized after install, run:

```bash
~/miniconda3/bin/conda init
```

Then restart the terminal.

### Windows

- Run PowerShell as your normal user (Admin not required).
- If `conda` is not recognized in a new terminal after install:

```powershell
$env:USERPROFILE\miniconda3\Scripts\conda.exe init powershell
```

Then close and reopen PowerShell.

## 7) Validation Checklist

Run these after setup:

```bash
conda activate mfa
mfa version
python -c "import fastapi, whisper; print('ok')"
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Expected:

- `mfa version` prints an MFA version
- Python import command prints `ok`
- Uvicorn starts without missing dependency errors

## 8) Troubleshooting

- `mfa: command not found`
  - Use `conda activate mfa`, then retry.
- `No module named ...`
  - Re-run installer to ensure all pip packages are installed.
- MFA model errors
  - Re-run `./download_mfa_models.sh` (macOS/Linux) or model commands on Windows.

## 9) Benchmarking (Accuracy + Speed)

PubSpeaker now includes a benchmark script at `benchmark.py`.

It reports:

- Model/component load time (`init_time_s`)
- Operation latency (`mean`, `median`, `p95`, `min`, `max`)
- Accuracy proxies for grammar, improvement, and pronunciation coverage
- Optional transcription accuracy (`WER`, `CER`) when reference text is provided
- Optional MFA alignment metrics (`precision`, `recall`, `f1`) when expected deviations are provided

### Run benchmark (no audio)

```bash
conda activate mfa
python benchmark.py
```

This runs grammar, transcript improvement, and pronunciation lookup benchmarks.

### Run benchmark with audio (transcription speed + WER/CER)

```bash
conda activate mfa
python benchmark.py \
  --audio "path/to/sample.wav" \
  --reference-text "exact ground truth transcript here"
```

### Run benchmark with MFA alignment metrics

```bash
conda activate mfa
python benchmark.py \
  --audio "path/to/sample.wav" \
  --transcript-text "transcript to align" \
  --with-alignment \
  --expected-deviations "word1,word2"
```

### Windows PowerShell examples

```powershell
conda activate mfa
python .\benchmark.py

python .\benchmark.py --audio "C:\path\sample.wav" --reference-text "exact ground truth transcript here"

python .\benchmark.py --audio "C:\path\sample.wav" --transcript-text "transcript to align" --with-alignment --expected-deviations "word1,word2"
```

### Output file

Each run saves a JSON report:

- Default: `benchmark_results_YYYYMMDD_HHMMSS.json`
- Custom path: use `--output your_report.json`

Example:

```bash
python benchmark.py --output benchmark_my_machine.json
```

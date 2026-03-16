Param(
    [string]$EnvName = "mfa",
    [string]$PythonVersion = "3.11"
)

$ErrorActionPreference = "Stop"

function Write-Info {
    param([string]$Message)
    Write-Host "[install] $Message"
}

function Get-CondaExe {
    $condaCmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($condaCmd) {
        return $condaCmd.Source
    }

    $defaultConda = Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"
    if (Test-Path $defaultConda) {
        return $defaultConda
    }

    return $null
}

function Install-Miniconda {
    Write-Info "Conda not found. Installing Miniconda..."

    $installer = Join-Path $env:TEMP "Miniconda3-latest-Windows-x86_64.exe"
    Invoke-WebRequest -Uri "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -OutFile $installer

    $installPath = Join-Path $env:USERPROFILE "miniconda3"
    $args = @(
        "/InstallationType=JustMe",
        "/RegisterPython=0",
        "/AddToPath=0",
        "/S",
        "/D=$installPath"
    )

    $proc = Start-Process -FilePath $installer -ArgumentList $args -Wait -PassThru
    if ($proc.ExitCode -ne 0) {
        throw "Miniconda installer failed with exit code $($proc.ExitCode)."
    }

    Remove-Item $installer -ErrorAction SilentlyContinue

    $condaExe = Join-Path $installPath "Scripts\conda.exe"
    if (-not (Test-Path $condaExe)) {
        throw "Conda installation completed but conda.exe not found at $condaExe"
    }

    return $condaExe
}

function New-OrUpdateMfaEnv {
    param(
        [string]$CondaExe,
        [string]$Name,
        [string]$PyVersion
    )

    $envList = & $CondaExe env list
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to list conda environments."
    }

    $exists = $false
    foreach ($line in $envList) {
        if ($line -match "^\s*$Name\s") {
            $exists = $true
            break
        }
    }

    if (-not $exists) {
        Write-Info "Creating conda environment '$Name' (python=$PyVersion)"
        & $CondaExe create -n $Name "python=$PyVersion" -y
    } else {
        Write-Info "Conda environment '$Name' already exists"
    }

    Write-Info "Installing MFA and ffmpeg from conda-forge"
    & $CondaExe install -n $Name -c conda-forge montreal-forced-aligner ffmpeg -y
}

function Install-PipRequirements {
    param(
        [string]$CondaExe,
        [string]$Name,
        [string]$ProjectDir
    )

    $requirements = Join-Path $ProjectDir "requirements.txt"
    if (-not (Test-Path $requirements)) {
        throw "requirements.txt not found at $requirements"
    }

    Write-Info "Upgrading pip in '$Name'"
    & $CondaExe run -n $Name python -m pip install --upgrade pip

    $tmpReq = Join-Path $env:TEMP ("pubspeaker-req-" + [Guid]::NewGuid().ToString() + ".txt")
    Get-Content $requirements |
        Where-Object { $_ -notmatch '^Montreal_Forced_Aligner(==.*)?$' } |
        Set-Content -Path $tmpReq -Encoding UTF8

    Write-Info "Installing pip dependencies"
    & $CondaExe run -n $Name python -m pip install -r $tmpReq
    Remove-Item $tmpReq -ErrorAction SilentlyContinue
}

function Download-MfaModels {
    param(
        [string]$CondaExe,
        [string]$Name
    )

    Write-Info "Downloading MFA dictionary model: english_us_arpa"
    & $CondaExe run -n $Name mfa model download dictionary english_us_arpa

    Write-Info "Downloading MFA acoustic model: english_us_arpa"
    & $CondaExe run -n $Name mfa model download acoustic english_us_arpa
}

function Verify-Install {
    param(
        [string]$CondaExe,
        [string]$Name
    )

    Write-Info "Verifying MFA and core app imports"
    & $CondaExe run -n $Name mfa version
    & $CondaExe run -n $Name python -c "import fastapi, whisper; print('python deps ok')"
}

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Info "Starting PubSpeaker full installation"

$condaExe = Get-CondaExe
if (-not $condaExe) {
    $condaExe = Install-Miniconda
}

New-OrUpdateMfaEnv -CondaExe $condaExe -Name $EnvName -PyVersion $PythonVersion
Install-PipRequirements -CondaExe $condaExe -Name $EnvName -ProjectDir $projectDir
Download-MfaModels -CondaExe $condaExe -Name $EnvName
Verify-Install -CondaExe $condaExe -Name $EnvName

Write-Host ""
Write-Host "Installation complete."
Write-Host ""
Write-Host "Use the environment:"
Write-Host "  conda activate $EnvName"
Write-Host ""
Write-Host "Run the API from project root:"
Write-Host "  uvicorn app:app --host 0.0.0.0 --port 8000 --reload"
Write-Host ""
Write-Host "If 'conda' is not recognized in a new terminal, run:"
Write-Host "  $env:USERPROFILE\miniconda3\Scripts\conda.exe init powershell"

# setup_pose2d_windows.ps1
# Run from repo root in PowerShell:
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\setup_pose2d_windows.ps1
#
# This script recreates the pose2d conda env, installs a pinned OpenMMLab stack,
# clones MMPose, and verifies that RTMPose/HRNet dependencies import correctly.

$ErrorActionPreference = "Stop"

Write-Host "== Pose reprojection research setup ==" -ForegroundColor Cyan

# -----------------------------
# 0. Basic repo setup
# -----------------------------
$RepoRoot = Get-Location
Write-Host "Repo root: $RepoRoot"

New-Item -ItemType Directory -Force -Path "scripts","configs","checkpoints","data/raw","data/processed","outputs","third_party" | Out-Null

@"
.env/
__pycache__/
*.pyc
checkpoints/
data/raw/
data/processed/
outputs/
third_party/mmpose/
"@ | Set-Content -Encoding UTF8 ".gitignore"

git add -f .gitignore | Out-Null
$gitChanges = git status --porcelain
if ($gitChanges) {
    git commit -m "Add setup gitignore" | Out-Null
    git push
} else {
    Write-Host "Git working tree clean."
}

# -----------------------------
# 1. Recreate conda env
# -----------------------------
$EnvName = "pose2d"

Write-Host "Checking conda env: $EnvName"
$envsJson = conda env list --json | ConvertFrom-Json
$envExists = $false
foreach ($envPath in $envsJson.envs) {
    if ((Split-Path $envPath -Leaf) -eq $EnvName) {
        $envExists = $true
    }
}

if ($envExists) {
    Write-Host "Removing existing conda env: $EnvName" -ForegroundColor Yellow
    conda env remove -n $EnvName -y
}

Write-Host "Creating conda env: $EnvName"
conda create -n $EnvName python=3.10 -y

function EnvRun {
    param(
        [Parameter(ValueFromRemainingArguments=$true)]
        [string[]]$Args
    )
    conda run -n $EnvName @Args
}

function PipInstall {
    param(
        [Parameter(ValueFromRemainingArguments=$true)]
        [string[]]$Args
    )
    EnvRun python -m pip install @Args
}

# -----------------------------
# 2. Core Python / PyTorch stack
# -----------------------------
Write-Host "Installing pinned Python packaging tools..."
PipInstall pip==24.2 setuptools==69.5.1 wheel

Write-Host "Installing numpy < 2 for OpenMMLab compatibility..."
PipInstall numpy==1.26.4

Write-Host "Installing PyTorch CUDA 11.8 stack..."
PipInstall torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

Write-Host "Verifying PyTorch..."
EnvRun python -c "import torch; print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available())"

# -----------------------------
# 3. OpenMMLab stack
# -----------------------------
Write-Host "Installing OpenMMLab dependencies..."
PipInstall -U openmim

EnvRun mim install mmengine==0.10.7
EnvRun mim install mmcv==2.1.0
EnvRun mim install mmdet==3.3.0

PipInstall "numpy==1.26.4" --force-reinstall

# -----------------------------
# 4. Clone and install MMPose
# -----------------------------
if (Test-Path "third_party/mmpose") {
    Write-Host "Removing existing third_party/mmpose..."
    Remove-Item -Recurse -Force "third_party/mmpose"
}

Write-Host "Cloning MMPose..."
git clone https://github.com/open-mmlab/mmpose.git third_party/mmpose

Push-Location "third_party/mmpose"
git checkout v1.3.2

Write-Host "Installing chumpy workaround..."
PipInstall chumpy==0.70 --no-build-isolation

Write-Host "Installing MMPose editable..."
EnvRun python -m pip install "-e" "." "--no-build-isolation"

Pop-Location

# -----------------------------
# 5. Verify imports
# -----------------------------
Write-Host "Verifying final imports..."
EnvRun python -c "import torch, mmcv, mmengine, mmdet, mmpose; print('OK imports'); print('torch:', torch.__version__); print('mmcv:', mmcv.__version__); print('mmengine:', mmengine.__version__); print('mmdet:', mmdet.__version__); print('mmpose:', mmpose.__version__)"

# -----------------------------
# 6. Lock env
# -----------------------------
Write-Host "Writing requirements.lock.txt..."
EnvRun python -m pip freeze | Set-Content -Encoding UTF8 "requirements.lock.txt"

git add requirements.lock.txt scripts configs .gitignore | Out-Null
$gitChanges2 = git status --porcelain
if ($gitChanges2) {
    git commit -m "Add reproducible pose2d environment setup" | Out-Null
    git push
}

Write-Host ""
Write-Host "DONE." -ForegroundColor Green
Write-Host "To use the environment later:"
Write-Host "  conda activate pose2d"
Write-Host ""
Write-Host "If mmcv tries to build from source or fails, use WSL2 Ubuntu instead of native Windows."

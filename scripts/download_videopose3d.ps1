$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path "third_party", "checkpoints/videopose3d" | Out-Null

if (!(Test-Path "third_party/VideoPose3D")) {
    git clone https://github.com/facebookresearch/VideoPose3D.git third_party/VideoPose3D
} else {
    Write-Host "[SKIP] third_party/VideoPose3D already exists"
}

$modelPath = "checkpoints/videopose3d/pretrained_h36m_cpn.bin"
$modelUrl = "https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_cpn.bin"

if (!(Test-Path $modelPath)) {
    Write-Host "[DOWNLOAD] VideoPose3D pretrained Human3.6M CPN model"
    Invoke-WebRequest -Uri $modelUrl -OutFile $modelPath
} else {
    Write-Host "[SKIP] $modelPath already exists"
}

Write-Host ""
Write-Host "Downloaded:"
Get-Item $modelPath | Select-Object Name, Length
Write-Host "Done."

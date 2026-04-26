$ErrorActionPreference = "Stop"

$root = "data/raw/mpi_inf_3dhp"
$seqRoot = "$root/S1/Seq1"
$imageRoot = "$seqRoot/imageSequence"

New-Item -ItemType Directory -Force -Path $seqRoot | Out-Null
New-Item -ItemType Directory -Force -Path $imageRoot | Out-Null

$baseUrl = "http://gvv.mpi-inf.mpg.de/3dhp-dataset/S1/Seq1"

function Download-IfMissing {
    param(
        [string]$Url,
        [string]$OutFile
    )

    if (Test-Path $OutFile) {
        Write-Host "[SKIP] $OutFile already exists"
    } else {
        Write-Host "[DOWNLOAD] $Url"
        Invoke-WebRequest -Uri $Url -OutFile $OutFile
    }
}

Download-IfMissing "$baseUrl/annot.mat" "$seqRoot/annot.mat"
Download-IfMissing "$baseUrl/camera.calibration" "$seqRoot/camera.calibration"
Download-IfMissing "$baseUrl/imageSequence/vnect_cameras.zip" "$imageRoot/vnect_cameras.zip"

Write-Host "Unzipping VNect cameras..."
Expand-Archive -Path "$imageRoot/vnect_cameras.zip" -DestinationPath $imageRoot -Force

Write-Host ""
Write-Host "Downloaded small MPI-INF-3DHP subset:"
Get-ChildItem $seqRoot
Get-ChildItem $imageRoot | Select-Object -First 10

Write-Host ""
Write-Host "Done."

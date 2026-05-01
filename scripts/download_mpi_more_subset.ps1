$ErrorActionPreference = "Stop"

$datasetRoot = "data/raw/mpi_inf_3dhp"
$sourceRoot = "http://gvv.mpi-inf.mpg.de/3dhp-dataset"

# Start small. Add more later if this works.
$jobs = @(
    @{ Subject = 1; Seq = 2 },
    @{ Subject = 2; Seq = 1 },
    @{ Subject = 2; Seq = 2 }
)

# Keep zip = same style as your first S1/Seq1 download.
# Set to $false later if disk space becomes annoying.
$KeepZip = $true

function Download-IfMissing($Url, $OutFile) {
    if (Test-Path $OutFile) {
        Write-Host "[SKIP] $OutFile already exists"
        return
    }

    New-Item -ItemType Directory -Force -Path (Split-Path $OutFile) | Out-Null
    Write-Host "[DOWNLOAD] $Url"
    curl.exe -L --retry 5 --continue-at - --output $OutFile $Url

    if (!(Test-Path $OutFile)) {
        throw "Download failed: $OutFile"
    }
}

foreach ($job in $jobs) {
    $s = $job.Subject
    $q = $job.Seq

    $seqRoot = "$datasetRoot/S$s/Seq$q"
    $imageRoot = "$seqRoot/imageSequence"
    $baseUrl = "$sourceRoot/S$s/Seq$q"

    New-Item -ItemType Directory -Force -Path $imageRoot | Out-Null

    Write-Host ""
    Write-Host "=== Downloading S$s Seq$q ==="

    Download-IfMissing "$baseUrl/annot.mat" "$seqRoot/annot.mat"
    Download-IfMissing "$baseUrl/camera.calibration" "$seqRoot/camera.calibration"
    Download-IfMissing "$baseUrl/imageSequence/vnect_cameras.zip" "$imageRoot/vnect_cameras.zip"

    Write-Host "[UNZIP] S$s Seq$q VNect cameras"
    Expand-Archive -Path "$imageRoot/vnect_cameras.zip" -DestinationPath $imageRoot -Force

    if (-not $KeepZip) {
        Remove-Item "$imageRoot/vnect_cameras.zip" -Force
    }

    Write-Host "[DONE] S$s Seq$q"
    Get-ChildItem $seqRoot | Select-Object Name, Length
    Get-ChildItem $imageRoot | Select-Object Name, Length | Select-Object -First 12
}

Write-Host ""
Write-Host "All requested MPI-INF-3DHP subset downloads finished."

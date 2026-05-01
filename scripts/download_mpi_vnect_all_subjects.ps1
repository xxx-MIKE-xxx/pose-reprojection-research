$ErrorActionPreference = "Stop"

$baseUrl = "http://gvv.mpi-inf.mpg.de/3dhp-dataset"
$root = "data/raw/mpi_inf_3dhp"

$subjects = 1..8
$seqs = 1..2
$keepZip = $false

function Download-IfMissing {
    param(
        [string]$Url,
        [string]$OutFile
    )

    $parent = Split-Path $OutFile -Parent
    New-Item -ItemType Directory -Force -Path $parent | Out-Null

    if (Test-Path $OutFile) {
        Write-Host "[SKIP] $OutFile already exists"
        return
    }

    $partFile = "$OutFile.part"

    Write-Host "[DOWNLOAD] $Url"
    curl.exe -L --fail --retry 5 --retry-delay 5 --continue-at - --output $partFile $Url

    Move-Item -Force $partFile $OutFile
}

foreach ($subject in $subjects) {
    foreach ($seq in $seqs) {
        Write-Host ""
        Write-Host "=== Downloading S$subject Seq$seq ==="

        $seqRoot = [System.IO.Path]::Combine($root, "S$subject", "Seq$seq")
        $imageRoot = [System.IO.Path]::Combine($seqRoot, "imageSequence")

        New-Item -ItemType Directory -Force -Path $seqRoot, $imageRoot | Out-Null

        $remoteRoot = "$baseUrl/S$subject/Seq$seq"

        Download-IfMissing "$remoteRoot/annot.mat" "$seqRoot/annot.mat"
        Download-IfMissing "$remoteRoot/camera.calibration" "$seqRoot/camera.calibration"

        $existingVideos = @(Get-ChildItem $imageRoot -Filter "video_*.avi" -ErrorAction SilentlyContinue)

        if ($existingVideos.Count -ge 8) {
            Write-Host "[SKIP] S$subject Seq$seq VNect videos already extracted"
        } else {
            $zipPath = [System.IO.Path]::Combine($imageRoot, "vnect_cameras.zip")

            Download-IfMissing "$remoteRoot/imageSequence/vnect_cameras.zip" $zipPath

            Write-Host "[UNZIP] S$subject Seq$seq VNect cameras"
            Expand-Archive -Path $zipPath -DestinationPath $imageRoot -Force

            if (-not $keepZip) {
                Remove-Item $zipPath -Force
            }
        }

        Write-Host "[DONE] S$subject Seq$seq"
    }
}

Write-Host ""
Write-Host "MPI-INF-3DHP VNect subset finished."

Get-ChildItem $root -Recurse -Filter "video_*.avi" |
    Select-Object FullName, Length |
    Export-Csv "data/raw/mpi_inf_3dhp/manifest_vnect_videos.csv" -NoTypeInformation

Write-Host "Manifest written to data/raw/mpi_inf_3dhp/manifest_vnect_videos.csv"

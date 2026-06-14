param(
  [string]$Name = "colab_m4td_topview_pose_bundle_v1"
)

$ErrorActionPreference = "Stop"

$production = Resolve-Path (Join-Path $PSScriptRoot "..")
$repo = Resolve-Path (Join-Path $production "..")
$outputs = Join-Path $repo "outputs"
$stageRoot = Join-Path $outputs $Name
$stageProduction = Join-Path $stageRoot "production"
$zip = Join-Path $outputs "$Name.zip"

if (Test-Path $stageRoot) {
  Remove-Item -LiteralPath $stageRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $stageProduction | Out-Null
New-Item -ItemType Directory -Force -Path $outputs | Out-Null

foreach ($dir in @("edge", "tools", "assets3d")) {
  Copy-Item -LiteralPath (Join-Path $production $dir) `
    -Destination (Join-Path $stageProduction $dir) -Recurse -Force
}

New-Item -ItemType Directory -Force -Path (Join-Path $stageProduction "scripts") | Out-Null
foreach ($script in @(
  "colab_topview_pose_setup.sh",
  "colab_m4td_topview_shell.sh"
)) {
  Copy-Item -LiteralPath (Join-Path $production "scripts\$script") `
    -Destination (Join-Path $stageProduction "scripts\$script") -Force
}

New-Item -ItemType Directory -Force -Path (Join-Path $stageProduction "config") | Out-Null
Copy-Item -LiteralPath (Join-Path $production "config\m4td.json") `
  -Destination (Join-Path $stageProduction "config\m4td.json") -Force
Copy-Item -LiteralPath (Join-Path $production "config\m4td_sim.json") `
  -Destination (Join-Path $stageProduction "config\m4td_sim.json") -Force

New-Item -ItemType Directory -Force -Path (Join-Path $stageProduction "prompts") | Out-Null
Copy-Item -LiteralPath (Join-Path $production "prompts\m4td_vlm_system.txt") `
  -Destination (Join-Path $stageProduction "prompts\m4td_vlm_system.txt") -Force

$mediaSrc = Join-Path $production "_media\real_world_hard"
$mediaDst = Join-Path $stageProduction "_media\real_world_hard"
if (Test-Path $mediaSrc) {
  New-Item -ItemType Directory -Force -Path $mediaDst | Out-Null
  Copy-Item -Path (Join-Path $mediaSrc "*") -Destination $mediaDst -Recurse -Force
}

Get-ChildItem -LiteralPath $stageProduction -Recurse -Directory -Force |
  Where-Object { $_.Name -eq "__pycache__" -or $_.Name -eq "dist" } |
  Remove-Item -Recurse -Force

Get-ChildItem -LiteralPath $stageProduction -Recurse -File -Force |
  Where-Object {
    $_.Name -like "*.pyc" -or
    $_.Name -like "*-pose.pt" -or
    $_.Name -like "yolov8*-pose.pt" -or
    $_.Name -like "yolo11*-pose.pt"
  } |
  Remove-Item -Force

$readme = @"
# GRADIS M4TD Top-View Pose Colab Bundle

This is the current DJI Matrice 4TD validation bundle.

Pipeline:

YOLO person detector/tracker -> MMPose RTMPose/ViTPose -> temporal SkeletonEngine -> redaction/mannequin rendering

Colab shell:

````bash
unzip -q /content/$Name.zip -d /content/gradis_m4td
cd /content/gradis_m4td/production
bash scripts/colab_m4td_topview_shell.sh
````

Quality pass:

````bash
POSE_PRESET=vitpose-s bash scripts/colab_m4td_topview_shell.sh _media/real_world_hard/caviar_walk_by_shop_front.mpg
````

Direct command:

````bash
bash scripts/colab_topview_pose_setup.sh _media/real_world_hard/caviar_walk_by_shop_front.mpg --max-frames 160
````
"@
$readme | Set-Content -LiteralPath (Join-Path $stageRoot "README_COLAB_M4TD_TOPVIEW.md") -Encoding UTF8

if (Test-Path $zip) {
  Remove-Item -LiteralPath $zip -Force
}

Compress-Archive -Path (Join-Path $stageRoot "*") -DestinationPath $zip -Force
Write-Host "Built $zip"

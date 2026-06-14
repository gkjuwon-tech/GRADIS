param(
  [string]$Name = "colab_topview_pose_bundle_v1"
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

$excludeDirs = @("__pycache__", "dist", "core\_data")
$excludeFiles = @("*-pose.pt", "yolov8*-pose.pt", "yolo11*-pose.pt", "*.pyc", "_vlm_view.png")

Get-ChildItem -LiteralPath $production -Force | ForEach-Object {
  $name = $_.Name
  if ($excludeDirs -contains $name) {
    return
  }
  $dest = Join-Path $stageProduction $name
  if ($_.PSIsContainer) {
    Copy-Item -LiteralPath $_.FullName -Destination $dest -Recurse -Force
  } elseif (-not ($excludeFiles | Where-Object { $name -like $_ })) {
    Copy-Item -LiteralPath $_.FullName -Destination $dest -Force
  }
}

Get-ChildItem -LiteralPath $stageProduction -Recurse -Directory -Force |
  Where-Object { $_.Name -eq "__pycache__" -or $_.FullName -like "*\core\_data*" } |
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
# GRADIS Colab Top-View Pose Bundle

This bundle uses the migrated pipeline:

YOLO person detector/tracker -> MMPose RTMPose/ViTPose -> temporal SkeletonEngine -> mannequin renderer

Colab shell:

````bash
unzip -q /content/$Name.zip -d /content/gradis
cd /content/gradis/production
bash scripts/colab_topview_pose_setup.sh _media/real_world_hard/caviar_walk_by_shop_front.mpg --max-frames 120
````

Quality pass:

````bash
bash scripts/colab_topview_pose_setup.sh _media/real_world_hard/caviar_walk_by_shop_front.mpg --pose-preset vitpose-s --max-frames 120
````
"@
$readme | Set-Content -LiteralPath (Join-Path $stageRoot "README_COLAB_TOPVIEW.md") -Encoding UTF8

if (Test-Path $zip) {
  Remove-Item -LiteralPath $zip -Force
}

Compress-Archive -Path (Join-Path $stageRoot "*") -DestinationPath $zip -Force
Write-Host "Built $zip"

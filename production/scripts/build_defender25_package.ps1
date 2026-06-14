param(
  [switch]$IncludeLargeModels
)

$ErrorActionPreference = "Stop"

$production = Resolve-Path (Join-Path $PSScriptRoot "..")
$dist = Join-Path $production "dist"
$stage = Join-Path $dist "gradis_defender25_package"
$package = Join-Path $dist "gradis_defender25_package.gradispack"
$zip = Join-Path $dist "gradis_defender25_package.zip"

if (Test-Path $stage) {
  Remove-Item -LiteralPath $stage -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $stage | Out-Null
New-Item -ItemType Directory -Force -Path $dist | Out-Null

$dirs = @(
  "assets3d",
  "config",
  "core",
  "docs",
  "edge",
  "firmware",
  "prompts",
  "scripts",
  "tools"
)

foreach ($dir in $dirs) {
  $src = Join-Path $production $dir
  if (Test-Path $src) {
    Copy-Item -LiteralPath $src -Destination (Join-Path $stage $dir) -Recurse -Force
  }
}

Copy-Item -LiteralPath (Join-Path $production "README_DEFENDER25.md") -Destination $stage -Force

if ($IncludeLargeModels) {
  Write-Warning "IncludeLargeModels is deprecated; pose models are bootstrapped by scripts/colab_topview_pose_setup.sh."
}

$manifest = @{
  name = "gradis-defender25-production"
  built_at_utc = (Get-Date).ToUniversalTime().ToString("o")
  platform = "iFlight Defender 25 O4 4S HD"
  extension = ".gradispack"
  default_detector_model = "yolo11x.pt"
  default_pose_preset = "rtmpose-m"
  quality_pose_preset = "vitpose-s"
  includes_embedded_pose_models = $false
  model_bootstrap_script = "scripts/colab_topview_pose_setup.sh"
  entrypoints = @{
    core = "python core/server.py"
    edge = "python -m edge.agent --source rtsp --url rtsp://127.0.0.1:8554/defender25 --detector-model yolo11x.pt --pose-preset rtmpose-m --vlm ollama --vlm-model moondream"
    companion = "python firmware/companion/mavlink_companion.py --config config/defender25.json --master udp:127.0.0.1:14550"
  }
} | ConvertTo-Json -Depth 5

$manifest | Set-Content -LiteralPath (Join-Path $stage "manifest.json") -Encoding UTF8

Get-ChildItem -LiteralPath $stage -Recurse -Directory -Force |
  Where-Object { $_.Name -eq "__pycache__" } |
  Remove-Item -Recurse -Force

$runtimeData = Join-Path $stage "core\_data"
if (Test-Path $runtimeData) {
  Remove-Item -LiteralPath $runtimeData -Recurse -Force
}

Get-ChildItem -LiteralPath $stage -Recurse -File -Include "*-pose.pt","yolov8*-pose.pt","yolo11*-pose.pt" |
  Remove-Item -Force

if (Test-Path $zip) {
  Remove-Item -LiteralPath $zip -Force
}
if (Test-Path $package) {
  Remove-Item -LiteralPath $package -Force
}

$items = Get-ChildItem -LiteralPath $stage -Force
if (-not $items) {
  throw "Package stage is empty: $stage"
}
Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zip -Force
Move-Item -LiteralPath $zip -Destination $package -Force

Write-Host "Built $package"

# GRADIS PoC - DJI Mini 2 + Litchi local launcher (Windows)
#
# Litchi pushes video to:
#   rtmp://<this-laptop-lan-ip>:1935/gradis-mini2
#
# mediamtx receives RTMP on 1935 and exposes the same path as:
#   rtsp://127.0.0.1:8554/gradis-mini2
param(
  [string]$VlmModel = "qwen2.5vl:3b",
  [switch]$SeedOnly,
  [switch]$MirrorPhone
)

$ErrorActionPreference = "Stop"
$production = Resolve-Path (Join-Path $PSScriptRoot "..")
$repo = Resolve-Path (Join-Path $production "..")
$mediaMtx = Join-Path $repo "tools_bin\mediamtx\mediamtx.exe"
$scrcpy = Join-Path $repo "tools_bin\scrcpy-win64-v4.0\scrcpy.exe"

if (-not (Test-Path $mediaMtx)) {
  throw "mediamtx.exe not found at $mediaMtx"
}

$ip = (Get-NetIPAddress -AddressFamily IPv4 |
  Where-Object {
    $_.IPAddress -notlike "127.*" -and
    $_.PrefixOrigin -ne "WellKnown" -and
    $_.InterfaceAlias -notmatch "Loopback|vEthernet|VMware|VirtualBox"
  } |
  Select-Object -First 1 -ExpandProperty IPAddress)

Write-Host "[poc] 1/5 mediamtx (RTMP :1935 -> RTSP :8554)"
Start-Process powershell -ArgumentList "-NoExit", "-Command",
  "cd '$production'; & '$mediaMtx'"
Start-Sleep 2

Write-Host "[poc] 2/5 GRADIS Core (:8088)"
Start-Process powershell -ArgumentList "-NoExit", "-Command",
  "cd '$production'; python core\server.py"
Start-Sleep 2

Write-Host "[poc] 3/5 edge.agent (Mini 2 Litchi stream)"
Start-Process powershell -ArgumentList "-NoExit", "-Command",
  ("cd '$production'; python -m edge.agent --source rtsp " +
   "--url 'rtsp://127.0.0.1:8554/gradis-mini2' " +
   "--drone-id GRADIS-MINI2-LITCHI-POC --zone 'Mini 2 Litchi PoC Zone' " +
   "--lat 37.52860 --lon 126.96520 " +
   "--detector-model yolo11x.pt --pose-preset rtmpose-m --imgsz 1280 --tiles 1 --avatar " +
   "--vlm ollama --vlm-model $VlmModel --vlm-interval 2.0")
Start-Sleep 2

Write-Host "[poc] 4/5 Litchi companion (mission CSV + operator cards)"
$seedArg = if ($SeedOnly) { "--seed-only" } else { "" }
Start-Process powershell -ArgumentList "-NoExit", "-Command",
  "cd '$production'; python poc\litchi_companion.py $seedArg"

Write-Host "[poc] 5/5 alert listener"
Start-Process powershell -ArgumentList "-NoExit", "-Command",
  "cd '$production'; python poc\alert_listener.py"

if ($MirrorPhone) {
  if (-not (Test-Path $scrcpy)) {
    Write-Host "[poc] scrcpy not found, skipping phone mirror"
  } else {
    Write-Host "[poc] optional Android mirror via scrcpy"
    Start-Process $scrcpy -ArgumentList "--stay-awake", "--turn-screen-off"
  }
}

Write-Host ""
Write-Host "=========================================================="
Write-Host " GRADIS MINI 2 + LITCHI PoC UP"
Write-Host "  Dashboard : http://127.0.0.1:8088/"
Write-Host "  Litchi RTMP URL:"
Write-Host "    rtmp://${ip}:1935/gradis-mini2"
Write-Host "  Generated mission files:"
Write-Host "    $production\poc\outbox"
Write-Host "=========================================================="

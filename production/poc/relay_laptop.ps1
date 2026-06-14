# GRADIS PoC gateway - DJI Mini 2 + Litchi to RunPod
#
# The Android Litchi device pushes RTMP to this laptop:
#   rtmp://<laptop-lan-ip>:1935/gradis-mini2
#
# This script forwards local port 1935 to the RunPod mediamtx server and local
# port 8088 to the RunPod Core dashboard. Keep this window open during the demo.
param(
  [Parameter(Mandatory=$true)][string]$Pod,
  [int]$Port = 22,
  [switch]$SeedOnly,
  [switch]$MirrorPhone
)

$ErrorActionPreference = "Stop"
$production = Resolve-Path (Join-Path $PSScriptRoot "..")
$repo = Resolve-Path (Join-Path $production "..")
$scrcpy = Join-Path $repo "tools_bin\scrcpy-win64-v4.0\scrcpy.exe"

$ip = (Get-NetIPAddress -AddressFamily IPv4 |
  Where-Object {
    $_.IPAddress -notlike "127.*" -and
    $_.PrefixOrigin -ne "WellKnown" -and
    $_.InterfaceAlias -notmatch "Loopback|vEthernet|VMware|VirtualBox"
  } |
  Select-Object -First 1 -ExpandProperty IPAddress)

Write-Host "[relay] 1/4 SSH tunnel (Core 8088 + RTMP 1935)"
Start-Process ssh -ArgumentList "-p", "$Port", "-N",
  "-L", "8088:127.0.0.1:8088",
  "-L", "0.0.0.0:1935:127.0.0.1:1935",
  "$Pod"
Start-Sleep 4

Write-Host "[relay] 2/4 Litchi companion (Core through tunnel)"
$seedArg = if ($SeedOnly) { "--seed-only" } else { "" }
Start-Process powershell -ArgumentList "-NoExit", "-Command",
  "cd '$production'; python poc\litchi_companion.py --core http://127.0.0.1:8088 $seedArg"

Write-Host "[relay] 3/4 alert listener"
Start-Process powershell -ArgumentList "-NoExit", "-Command",
  "cd '$production'; python poc\alert_listener.py"

if ($MirrorPhone) {
  if (-not (Test-Path $scrcpy)) {
    Write-Host "[relay] scrcpy not found, skipping phone mirror"
  } else {
    Write-Host "[relay] 4/4 optional Android mirror via scrcpy"
    Start-Process $scrcpy -ArgumentList "--stay-awake", "--turn-screen-off"
  }
} else {
  Write-Host "[relay] 4/4 phone mirror skipped"
}

Write-Host ""
Write-Host "=========================================================="
Write-Host " GRADIS MINI 2 + LITCHI GATEWAY UP"
Write-Host "  Dashboard       : http://127.0.0.1:8088/  (SSH tunnel)"
Write-Host "  Litchi RTMP URL : rtmp://${ip}:1935/gradis-mini2"
Write-Host "  RunPod logs     : ssh -p $Port $Pod 'tail -f /workspace/agent.log'"
Write-Host "=========================================================="

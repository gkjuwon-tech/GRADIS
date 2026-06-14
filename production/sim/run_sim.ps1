# GRADIS 디지털 트윈 — 원클릭 기동
#
# 부팅 순서 (의존성 순):
#   1. GRADIS Core (관제 서버)            : 8088
#   2. ArduCopter SITL (진짜 비행 펌웨어)  : tcp 5760(companion용) / 5762(브리지용)
#   3. Webots (도시 + 드론 + 액터)         : 카메라 MJPEG 8881
#   4. edge.agent (스켈레톤 추출 + VLM 판단): MJPEG 구독, ollama 호출
#   5. mavlink_companion (우리 펌웨어 레이어): SITL EKF 수렴 후 이륙·순찰
#
# 사용:  cd production;  .\sim\run_sim.ps1
param(
  [string]$VlmModel = "qwen2.5vl:3b",
  [int]$SkipCompanionDelay = 0   # 디버그: companion 대기 생략
)

$ErrorActionPreference = "Stop"
$production = Resolve-Path (Join-Path $PSScriptRoot "..")
$sim = Join-Path $production "sim"

# ---- 사전 점검 --------------------------------------------------------------
$webots = "C:\Program Files\Webots\webots.exe"
if (-not (Test-Path $webots)) { throw "Webots 없음: $webots (README_SIM.md 참조)" }
$sitlExe = Join-Path $sim "sitl\ArduCopter.elf"
if (-not (Test-Path $sitlExe)) { throw "SITL 없음: $sitlExe" }
try {
  $tags = Invoke-RestMethod -Uri http://localhost:11434/api/tags -TimeoutSec 3
  $names = $tags.models | ForEach-Object { $_.name }
  Write-Host "[sim] ollama OK: $($names -join ', ')"
  if (-not ($names -like "$($VlmModel.Split(':')[0])*")) {
    Write-Warning "[sim] '$VlmModel' 모델이 없을 수 있음. ollama pull $VlmModel"
  }
} catch {
  Write-Warning "[sim] ollama 응답 없음 — VLM 판단 없이 추출만 동작합니다."
}

# ---- 1. Core ----------------------------------------------------------------
Write-Host "[sim] 1/5 GRADIS Core 시작..."
Start-Process powershell -ArgumentList "-NoExit", "-Command",
  "cd '$production'; python core\server.py"
Start-Sleep 2

# ---- 2. SITL ----------------------------------------------------------------
# home = 월드 원점 = DOCK-A (config/defender25.json과 일치해야 함)
Write-Host "[sim] 2/5 ArduCopter SITL 시작 (home=DOCK-A)..."
$sitlDir = Join-Path $sim "sitl"
Start-Process powershell -ArgumentList "-NoExit", "-Command",
  ("cd '$sitlDir'; .\ArduCopter.elf --model quad " +
   "--home 37.52860,126.96520,38,0 " +
   "--defaults copter.parm,gradis.parm")
Start-Sleep 4

# ---- 3. Webots --------------------------------------------------------------
Write-Host "[sim] 3/5 Webots 시작 (도시 월드)..."
Start-Process $webots -ArgumentList "--mode=realtime",
  (Join-Path $sim "worlds\gradis_city.wbt")

Write-Host "[sim]     카메라 스트림 대기 (http://127.0.0.1:8881/cam.mjpg)..."
$deadline = (Get-Date).AddMinutes(3)
while ((Get-Date) -lt $deadline) {
  try {
    $r = [System.Net.Sockets.TcpClient]::new("127.0.0.1", 8881)
    $r.Close(); break
  } catch { Start-Sleep 3 }
}

# ---- 4. Edge agent (스켈레톤 + VLM) ------------------------------------------
Write-Host "[sim] 4/5 edge.agent 시작 (YOLO detector + MMPose + $VlmModel)..."
Start-Process powershell -ArgumentList "-NoExit", "-Command",
  ("cd '$production'; python -m edge.agent --source rtsp " +
   "--url http://127.0.0.1:8881/cam.mjpg " +
   "--drone-id GRADIS-DEFENDER25-01 --zone 'Riverside Park Area 2 (SIM)' " +
   "--lat 37.52860 --lon 126.96520 " +
   "--detector-model yolo11x.pt --pose-preset rtmpose-m --imgsz 1280 --tiles 1 " +
   "--vlm ollama --vlm-model $VlmModel --vlm-interval 2.0")

# ---- 5. Companion (SITL EKF 수렴 대기 후) -------------------------------------
if (-not $SkipCompanionDelay) {
  Write-Host "[sim] 5/5 SITL EKF/GPS 수렴 대기 40초 후 companion 시작..."
  Start-Sleep 40
}
Start-Process powershell -ArgumentList "-NoExit", "-Command",
  ("cd '$production'; python firmware\companion\mavlink_companion.py " +
   "--master tcp:127.0.0.1:5760 --config config\defender25.json " +
   "--node-id GRADIS-DEFENDER25-01")

Write-Host ""
Write-Host "=========================================================="
Write-Host " GRADIS DIGITAL TWIN UP"
Write-Host "  관제 대시보드 : http://127.0.0.1:8088/"
Write-Host "  드론 카메라   : http://127.0.0.1:8881/cam.mjpg"
Write-Host "  시나리오     : t=40s 몸싸움 → 한 명 쓰러짐 → 미동 없음"
Write-Host "                (120초 주기 반복)"
Write-Host "=========================================================="

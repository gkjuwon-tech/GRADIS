#!/usr/bin/env python3
"""
GRADIS Alert Listener — 적발 시 노트북 윈도우 알림센터로 토스트를 쏜다.

Core의 SSE 스트림(/api/events)을 구독하다가 incident 이벤트가 오면:
  1. 윈도우 토스트 알림 (알림센터에 쌓임 + 소리)
  2. 콘솔에 한 줄 로그

의존성 0 — 토스트는 윈도우 내장 WinRT API를 PowerShell로 호출한다.
Core 주소가 localhost:8088이면 SSH 터널 너머 RunPod이든 로컬이든 동일하게 동작.

실행:  python poc/alert_listener.py            (기본 http://127.0.0.1:8088)
       python poc/alert_listener.py --min-risk 0.5
       python poc/alert_listener.py --test      (토스트 1발 시험 발사)
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 윈도우 내장 토스트 (모듈 설치 불필요). 제목/본문은 env로 전달해 따옴표 지옥 회피.
_TOAST_PS = r"""
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
$xml = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
$t = $xml.GetElementsByTagName('text')
$t.Item(0).AppendChild($xml.CreateTextNode($env:GRADIS_TOAST_TITLE)) | Out-Null
$t.Item(1).AppendChild($xml.CreateTextNode($env:GRADIS_TOAST_BODY)) | Out-Null
$toast = [Windows.UI.Notifications.ToastNotification]::new($xml)
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('GRADIS').Show($toast)
"""


def toast(title: str, body: str) -> None:
    env = dict(os.environ,
               GRADIS_TOAST_TITLE=title[:80],
               GRADIS_TOAST_BODY=body[:160])
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", _TOAST_PS],
            env=env, capture_output=True, timeout=15)
    except Exception as e:
        print(f"[alert] toast 실패({e}) — 콘솔 벨로 대체\a", flush=True)


def listen(core: str, min_risk: float) -> None:
    url = core.rstrip("/") + "/api/events"
    while True:
        try:
            print(f"[alert] SSE 연결: {url}", flush=True)
            req = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
            with urllib.request.urlopen(req, timeout=60) as r:
                event = None
                for raw in r:
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if line.startswith("event:"):
                        event = line.split(":", 1)[1].strip()
                    elif line.startswith("data:") and event == "incident":
                        try:
                            row = json.loads(line.split(":", 1)[1])
                        except json.JSONDecodeError:
                            continue
                        risk = float(row.get("risk") or 0)
                        if risk < min_risk:
                            continue
                        kind = row.get("kind", "Unknown")
                        loc = row.get("location", "")
                        clip = " (클립 도착)" if row.get("clip") else ""
                        title = f"GRADIS 적발: {kind}"
                        body = f"위험도 {risk:.0%} @ {loc}{clip} - 대시보드 확인"
                        print(f"[alert] {title} | {body}", flush=True)
                        toast(title, body)
                    elif line == "":
                        event = None
        except KeyboardInterrupt:
            print("\n[alert] bye.", flush=True)
            return
        except Exception as e:
            print(f"[alert] 연결 끊김({e}) - 3초 후 재연결", flush=True)
            time.sleep(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default="http://127.0.0.1:8088")
    ap.add_argument("--min-risk", type=float, default=0.0)
    ap.add_argument("--test", action="store_true", help="토스트 1발 시험 발사")
    args = ap.parse_args()

    if args.test:
        toast("GRADIS 알림 테스트",
              "이게 보이면 알림 경로 정상입니다. 출동 안 하셔도 됩니다.")
        print("[alert] 테스트 토스트 발사 - 알림센터 확인", flush=True)
        return
    listen(args.core, args.min_risk)


if __name__ == "__main__":
    main()

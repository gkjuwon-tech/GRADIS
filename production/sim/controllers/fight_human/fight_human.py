"""
GRADIS 디지털 트윈 — 휴머노이드 시나리오 액터.

FightHuman PROTO의 관절 모터를 구동해 실제 '몸짓'을 만든다.
드론 카메라가 이 몸짓을 보고 → YOLO가 스켈레톤을 뽑고 → VLM이 판단한다.
어디에도 가짜 트리거는 없다. 액터는 연기만 한다.

시나리오 (--role):
  aggressor : 펀치 연타 + 몸 기울이며 압박 (계속)
  victim    : 같이 응전하다 t=fall_at 에 뒤로 쓰러져 미동 없음 (낙상→의식소실)
  idle      : 가만히 서서 팔만 미세하게 (대조군 — 이 사람 보고 알람 뜨면 오탐)

controllerArgs 예: ["--role", "victim", "--fall-at", "40"]
"""
import math
import sys

from controller import Robot

robot = Robot()
ts = int(robot.getBasicTimeStep())

argv = sys.argv[1:]


def arg(name, default):
    if name in argv:
        return argv[argv.index(name) + 1]
    return default


role = arg("--role", "idle")
fall_at = float(arg("--fall-at", "40"))
cycle = float(arg("--cycle", "120"))   # 시나리오 반복 주기(초). 0=반복 없음

M = {n: robot.getDevice(n) for n in
     ["body_pitch", "sho_l", "sho_r", "elb_l", "elb_r",
      "hip_l", "hip_r", "kne_l", "kne_r"]}


def pose(**kw):
    for k, v in kw.items():
        M[k].setPosition(v)


def act_idle(t):
    sway = 0.04 * math.sin(t * 0.7)
    pose(body_pitch=0.02 * math.sin(t * 0.5),
         sho_l=0.05 + sway, sho_r=0.05 - sway,
         elb_l=-0.15, elb_r=-0.15,
         hip_l=0.0, hip_r=0.0, kne_l=0.0, kne_r=0.0)


def act_fight(t, phase=0.0):
    """펀치 연타: 어깨 전방 스윙(교차 위상) + 팔꿈치 신전 + 런지."""
    w = 2 * math.pi * 1.6                     # 1.6 Hz — 사람 펀치 템포
    swing = math.sin(w * t + phase)
    pose(
        sho_l=-1.2 + 0.9 * swing,             # 앞으로 휘두름
        sho_r=-1.2 - 0.9 * swing,             # 반대 위상
        elb_l=-0.9 + 0.7 * max(0.0, swing),   # 뻗을 때 팔꿈치 신전
        elb_r=-0.9 + 0.7 * max(0.0, -swing),
        body_pitch=0.18 + 0.10 * math.sin(w * t * 0.5),   # 전방 압박
        hip_l=0.25 * math.sin(w * t * 0.5),               # 스텝 흉내
        hip_r=-0.25 * math.sin(w * t * 0.5),
        kne_l=-0.2, kne_r=-0.2,
    )


def act_fall(t_since_fall):
    """0.7초에 걸쳐 뒤로 무너짐 → 바닥에 누워 미동 없음(DOWN)."""
    p = min(1.0, t_since_fall / 0.7)
    ease = p * p * (3 - 2 * p)               # smoothstep — 자연스러운 붕괴
    pose(
        body_pitch=-1.45 * ease,             # 뒤로 넘어감 (거의 수평)
        sho_l=-0.4 + 0.8 * ease, sho_r=-0.5 + 0.9 * ease,   # 팔 풀림
        elb_l=-0.3 * (1 - ease), elb_r=-0.25 * (1 - ease),
        hip_l=0.25 * ease, hip_r=0.18 * ease,
        kne_l=-0.35 * ease, kne_r=-0.3 * ease,
    )
    # 이후 미동 없음 — 모터 위치 유지가 곧 '죽은 듯 누움'


while robot.step(ts) != -1:
    t = robot.getTime()
    tc = t % cycle if cycle > 0 else t

    if role == "aggressor":
        if cycle > 0 and tc > fall_at + 12:
            act_idle(tc)          # 상대 쓰러진 뒤 잠잠 (현장 이탈 연기)
        else:
            act_fight(tc)
    elif role == "victim":
        if tc < fall_at:
            act_fight(tc, phase=math.pi)   # 응전 (반대 위상)
        else:
            act_fall(tc - fall_at)
    else:
        act_idle(tc)

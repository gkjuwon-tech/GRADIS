"""
Cognition — 장면 판단 두뇌. GRADIS의 '유일한 판단 레이어'.

신뢰 경계(trust boundary) 아키텍처(v3):
  - 생영상(raw)은 절대 기기를 떠나지 않는다. 로컬 Jetson Orin/지상 GPU에서 YOLO detector/tracker + MMPose가
    스켈레톤만 뽑고 원본 픽셀은 즉시 버린다(익명화 전담).
  - 스켈레톤 → 마네킹(아바타) 렌더 = 익명화 결과물(얼굴·몸·옷 0).
  - 이 '마네킹 프레임'만 비전 모델로 나간다. 모델이 보는 건 영원히 익명 마네킹뿐 —
    생사람은 단 한 프레임도 전송되지 않는다. 마네킹이 곧 방화벽이다.
  왜 이 구조인가: "로컬이라 안전하면 마네킹이 불필요하고, 로컬도 위험하면 raw를
  YOLO에 주는 것부터 모순"이라는 딜레마를, '기기 밖으로 나가는 유일한 표상=마네킹'
  으로 풀었다. 판단 모델(Gemini)이 클라우드여도 프라이버시가 깨지지 않는 이유.

룰 기반 휴리스틱이 아니라 장면+자세 맥락으로 판단하므로 "푸쉬업 vs 낙상",
"포옹 vs 몸싸움" 함정에 강하다. 빠른 클라우드 모델(Flash Lite)이라 자율비행
회피 루프와 섞어도 판단 지연으로 벽에 박을 일이 적다.

입력: avatar/redact 레이어가 만든 마네킹 프레임 (환경 O, 사람=회색 마네킹). 생얼굴 X.
출력(JSON 강제):
  {
    "navigation": {"obstacle_ahead": bool,
                   "action": "PROCEED|HOLD|ASCEND|DESCEND|YAW_LEFT|YAW_RIGHT|RTL",
                   "reason": str},
    "situation":  {"risk": 0.0..1.0, "kind": "none|fall|fight|medical|fall_risk|crowd|other",
                   "reason": str}
  }

백엔드(실제로 돈다, 목업 아님):
  - gemini : Google Generative Language API (기본 gemini-3.1-flash-lite). 프로덕션 단일 백엔드.
             클라우드라 콜드스타트 없음, 빠르고 정확. GEMINI_API_KEY(.env)에서 로드.
             업로드되는 건 익명 마네킹 프레임뿐 — 생영상은 절대 안 나간다.
  - none   : 키 없거나 끌 때. None 반환 → agent는 반사신경만으로 동작(안전).
"""
import json, base64, io, time, urllib.request, urllib.error, re

SYS_PROMPT = (
    "You are the on-board cognition system of an autonomous public-safety drone. "
    "The image is PRIVACY-REDACTED: the environment is real, but every human is "
    "replaced by a faceless gray mannequin figure (or a blurred silhouette with a "
    "yellow skeleton) that exactly mirrors their real body pose. You can NEVER see "
    "faces or identities — only body poses and the surrounding scene. Treat each "
    "mannequin as a real person in that exact posture. "
    "Assess two things from the aerial view:\n"
    "1) NAVIGATION: is there an obstacle the drone should avoid? what flight action?\n"
    "2) SITUATION: is any person in danger based on POSE and SCENE context "
    "(e.g. lying collapsed, fighting, leaning over a railing/ledge, crowd crush)?\n"
    "If a person is in danger, also give the approximate image location of the "
    "concern as normalized center [cx,cy] (0..1, top-left origin) so the drone "
    "can turn toward it.\n"
    "Respond ONLY with strict JSON, no prose:\n"
    '{"navigation":{"obstacle_ahead":false,"action":"PROCEED","reason":""},'
    '"situation":{"risk":0.0,"kind":"none","center":[0.5,0.5],"reason":""}}'
)

_TOOL_PROMPT = (
    "\n\nYou COMMAND the aircraft through tools. The drone flies its patrol "
    "mission by itself (autopilot on). Only intervene when the situation "
    "demands it, then hand control back. Available tools:\n{tool_lines}\n"
    'To use tools, add to your JSON: "tools":[{{"name":"...","args":{{...}}}}] '
    "(max 3 per response, [] if none). Typical play: spot something odd -> "
    '{{"name":"autopilot","args":{{"mode":"off"}}}} then goto/zoom/thermal; '
    "all clear -> autopilot on."
)

_ACTIONS = {"PROCEED", "HOLD", "ASCEND", "DESCEND", "YAW_LEFT", "YAW_RIGHT", "RTL"}
_KINDS = {"none", "fall", "fight", "medical", "fall_risk", "crowd", "other"}


def _jpeg_b64(pil_img, max_w=512, quality=85):
    """VLM 입력 인코딩. JPEG(q85)는 PNG 대비 3~5배 작다 = 인코딩+전송+모델
    전처리 전부 빨라진다. redact된 장면이라 무손실일 이유가 없다."""
    img = pil_img.convert("RGB")
    if img.width > max_w:
        img = img.resize((max_w, int(img.height * max_w / img.width)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _coerce(obj, tool_names=()):
    """모델 출력 JSON을 안전 스키마로 정규화."""
    nav = (obj or {}).get("navigation", {}) or {}
    sit = (obj or {}).get("situation", {}) or {}
    raw_tools = (obj or {}).get("tools", []) or []
    tools = []
    if isinstance(raw_tools, list):
        for t in raw_tools[:3]:
            if not isinstance(t, dict):
                continue
            name = str(t.get("name", "")).strip()
            if name in tool_names:
                args = t.get("args", {})
                tools.append({"name": name,
                              "args": args if isinstance(args, dict) else {}})
    action = str(nav.get("action", "PROCEED")).upper()
    if action not in _ACTIONS:
        action = "PROCEED"
    kind = str(sit.get("kind", "none")).lower()
    if kind not in _KINDS:
        kind = "other"
    try:
        risk = float(sit.get("risk", 0.0))
    except (TypeError, ValueError):
        risk = 0.0
    center = sit.get("center", [0.5, 0.5])
    try:
        cx, cy = float(center[0]), float(center[1])
        cx = max(0.0, min(1.0, cx)); cy = max(0.0, min(1.0, cy))
    except (TypeError, ValueError, IndexError):
        cx, cy = 0.5, 0.5
    return {
        "navigation": {
            "obstacle_ahead": bool(nav.get("obstacle_ahead", False)),
            "action": action,
            "reason": str(nav.get("reason", ""))[:240],
        },
        "situation": {
            "risk": max(0.0, min(1.0, risk)),
            "kind": kind,
            "center": [cx, cy],
            "reason": str(sit.get("reason", ""))[:240],
        },
        "tools": tools,
    }


def _extract_json(text):
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


class Cognition:
    def __init__(self, backend="none", model="gemini-3.1-flash-lite",
                 host="https://generativelanguage.googleapis.com",
                 api_key=None, timeout=30, toolbox=None):
        self.backend = backend
        self.model = model
        self.host = host.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout          # 클라우드 Flash Lite는 빠르다 → 짧게
        # 툴박스가 있으면 모델은 '지휘관' — 프롬프트에 툴 스펙 삽입
        self.tool_names = tuple(toolbox.names()) if toolbox else ()
        self.sys_prompt = SYS_PROMPT
        if toolbox:
            self.sys_prompt += _TOOL_PROMPT.format(
                tool_lines="\n".join(toolbox.spec_lines()))
        self.available = self._probe()

    # ---- backend probe ----
    def _probe(self):
        if self.backend == "none":
            return False
        try:
            if self.backend == "gemini":
                if not self.api_key:
                    print("[cognition] gemini 백엔드인데 api_key 없음 — "
                          ".env의 GEMINI_API_KEY 설정 필요.")
                return bool(self.api_key)
        except (urllib.error.URLError, OSError) as e:
            print(f"[cognition] {self.backend} 백엔드 응답 없음: {e}")
            return False
        return False

    def warmup(self):
        """루프 진입 전 모델을 RAM에 로드(콜드스타트 1회 비용을 미리 치른다)."""
        if not self.available:
            return False
        if self.backend == "gemini":
            return True   # 클라우드 — RAM 콜드스타트 없음(워밍업 API콜은 토큰 낭비)
        from PIL import Image
        try:
            self.infer(Image.new("RGB", (320, 180), (16, 16, 18)))
            return True
        except Exception:
            return False

    # ---- inference ----
    def infer(self, redacted_frame):
        """redact된 PIL 프레임 -> 정규화 dict. 실패 시 None."""
        if not self.available:
            return None
        img_b64 = _jpeg_b64(redacted_frame)
        t0 = time.time()
        try:
            if self.backend == "gemini":
                raw = self._gemini(img_b64)
            else:
                return None
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            print(f"[cognition] infer 실패: {e}")
            return None
        obj = _extract_json(raw)
        if obj is None:
            return None
        result = _coerce(obj, self.tool_names)
        result["_latency_ms"] = int((time.time() - t0) * 1000)
        result["_model"] = f"{self.backend}:{self.model}"
        return result

    def _gemini(self, img_b64):
        """Google Generative Language API (generateContent). 업로드는 익명 마네킹뿐.
        host 기본 https://generativelanguage.googleapis.com, 키는 ?key= 로 전달."""
        url = (f"{self.host}/v1beta/models/{self.model}:generateContent"
               f"?key={self.api_key}")
        payload = {
            "system_instruction": {"parts": [{"text": self.sys_prompt}]},
            "contents": [{"role": "user", "parts": [
                {"text": "Analyze this privacy-redacted aerial view. JSON only."},
                {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
            ]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 512,
                                 "responseMimeType": "application/json"},
        }
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read())
        cands = data.get("candidates", [])
        if not cands:
            return ""
        parts = (cands[0].get("content", {}) or {}).get("parts", []) or []
        return "".join(p.get("text", "") for p in parts)

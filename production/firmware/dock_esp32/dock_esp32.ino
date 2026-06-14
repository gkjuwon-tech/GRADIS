/*
 * GRADIS Dock — ESP32 도킹스테이션 펌웨어 (실제 하드웨어용)
 * ---------------------------------------------------------------------------
 * 도킹스테이션이 GRADIS 24/7 운영의 핵심이다. 드론이 배터리 30%에 복귀하면
 * 이놈이 충전시키고, 다른 드론을 내보내 구역 커버리지를 끊김 없이 유지한다.
 *
 * 이 펌웨어가 실제로 하는 일:
 *   - 드론 착륙 감지 (리드 스위치/IR)
 *   - 충전 릴레이 ON/OFF + 충전 전류 모니터(ACS712)
 *   - 우천 감지 시 방수 도어(서보) 닫기/열기
 *   - 상태 LED (대기/충전/오류)
 *   - WiFi로 GRADIS Core에 하트비트 POST (node_type="dock")
 *
 * 보드: ESP32 DevKit (Arduino-ESP32 core 2.x/3.x)
 * 라이브러리: WiFi, HTTPClient, ArduinoJson, ESP32Servo (모두 보드매니저/라이브러리매니저)
 *
 * 배선은 docs/WIRING.md 참조.
 */
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <ESP32Servo.h>

// ---- 설정 ----------------------------------------------------------------
const char* WIFI_SSID = "GRADIS-FIELD";
const char* WIFI_PASS = "change-me";
const char* CORE_URL  = "http://192.168.1.50:8088/api/heartbeat";
const char* NODE_ID   = "DOCK-A";

// ---- 핀맵 ----------------------------------------------------------------
const int PIN_DRONE_PRESENT = 27;  // 리드 스위치 (착륙=LOW)
const int PIN_RAIN          = 34;  // 빗물 센서 (아날로그, 낮을수록 젖음)
const int PIN_CURRENT       = 35;  // ACS712 충전 전류 (아날로그)
const int PIN_RELAY_CHARGE  = 26;  // 충전 릴레이
const int PIN_LED_IDLE      = 2;   // 파랑: 대기
const int PIN_LED_CHARGE    = 4;   // 노랑: 충전 (GRADIS 시그니처!)
const int PIN_LED_ERROR     = 15;  // 빨강: 오류
const int PIN_DOOR_SERVO    = 13;  // 방수 도어 서보

const int RAIN_WET_THRESH = 1500;  // ADC < 이값 = 비
const unsigned long HEARTBEAT_MS = 5000;

// ---- 전류 측정 보정 -------------------------------------------------------
// ACS712는 5V 센서다(출력 0~5V, 0A=2.5V). ESP32 ADC는 3.3V까지만 받으므로
// 반드시 분압기를 거쳐야 한다. 기본값: 10k/20k 분압 (5V→3.3V, 비율 0.66).
// 캘리브레이션: 무부하 상태에서 시리얼로 zero 오프셋 확인 후 ZERO_V 조정.
const float ADC_DIVIDER   = 0.66f;   // 분압비 (센서출력 × 이값 = ADC 입력)
const float ACS_ZERO_V    = 2.50f;   // 0A일 때 센서 출력 전압
const float ACS_V_PER_A   = 0.100f;  // ACS712-20A: 100mV/A
const int   ADC_SAMPLES   = 9;       // 멀티샘플 중앙값(스위칭 노이즈 제거)

// ---- 충전 보호 ------------------------------------------------------------
const float CHARGE_MIN_A      = 0.2f;   // 이 이상이면 '충전 중'
const float OVERCURRENT_A     = 6.0f;   // 이 이상이면 즉시 릴레이 차단 + ERR 래치
const unsigned long CHARGE_TIMEOUT_MS = 4UL * 3600UL * 1000UL; // 4h 초과 = 이상
const unsigned long WIFI_RETRY_BASE_MS = 5000;                  // 재접속 백오프

Servo doorServo;
bool doorOpen = true;
unsigned long lastHeartbeat = 0;
unsigned long chargeStartMs = 0;
unsigned long lastWifiAttempt = 0;
unsigned long wifiBackoffMs = WIFI_RETRY_BASE_MS;
enum DockState { IDLE, CHARGING, RAIN_CLOSED, ERR };
DockState state = IDLE;
const char* errReason = "";

// --------------------------------------------------------------------------
void setLeds(bool idle, bool charge, bool err) {
  digitalWrite(PIN_LED_IDLE, idle);
  digitalWrite(PIN_LED_CHARGE, charge);
  digitalWrite(PIN_LED_ERROR, err);
}

void setDoor(bool open) {
  if (open == doorOpen) return;
  doorServo.write(open ? 95 : 5);   // 95°=열림, 5°=닫힘
  doorOpen = open;
  Serial.printf("[dock] door %s\n", open ? "OPEN" : "CLOSED");
}

float readChargeAmps() {
  // 멀티샘플 중앙값 — 릴레이/모뎀 스위칭 노이즈가 만드는 가짜 스파이크 제거.
  // 단발 샘플로 과전류 차단을 걸면 노이즈 한 방에 충전이 끊긴다.
  int s[ADC_SAMPLES];
  for (int i = 0; i < ADC_SAMPLES; i++) {
    s[i] = analogRead(PIN_CURRENT);
    delayMicroseconds(200);
  }
  // 삽입 정렬(9개라 충분)
  for (int i = 1; i < ADC_SAMPLES; i++) {
    int key = s[i], j = i - 1;
    while (j >= 0 && s[j] > key) { s[j + 1] = s[j]; j--; }
    s[j + 1] = key;
  }
  float vAdc = s[ADC_SAMPLES / 2] * 3.3f / 4095.0f;   // ADC 입력 전압
  float vSensor = vAdc / ADC_DIVIDER;                  // 분압 역산 = 센서 출력
  return (vSensor - ACS_ZERO_V) / ACS_V_PER_A;         // 암페어
}

void connectWifi() {
  // 비차단 재접속 + 지수 백오프. 라우터가 죽었다고 도크 제어 루프(도어/충전/
  // 과전류 보호)까지 15초씩 멈추면 안 된다.
  if (WiFi.status() == WL_CONNECTED) { wifiBackoffMs = WIFI_RETRY_BASE_MS; return; }
  if (millis() - lastWifiAttempt < wifiBackoffMs) return;
  lastWifiAttempt = millis();
  wifiBackoffMs = min(wifiBackoffMs * 2, 60000UL);
  Serial.printf("[dock] WiFi reconnect (next retry in %lums)\n", wifiBackoffMs);
  WiFi.disconnect();
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
}

void sendHeartbeat(float amps, bool dronePresent) {
  if (WiFi.status() != WL_CONNECTED) return;   // 재접속은 loop()의 connectWifi가 담당
  StaticJsonDocument<256> doc;
  doc["node_id"]   = NODE_ID;
  doc["node_type"] = "dock";
  // '배터리' 칸을 도크 충전율 추정치로 재사용 (관제 화면에서 한눈에)
  doc["battery"]   = dronePresent ? min(100.0f, amps * 20.0f) : (float)0;
  doc["lat"]       = 37.5286;        // 도크 고정 좌표
  doc["lon"]       = 126.9652;
  const char* s = state == CHARGING ? "CHARGING" :
                  state == RAIN_CLOSED ? "RAIN-CLOSED" :
                  state == ERR ? "ERROR" :
                  dronePresent ? "DRONE-DOCKED" : "IDLE-READY";
  doc["state"] = s;

  String body; serializeJson(doc, body);
  HTTPClient http;
  http.begin(CORE_URL);
  http.addHeader("Content-Type", "application/json");
  int code = http.POST(body);
  Serial.printf("[dock] heartbeat -> %d  state=%s amps=%.2f\n", code, s, amps);
  http.end();
}

// --------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  // 11dB 감쇠 = 풀스케일 ~3.3V. 미설정 시 기본 풀스케일이 ~1.1V라
  // 분압된 ACS712 출력(~1.65V@0A)이 포화되어 전류값이 전부 틀린다.
  analogSetPinAttenuation(PIN_CURRENT, ADC_11db);
  analogSetPinAttenuation(PIN_RAIN, ADC_11db);
  pinMode(PIN_DRONE_PRESENT, INPUT_PULLUP);
  pinMode(PIN_RELAY_CHARGE, OUTPUT);
  pinMode(PIN_LED_IDLE, OUTPUT);
  pinMode(PIN_LED_CHARGE, OUTPUT);
  pinMode(PIN_LED_ERROR, OUTPUT);
  digitalWrite(PIN_RELAY_CHARGE, LOW);
  doorServo.attach(PIN_DOOR_SERVO);
  setDoor(true);
  setLeds(true, false, false);
  connectWifi();
  Serial.println("[dock] GRADIS dock online.");
}

void loop() {
  bool dronePresent = (digitalRead(PIN_DRONE_PRESENT) == LOW);
  int rain = analogRead(PIN_RAIN);
  bool wet = rain < RAIN_WET_THRESH;

  // ERR 래치: 과전류/충전이상은 사람이 점검하기 전까지 충전 재개 금지.
  // (하트비트는 계속 보낸다 — 관제가 ERROR 상태를 봐야 출동하니까)
  if (state == ERR) {
    digitalWrite(PIN_RELAY_CHARGE, LOW);
    setLeds(false, false, true);
    connectWifi();
    if (millis() - lastHeartbeat > HEARTBEAT_MS) {
      sendHeartbeat(0, dronePresent);
      lastHeartbeat = millis();
      Serial.printf("[dock] ERR latched: %s\n", errReason);
    }
    delay(200);
    return;
  }

  // 우천: 드론이 안에 있으면 도어 닫아 보호. 비행 중이면 착륙 유도(코어가 처리).
  if (wet && dronePresent) {
    setDoor(false);
    state = RAIN_CLOSED;
  } else if (!wet) {
    setDoor(true);
  }

  // 충전 제어 (도어 상태 무관 — 닫혀 있어도 접점 충전은 안전)
  float amps = 0;
  if (dronePresent) {
    digitalWrite(PIN_RELAY_CHARGE, HIGH);
    amps = readChargeAmps();

    // 과전류 → 즉시 차단 + 래치
    if (amps > OVERCURRENT_A) {
      digitalWrite(PIN_RELAY_CHARGE, LOW);
      state = ERR;
      errReason = "overcurrent";
      Serial.printf("[dock] !! OVERCURRENT %.2fA > %.1fA — relay cut\n",
                    amps, OVERCURRENT_A);
      return;
    }

    bool charging = amps > CHARGE_MIN_A;
    if (charging && chargeStartMs == 0) chargeStartMs = millis();
    if (!charging) chargeStartMs = 0;

    // 충전 타임아웃 — 4시간 넘게 전류가 흐르면 배터리/접점 이상 의심
    if (charging && millis() - chargeStartMs > CHARGE_TIMEOUT_MS) {
      digitalWrite(PIN_RELAY_CHARGE, LOW);
      state = ERR;
      errReason = "charge timeout";
      Serial.println("[dock] !! charge timeout — relay cut");
      return;
    }

    if (!doorOpen)      state = RAIN_CLOSED;
    else                state = charging ? CHARGING : IDLE;
  } else {
    digitalWrite(PIN_RELAY_CHARGE, LOW);
    chargeStartMs = 0;
    state = IDLE;
  }

  // LED
  if (state == CHARGING)  setLeds(false, true, false);
  else                    setLeds(true, false, false);

  // WiFi 유지(비차단) + 하트비트
  connectWifi();
  if (millis() - lastHeartbeat > HEARTBEAT_MS) {
    sendHeartbeat(amps, dronePresent);
    lastHeartbeat = millis();
  }
  delay(200);
}

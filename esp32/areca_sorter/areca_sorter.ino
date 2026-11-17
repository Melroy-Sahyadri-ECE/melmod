#include <WiFi.h>
#include <WebServer.h>
#include <ESP32Servo.h>

// ---------- WIFI (only used to receive ML detections) ----------
const char* AP_SSID = "ESP32-Servo";
const char* AP_PASS = "12345678";      // min 8 characters

// ---------- SERVO 1 (pin 18) ----------
const int SERVO1_PIN = 18;             // always 100% forward
const int SERVO1_STOP_US  = 1500;
const int SERVO1_RANGE_US = 500;
int TRIM1 = 0;

// ---------- SERVO 2 (pin 19) + HOME SENSOR (pin 21) ----------
// Sorting arm: waits at HOME and turns only when an arecanut is received over WiFi
//   http://192.168.4.1/sort?type=kole&n=5   (sent by receiver.py)
//   bette / pure ->   0 deg (HOME)
//   kole         -> +40 deg
//   gottu        -> -40 deg
// It stays at that position until the next arecanut is received.
const int SERVO2_PIN = 19;
const int HOME_PIN   = 21;             // switch/sensor to GND, LOW at initial position

const int STOP_US  = 1500;             // pulse that stops the servo (adjust if it creeps)
const int SPEED_US = 80;               // offset from stop: 80 -> 1580/1420 us
const float DEG_PER_SEC = 120.0;       // calibrated value
const float MOVE_ANGLE = 40.0;
const int DIR_SIGN = 1;                // change to -1 if forward/backward are reversed
const float RETURN_TIMEOUT_FACTOR = 2.0;  // safety: give up after 2x the normal move time

// ---------- DEFAULT BEHAVIOUR AT POWER-ON (no WiFi needed) ----------
const bool DEFAULT_STEPPER_RUN  = true;
const int  DEFAULT_STEPPER_DIR  = 0;   // 0 = CW, 1 = CCW

// How long a detection action is held before returning to defaults
const unsigned long DETECT_HOLD_MS = 3000;

// ---------- STEPPER (TB6600) ----------
const int STEP_PUL_PIN = 26;
const int STEP_DIR_PIN = 27;

const unsigned long STEPPER_ON_MS  = 1000;     // run time
const unsigned long STEPPER_OFF_MS = 1000;     // stop time
const unsigned int  STEP_HALF_PERIOD_US = 500; // pulse high/low time (smaller = faster)

Servo servo1, servo2;
WebServer server(80);

// ---------- STATE ----------
bool stepRun = DEFAULT_STEPPER_RUN;
int  stepDirWanted = DEFAULT_STEPPER_DIR;
int  stepDirActual = -1;               // -1 forces DIR pin to be written first time

bool stepperPhaseOn = true;
unsigned long phaseStartMs = 0;
unsigned long lastPulseUs = 0;
bool pulseHigh = false;

int  lastClass = 0;                    // last detected class (0 = none yet)
unsigned long lastDetectTime = 0;
bool detectActive = false;

// Servo 2 sorting state
unsigned long moveMs;                  // time for one 40 deg move
enum SortPhase { SORT_IDLE, SORT_RETURNING, SORT_MOVING };
SortPhase sortPhase = SORT_IDLE;
int  curPos = 0;                       // -1 = -40 deg, 0 = HOME, +1 = +40 deg
int  targetPos = 0;
int  movingTo = 0;
unsigned long sortStart = 0;
int  lastSortN = 0;                    // number of the last arecanut received
String lastSortType = "none";

// ---------- SERVO 1 ----------
void startServo1() {
  servo1.writeMicroseconds(SERVO1_STOP_US + TRIM1 + SERVO1_RANGE_US);
}

// ---------- SERVO 2 SORTING ----------
bool atHome() { return digitalRead(HOME_PIN) == LOW; }

void drive2(int dir) {                 // +1 forward, -1 backward, 0 = stop
  servo2.writeMicroseconds(STOP_US + DIR_SIGN * dir * SPEED_US);
}

// Outward 40 deg is timed from HOME, the way back stops at the HOME sensor
// (timeout is only a safety). A command received during a move is done right after it.
void updateServo2() {
  unsigned long elapsed = millis() - sortStart;
  switch (sortPhase) {
    case SORT_IDLE:
      if (targetPos != curPos) {
        if (curPos != 0) {             // first back to HOME
          drive2(-curPos);
          sortPhase = SORT_RETURNING;
          Serial.println("Back to home");
        } else {                       // HOME -> +40 / -40
          movingTo = targetPos;
          drive2(movingTo);
          sortPhase = SORT_MOVING;
          Serial.println(movingTo > 0 ? "Forward 40 (kole)" : "Backward 40 (gottu)");
        }
        sortStart = millis();
      }
      break;
    case SORT_RETURNING:
      if (atHome() || elapsed >= (unsigned long)(moveMs * RETURN_TIMEOUT_FACTOR)) {
        if (!atHome()) Serial.println("Home not found (timeout)");
        drive2(0);
        curPos = 0;
        sortPhase = SORT_IDLE;
        if (targetPos == 0) Serial.println("At HOME 0 (bette)");
      }
      break;
    case SORT_MOVING:
      if (elapsed >= moveMs) {
        drive2(0);
        curPos = movingTo;
        sortPhase = SORT_IDLE;
      }
      break;
  }
}

// ---------- STEPPER FUNCTIONS ----------
void setupStepper() {
  pinMode(STEP_PUL_PIN, OUTPUT);
  pinMode(STEP_DIR_PIN, OUTPUT);
  digitalWrite(STEP_PUL_PIN, LOW);
  digitalWrite(STEP_DIR_PIN, LOW);
}

// Runs 1 s, stops 1 s, repeats forever (non-blocking)
void updateStepper() {
  // Stopped by a detection (class 3): keep pulse pin low
  if (!stepRun) {
    digitalWrite(STEP_PUL_PIN, LOW);
    pulseHigh = false;
    return;
  }

  // Direction change (only between pulses)
  if (stepDirWanted != stepDirActual && !pulseHigh) {
    stepDirActual = stepDirWanted;
    digitalWrite(STEP_DIR_PIN, stepDirActual);
    delayMicroseconds(20);
  }

  // Switch between ON and OFF phases
  unsigned long now = millis();
  unsigned long phaseLen = stepperPhaseOn ? STEPPER_ON_MS : STEPPER_OFF_MS;
  if (now - phaseStartMs >= phaseLen) {
    stepperPhaseOn = !stepperPhaseOn;
    phaseStartMs = now;
    digitalWrite(STEP_PUL_PIN, LOW);
    pulseHigh = false;
  }

  // Generate continuous pulses only during the ON phase
  if (!stepperPhaseOn) return;
  unsigned long nowUs = micros();
  if (nowUs - lastPulseUs >= STEP_HALF_PERIOD_US) {
    lastPulseUs = nowUs;
    pulseHigh = !pulseHigh;
    digitalWrite(STEP_PUL_PIN, pulseHigh);
  }
}

// ---------- DETECTION ACTIONS ----------
// Edit this function to decide what each detected object does.
// (Servo 2 is moved by /sort, so detections only control the stepper.)
void applyDetection(int cls) {
  switch (cls) {
    case 1:  // Object type 1
      stepDirWanted = 0;           // stepper CW
      stepRun = true;
      break;
    case 2:  // Object type 2
      stepDirWanted = 1;           // stepper CCW
      stepRun = true;
      break;
    case 3:  // Object type 3
      stepRun = false;             // stepper stop
      break;
    default:
      return;
  }
  lastClass = cls;
  lastDetectTime = millis();
  detectActive = true;
}

void restoreDefaults() {
  stepRun = DEFAULT_STEPPER_RUN;
  stepDirWanted = DEFAULT_STEPPER_DIR;
  detectActive = false;
}

// ---------- WIFI HANDLERS ----------
// Laptop calls: http://192.168.4.1/detect?class=1   (1, 2 or 3)
void handleDetect() {
  if (!server.hasArg("class")) {
    server.send(400, "text/plain", "missing class");
    return;
  }
  int cls = server.arg("class").toInt();
  if (cls < 1 || cls > 3) {
    server.send(400, "text/plain", "class must be 1-3");
    return;
  }
  applyDetection(cls);
  Serial.print("Detection received: class ");
  Serial.println(cls);
  server.send(200, "text/plain", "OK");
}

// receiver.py calls: http://192.168.4.1/sort?type=kole&n=5
//   type = bette | pure -> 0 deg, kole -> +40 deg, gottu -> -40 deg
void handleSort() {
  String t = server.hasArg("type") ? server.arg("type") : "";
  t.toLowerCase();
  int pos;
  if (t == "bette" || t == "pure")  pos = 0;
  else if (t == "kole")             pos = +1;
  else if (t == "gottu")            pos = -1;
  else {
    server.send(400, "text/plain", "type must be bette, pure, kole or gottu");
    return;
  }
  targetPos = pos;
  lastSortType = t;
  lastSortN = server.hasArg("n") ? server.arg("n").toInt() : lastSortN + 1;
  Serial.print("Arecanut #"); Serial.print(lastSortN); Serial.print(" "); Serial.print(t);
  Serial.print(" -> "); Serial.print(pos * 40); Serial.println(" deg");
  server.send(200, "text/plain", "OK " + t + " " + String(pos * 40));
}

void handleStatus() {
  String s = "last_class=" + String(lastClass) +
             " servo2=" + String(curPos * 40) + "deg last=#" + String(lastSortN) + ":" + lastSortType +
             " home=" + String(atHome() ? "yes" : "no") +
             " stepper=" + String(stepRun ? "run" : "stop") +
             " dir=" + String(stepDirWanted ? "CCW" : "CW");
  server.send(200, "text/plain", s);
}

void setup() {
  Serial.begin(115200);
  pinMode(HOME_PIN, INPUT_PULLUP);

  // ---- Motors start FIRST, independent of WiFi ----
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  servo1.setPeriodHertz(50);
  servo2.setPeriodHertz(50);
  servo1.attach(SERVO1_PIN, 500, 2500);
  servo2.attach(SERVO2_PIN, 500, 2500);
  startServo1();

  moveMs = (unsigned long)((MOVE_ANGLE / DEG_PER_SEC) * 1000.0);
  Serial.print("Time per 40 deg move (ms): "); Serial.println(moveMs);
  Serial.println(atHome() ? "Starting at HOME" : "WARNING: not at HOME at power-on");
  drive2(0);                       // servo 2 waits at HOME for the first arecanut

  setupStepper();
  phaseStartMs = millis();
  stepperPhaseOn = true;           // stepper starts running immediately

  // ---- WiFi (non-blocking AP; nothing waits for a client) ----
  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID, AP_PASS);
  Serial.print("WiFi: "); Serial.println(AP_SSID);
  Serial.print("IP: http://"); Serial.println(WiFi.softAPIP());

  server.on("/detect", handleDetect);
  server.on("/sort", handleSort);
  server.on("/status", handleStatus);
  server.begin();
}

void loop() {
  server.handleClient();

  // Return to default behaviour if no new detection for a while
  if (detectActive && millis() - lastDetectTime > DETECT_HOLD_MS) {
    restoreDefaults();
  }

  updateServo2();
  updateStepper();
}
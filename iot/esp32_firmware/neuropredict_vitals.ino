/*
 * NeuroPredict — ESP32 vascular-sensor companion
 * -----------------------------------------------
 * Reads heart rate + blood-oxygen (SpO2) from a MAX30102 pulse-oximeter over
 * I2C and POSTs them as JSON to the NeuroPredict server's /iot/vitals endpoint.
 *
 * The server scores the vitals into a vascular-risk level and fuses them with
 * the latest MRI prediction into a combined risk (see iot/README.md).
 *
 * RESEARCH / EDUCATIONAL USE ONLY — not a medical device.
 *
 * Hardware:
 *   ESP32 dev board  +  MAX30102 breakout (I2C)
 *   MAX30102 VIN -> 3V3,  GND -> GND,  SDA -> GPIO21,  SCL -> GPIO22
 *
 * Libraries (install via Arduino Library Manager):
 *   - "SparkFun MAX3010x Pulse and Proximity Sensor Library"
 *   - ArduinoJson
 */

#include <Wire.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include "MAX30105.h"
#include "spo2_algorithm.h"

// ---- Configure these for your setup ----------------------------------------
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
// Server URL: the machine running `uvicorn webapp.main:app`. Use its LAN IP.
const char* SERVER_URL    = "http://192.168.1.100:8000/iot/vitals";
const char* DEVICE_ID     = "esp32-01";
const int   PATIENT_AGE   = 68;   // optional context; set 0 to omit
// ----------------------------------------------------------------------------

MAX30105 sensor;

// Buffers for the SpO2 / heart-rate algorithm.
uint32_t irBuffer[100];
uint32_t redBuffer[100];
const int32_t BUFFER_LENGTH = 100;

int32_t  spo2;
int8_t   validSPO2;
int32_t  heartRate;
int8_t   validHeartRate;

void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.print("\nConnected. IP: ");
  Serial.println(WiFi.localIP());
}

void setup() {
  Serial.begin(115200);
  delay(200);

  if (!sensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 not found. Check wiring/power.");
    while (true) { delay(1000); }
  }
  // Sensible defaults for finger pulse oximetry.
  sensor.setup(60, 4, 2, 100, 411, 4096);

  connectWiFi();
}

void postVitals(int hr, int spo2Pct) {
  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
  }

  StaticJsonDocument<256> doc;
  doc["device_id"]  = DEVICE_ID;
  doc["heart_rate"] = hr;
  doc["spo2"]       = spo2Pct;
  if (PATIENT_AGE > 0) {
    doc["age"] = PATIENT_AGE;
  }
  // systolic / diastolic are omitted unless you add a blood-pressure module.

  String body;
  serializeJson(doc, body);

  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");
  int code = http.POST(body);
  Serial.printf("POST %s -> %d\n", SERVER_URL, code);
  if (code > 0) {
    Serial.println(http.getString());
  }
  http.end();
}

void loop() {
  // Collect 100 samples (~4s at 25 Hz effective) for the SpO2 algorithm.
  for (int i = 0; i < BUFFER_LENGTH; i++) {
    while (!sensor.available()) {
      sensor.check();
    }
    redBuffer[i] = sensor.getRed();
    irBuffer[i]  = sensor.getIR();
    sensor.nextSample();
  }

  maxim_heart_rate_and_oxygen_saturation(
      irBuffer, BUFFER_LENGTH, redBuffer,
      &spo2, &validSPO2, &heartRate, &validHeartRate);

  if (validHeartRate && validSPO2 && heartRate > 0 && spo2 > 0) {
    Serial.printf("HR=%d bpm  SpO2=%d%%\n", heartRate, spo2);
    postVitals(heartRate, spo2);
  } else {
    Serial.println("Reading invalid (keep finger steady on sensor)...");
  }

  delay(3000);  // send roughly every few seconds
}

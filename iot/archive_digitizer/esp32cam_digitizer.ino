/*
 * NeuroPredict — Archive MRI Digitizer (ESP32-CAM firmware)
 * --------------------------------------------------------
 * Photographs a printed MRI film sheet (a grid of slices) and uploads the JPEG
 * to the NeuroPredict server's /ingest/film endpoint. The server reconstructs
 * the slices into a 3D volume and runs the normal prediction.
 *
 * This is an ASSISTIVE device: it only helps the website obtain its usual input
 * (a scan). It does not add a new prediction signal.
 *
 * Board: AI-Thinker ESP32-CAM (select "AI Thinker ESP32-CAM" in Arduino IDE).
 * Wiring:
 *   - Push button between GPIO13 and GND (capture trigger; uses internal pullup).
 *   - The on-board flash LED (GPIO4) briefly lights the film during capture.
 *   - Power the board from a stable 5V supply (the camera browns out on weak USB).
 * Libraries: bundled with the ESP32 Arduino core (esp_camera, WiFi, HTTPClient).
 *
 * Set WIFI_SSID / WIFI_PASS / SERVER_URL below, then flash with an FTDI adapter
 * (GPIO0 -> GND to enter flashing mode).
 */

#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>

// ---- User configuration ----------------------------------------------------
const char *WIFI_SSID = "YOUR_WIFI_SSID";
const char *WIFI_PASS = "YOUR_WIFI_PASSWORD";
// Point this at the machine running `uvicorn webapp.main:app --port 8000`.
const char *SERVER_URL = "http://192.168.1.100:8000/ingest/film";

// The slice grid printed on your film sheet, and how many slices to keep.
const int SHEET_COLS = 8;
const int SHEET_DEPTH = 64;
const float PATIENT_AGE = 55.0f;  // optional context sent with the upload

const int BUTTON_PIN = 13;
const int FLASH_PIN = 4;

// ---- AI-Thinker ESP32-CAM pin map ------------------------------------------
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  // A film sheet has fine detail, so capture at high resolution when PSRAM is present.
  config.frame_size = psramFound() ? FRAMESIZE_UXGA : FRAMESIZE_SVGA;
  config.jpeg_quality = psramFound() ? 10 : 12;
  config.fb_count = psramFound() ? 2 : 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return false;
  }
  return true;
}

void connectWifi() {
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.printf("\nConnected. IP: %s\n", WiFi.localIP().toString().c_str());
}

// Upload one JPEG as multipart/form-data to /ingest/film.
void uploadFrame(camera_fb_t *fb) {
  if (WiFi.status() != WL_CONNECTED) {
    connectWifi();
  }

  String boundary = "----neuropredictcam";
  String head =
      "--" + boundary + "\r\n"
      "Content-Disposition: form-data; name=\"cols\"\r\n\r\n" + String(SHEET_COLS) + "\r\n"
      "--" + boundary + "\r\n"
      "Content-Disposition: form-data; name=\"depth\"\r\n\r\n" + String(SHEET_DEPTH) + "\r\n"
      "--" + boundary + "\r\n"
      "Content-Disposition: form-data; name=\"age\"\r\n\r\n" + String(PATIENT_AGE, 0) + "\r\n"
      "--" + boundary + "\r\n"
      "Content-Disposition: form-data; name=\"sheet\"; filename=\"film.jpg\"\r\n"
      "Content-Type: image/jpeg\r\n\r\n";
  String tail = "\r\n--" + boundary + "--\r\n";

  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "multipart/form-data; boundary=" + boundary);

  size_t totalLen = head.length() + fb->len + tail.length();
  uint8_t *payload = (uint8_t *)malloc(totalLen);
  if (!payload) {
    Serial.println("Out of memory for upload buffer");
    return;
  }
  memcpy(payload, head.c_str(), head.length());
  memcpy(payload + head.length(), fb->buf, fb->len);
  memcpy(payload + head.length() + fb->len, tail.c_str(), tail.length());

  int code = http.POST(payload, totalLen);
  if (code > 0) {
    Serial.printf("Server responded %d: %s\n", code, http.getString().c_str());
  } else {
    Serial.printf("Upload failed: %s\n", http.errorToString(code).c_str());
  }
  free(payload);
  http.end();
}

void setup() {
  Serial.begin(115200);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(FLASH_PIN, OUTPUT);
  digitalWrite(FLASH_PIN, LOW);

  if (!initCamera()) {
    Serial.println("Halting: camera unavailable.");
    while (true) delay(1000);
  }
  connectWifi();
  Serial.println("Ready. Place a film sheet on the light box and press the button.");
}

void loop() {
  if (digitalRead(BUTTON_PIN) == LOW) {  // pressed (active low)
    Serial.println("Capturing...");
    digitalWrite(FLASH_PIN, HIGH);
    delay(120);
    camera_fb_t *fb = esp_camera_fb_get();
    digitalWrite(FLASH_PIN, LOW);

    if (!fb) {
      Serial.println("Capture failed");
    } else {
      Serial.printf("Captured %u bytes, uploading...\n", fb->len);
      uploadFrame(fb);
      esp_camera_fb_return(fb);
    }
    delay(1500);  // simple debounce / avoid double-fire
  }
  delay(50);
}

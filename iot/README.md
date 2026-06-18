# NeuroPredict IoT companion (ESP32 vascular sensor)

A small, optional hardware add-on that **enhances** NeuroPredict: it streams a
patient's vascular vitals (heart rate, blood-oxygen) to the server, which scores
them into a vascular-risk level and **fuses** them with the MRI model's
prediction into a *combined risk*. The MRI software works perfectly on its own;
this just adds an extra, independent signal.

> Research / educational use only. This is **not** a medical device and must not
> be used for diagnosis or clinical decisions.

---

## How it fits together

```
  ┌──────────────┐   WiFi / HTTP POST    ┌─────────────────────┐
  │  ESP32 +     │  JSON {hr, spo2,...}  │  NeuroPredict server │
  │  MAX30102    │ ───────────────────►  │  POST /iot/vitals    │
  │ (this device)│                       │  GET  /iot dashboard │
  └──────────────┘                       └─────────────────────┘
                                              │  fuses with
                                              ▼
                                     latest MRI CNN prediction
                                       → combined risk score
```

---

## Parts list (~$15–25)

| Part | Notes |
|------|-------|
| ESP32 dev board (e.g. ESP32-WROOM DevKit v1) | Built-in WiFi; ~$6–10 |
| MAX30102 pulse-oximeter breakout | Heart rate + SpO₂ over I²C; ~$3–6 |
| 4 × jumper wires (female–female) | VIN, GND, SDA, SCL |
| Micro-USB / USB-C cable | Power + flashing (match your board) |
| *(optional)* blood-pressure cuff module | If you also want systolic/diastolic |

## Wiring (I²C)

| MAX30102 pin | ESP32 pin |
|--------------|-----------|
| VIN | 3V3 |
| GND | GND |
| SDA | GPIO21 |
| SCL | GPIO22 |

(GPIO21/22 are the ESP32's default I²C pins.)

---

## Build & flash the prototype

1. **Install the Arduino IDE** and add ESP32 board support
   (Boards Manager → search "esp32" → install).
2. **Install libraries** (Library Manager):
   - *SparkFun MAX3010x Pulse and Proximity Sensor Library*
   - *ArduinoJson*
3. Open `esp32_firmware/neuropredict_vitals.ino`.
4. Edit the config block at the top:
   - `WIFI_SSID`, `WIFI_PASSWORD`
   - `SERVER_URL` → `http://<your-server-LAN-IP>:8000/iot/vitals`
     (find it with `ip addr` / `ifconfig` on the machine running uvicorn)
   - `DEVICE_ID`, `PATIENT_AGE` (optional)
5. Select your board (e.g. "ESP32 Dev Module") and port, then **Upload**.
6. Open the Serial Monitor at **115200 baud**. Place a fingertip on the sensor;
   you should see `HR=.. SpO2=..` and `POST .../iot/vitals -> 200`.
7. Open **`http://<server>:8000/iot`** to watch readings + combined risk live.

---

## Test the whole pipeline **without** hardware

You can validate everything today using the simulator:

```bash
# 1) start the server
uvicorn webapp.main:app --port 8000

# 2) (optional) upload an MRI scan at http://localhost:8000 so the dashboard
#    has an MRI probability to fuse with

# 3) stream simulated vitals
python iot/simulate_vitals.py --profile at_risk      # elevated-risk vitals
python iot/simulate_vitals.py --profile healthy      # normal vitals
python iot/simulate_vitals.py --once                 # single reading

# 4) open http://localhost:8000/iot to see the dashboard update
```

When the real ESP32 is ready, it posts to the exact same endpoint — **no server
changes needed**.

---

## API reference

`POST /iot/vitals` — JSON body (all fields optional except `device_id`):

```json
{
  "device_id": "esp32-01",
  "heart_rate": 72,
  "spo2": 97,
  "systolic": 128,
  "diastolic": 82,
  "age": 68
}
```

Response:

```json
{
  "ok": true,
  "device_id": "esp32-01",
  "vascular_score": 0.0,
  "vascular_level": "low",
  "factors": ["no abnormal vascular risk factors detected"]
}
```

`GET /iot` — HTML dashboard (auto-refreshing) showing the latest reading per
device, its vascular-risk level, and — if an MRI scan has been analyzed this
session — the combined risk.

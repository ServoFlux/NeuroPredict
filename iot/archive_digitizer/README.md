# Archive MRI Digitizer (IoT companion)

A small camera station that **assists** the NeuroPredict website: it photographs
an archived MRI **film sheet** (a printed grid of slices) and uploads the photo.
The server reconstructs the slices into the 3D volume the website already uses
and runs the normal prediction.

> This device only helps the website obtain its usual input (a scan). It does
> **not** add a new prediction signal. Like the rest of the project it is a
> research/education demo — a photo of film is far lower fidelity than a native
> DICOM/NIfTI volume, so reconstructed scans are illustrative, not clinical.

## How it works

```
 film sheet (grid of slices)
        |  ESP32-CAM photographs it (button press, flash lights the film)
        v
 POST /ingest/film  (multipart: sheet=<jpeg>, cols, depth, age)
        |  server: volume_from_contact_sheet() splits the grid back into slices
        v
 3D volume -> existing preprocessing -> 3D CNN -> prediction (label + cause)
```

The conversion lives in `src/wmd/filmscan.py`; the endpoint is in
`webapp/main.py` (`/ingest/film` for the device, `/digitizer` for a browser).

## Try it without hardware

```bash
# 1. Start the app
uvicorn webapp.main:app --port 8000

# 2. Simulate a capture (renders a sample volume as a film sheet and posts it)
python iot/simulate_digitizer.py --etiology genetic --age 52

# Or save the film-sheet image and upload it via the browser at /digitizer
python iot/simulate_digitizer.py --etiology vascular --save sheet.png --no-post
```

The latest digitized result also shows up at `http://localhost:8000/digitizer`.

## Which board?

Two firmwares are included — pick the board you bought:

| | **Seeed XIAO ESP32-S3 Sense** (recommended) | **AI-Thinker ESP32-CAM** |
| --- | --- | --- |
| Firmware | `xiao_esp32s3_digitizer.ino` | `esp32cam_digitizer.ino` |
| Flashing | **USB-C, no extra hardware** | needs an FTDI adapter + GPIO0 jumper |
| PSRAM | 8MB (high-res capture) | varies |
| Price | ~$15–20 | ~$10 |
| Best for | first build / ease of use | lowest cost |

The rest of this guide covers the recommended **XIAO** path; the ESP32-CAM
wiring/flashing notes follow at the end.

## Parts list — XIAO ESP32-S3 Sense (~$40–60)

| Part | Search term | Notes |
| --- | --- | --- |
| Seeed XIAO ESP32-S3 Sense | `Seeed Studio XIAO ESP32S3 Sense` | OV2640 camera + WiFi + 8MB PSRAM, USB-C |
| USB-C cable | `USB-C cable` | power + flashing (skip if you have one) |
| Push button | `momentary push button switch kit` | capture trigger (D1/GPIO2 → GND) |
| LED light box / tracing pad (A4) | `A4 LED tracing light box pad` | even backlight so the film is readable |
| Camera stand / arm | `phone gooseneck clamp mount stand` | holds the camera over the film |
| Jumper wires (female-female) | `female to female jumper wires` | for the button |

No FTDI adapter and no Raspberry Pi are needed for this build.

### Legacy parts list — AI-Thinker ESP32-CAM (~$45–70)

| Part | Notes |
| --- | --- |
| ESP32-CAM (AI-Thinker) | ~$10; has the OV2640 camera + WiFi |
| FTDI USB-serial adapter (3.3V) | for flashing the ESP32-CAM |
| Push button | capture trigger (GPIO13 → GND) |
| LED light box / tracing pad | even backlight so the film is readable |
| Stand / arm to hold the camera over the film | 3D-printed or any phone arm |
| 5V power supply (stable) | the camera browns out on weak USB |

## Wiring (XIAO ESP32-S3 Sense)

Only one external connection is needed — the button. The film is lit by your
LED light box, and power/flashing both go over the single USB-C cable.

| XIAO pin | Connect to |
| --- | --- |
| **D1 (GPIO2)** | one leg of the push button |
| **GND** | the other leg of the push button |
| USB-C | your computer (flashing) or a USB charger (standalone use) |

The on-board user LED (GPIO21) blinks while a photo is captured. Make sure the
camera module is seated in the board's B2B connector and the antenna is attached.

## Flashing (XIAO ESP32-S3 Sense — USB-C, no FTDI)

1. In Arduino IDE, install the **ESP32 board package** (Boards Manager →
   "esp32" by Espressif), then select board **XIAO_ESP32S3** and set
   **PSRAM: OPI PSRAM** under Tools.
2. Open `xiao_esp32s3_digitizer.ino` and set `WIFI_SSID`, `WIFI_PASS`, and
   `SERVER_URL` (e.g. `http://<your-computer-ip>:8000/ingest/film`).
3. Set `SHEET_COLS` and `SHEET_DEPTH` to match how your film is laid out.
4. Plug the board into your computer with USB-C and click **Upload**.
   - If upload doesn't start, put the board in bootloader mode: hold **BOOT**,
     tap **RESET**, release **BOOT** (or just double-tap **RESET**), then upload.
5. Open Serial Monitor (115200 baud) to see the WiFi IP and upload responses.

## Wiring & flashing (legacy AI-Thinker ESP32-CAM)

| ESP32-CAM | Connect to |
| --- | --- |
| GPIO13 | push button → other leg to **GND** (internal pull-up used) |
| GPIO4 | on-board flash LED (already wired) — lights the film during capture |
| 5V / GND | stable 5V supply |
| U0T / U0R / GND / 5V + GPIO0→GND | FTDI adapter (GPIO0 to GND **only while flashing**) |

1. Install the **ESP32 board package** in Arduino IDE and select
   **AI Thinker ESP32-CAM**.
2. Open `esp32cam_digitizer.ino` and set `WIFI_SSID`, `WIFI_PASS`, and
   `SERVER_URL` (e.g. `http://<your-computer-ip>:8000/ingest/film`).
3. Set `SHEET_COLS` and `SHEET_DEPTH` to match how your film is laid out.
4. Connect the FTDI adapter, jumper **GPIO0 → GND**, press reset, and upload.
5. Remove the GPIO0 jumper, press reset. Open Serial Monitor (115200 baud) to
   see the WiFi IP and upload responses.

## API reference

`POST /ingest/film` — multipart/form-data

| field | type | description |
| --- | --- | --- |
| `sheet` | file | photo of the film sheet (JPEG/PNG) |
| `cols` | int | slices per row on the sheet (default 8) |
| `depth` | int | number of slices to keep (default 64) |
| `age` | float | optional patient age |
| _clinical fields_ | optional | same names as the questionnaire (e.g. `hypertension=1`) |

Response:

```json
{
  "ok": true,
  "label": "genetic",
  "label_pretty": "Genetic (e.g. CADASIL / CARASIL)",
  "confidence": 98.7,
  "wmd_probability": 99.9,
  "probabilities": {"no_wmd": 0.1, "vascular": 0.4, "genetic": 98.7, ...}
}
```

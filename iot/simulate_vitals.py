"""Simulate the ESP32 vascular-sensor companion without real hardware.

POSTs vitals readings to the NeuroPredict server's /iot/vitals endpoint so you
can run and demo the whole IoT pipeline (and the /iot dashboard) before any
hardware arrives. Swap in the real ESP32 later with zero server changes.

Usage:
    python iot/simulate_vitals.py                      # healthy-ish vitals
    python iot/simulate_vitals.py --profile at_risk    # elevated risk vitals
    python iot/simulate_vitals.py --once               # send a single reading
    python iot/simulate_vitals.py --url http://HOST:8000/iot/vitals
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.request

PROFILES = {
    # (heart_rate, spo2, systolic, diastolic, age) ranges
    "healthy": {
        "heart_rate": (60, 80),
        "spo2": (96, 99),
        "systolic": (110, 125),
        "diastolic": (70, 82),
        "age": (35, 50),
    },
    "at_risk": {
        "heart_rate": (95, 110),
        "spo2": (88, 93),
        "systolic": (150, 175),
        "diastolic": (95, 105),
        "age": (70, 82),
    },
}


def sample(profile: str, device_id: str) -> dict[str, float | str]:
    ranges = PROFILES[profile]
    return {
        "device_id": device_id,
        "heart_rate": round(random.uniform(*ranges["heart_rate"]), 0),
        "spo2": round(random.uniform(*ranges["spo2"]), 0),
        "systolic": round(random.uniform(*ranges["systolic"]), 0),
        "diastolic": round(random.uniform(*ranges["diastolic"]), 0),
        "age": round(random.uniform(*ranges["age"]), 0),
    }


def post(url: str, payload: dict[str, float | str]) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        print(f"sent {payload} -> {body}")
    except urllib.error.URLError as exc:
        print(f"failed to POST to {url}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/iot/vitals")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="healthy")
    parser.add_argument("--device-id", default="esp32-sim")
    parser.add_argument("--interval", type=float, default=3.0)
    parser.add_argument("--once", action="store_true", help="send one reading and exit")
    args = parser.parse_args()

    if args.once:
        post(args.url, sample(args.profile, args.device_id))
        return

    print(f"Streaming '{args.profile}' vitals to {args.url} every {args.interval}s. Ctrl+C to stop.")
    try:
        while True:
            post(args.url, sample(args.profile, args.device_id))
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()

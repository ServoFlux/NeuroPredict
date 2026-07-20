from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import webapp.main as web
from fastapi.testclient import TestClient

def test_cleanup_old_previews_removes_only_stale(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(web, "PREVIEW_DIR", tmp_path)
    fresh = tmp_path / "fresh.png"
    stale = tmp_path / "stale.png"
    fresh.write_bytes(b"x")
    stale.write_bytes(b"x")
    old = time.time() - 10_000
    import os

    os.utime(stale, (old, old))

    web._cleanup_old_previews(max_age_seconds=600)

    assert fresh.exists()
    assert not stale.exists()

class _FakeRequest:
    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers

def test_ingest_key_open_when_unset(monkeypatch) -> None:
    monkeypatch.setattr(web, "INGEST_API_KEY", None)
    assert web._ingest_key_ok(_FakeRequest({}), {}) is True

def test_ingest_key_requires_match_when_set(monkeypatch) -> None:
    monkeypatch.setattr(web, "INGEST_API_KEY", "s3cret")
    assert web._ingest_key_ok(_FakeRequest({}), {}) is False
    assert web._ingest_key_ok(_FakeRequest({"x-api-key": "wrong"}), {}) is False
    assert web._ingest_key_ok(_FakeRequest({"x-api-key": "s3cret"}), {}) is True
    assert web._ingest_key_ok(_FakeRequest({}), {"api_key": "s3cret"}) is True

def test_ingest_endpoint_rejects_bad_key(monkeypatch) -> None:
    if web.predictor is None:
        return
    monkeypatch.setattr(web, "INGEST_API_KEY", "s3cret")
    client = TestClient(web.app)
    resp = client.post("/ingest/film", headers={"X-API-Key": "nope"}, files={})
    assert resp.status_code == 401
    resp = client.post("/ingest/film", headers={"X-API-Key": "s3cret"}, files={})
    assert resp.status_code == 400

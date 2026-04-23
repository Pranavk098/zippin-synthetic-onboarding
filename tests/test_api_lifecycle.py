"""
API lifecycle test — validates HTTP contract and schema, not pipeline execution.
run_pipeline is monkeypatched to return immediately.
"""
import io
import pytest
from fastapi.testclient import TestClient


MOCK_METRICS = {
    "map50": 0.55,
    "map50_95": 0.31,
    "n_images": 5,
    "n_detections": 8,
    "mean_confidence": 0.72,
    "failure_gallery_dir": "",
    "eval_mode": "proxy_only",
    "eval_mode_note": "No GT annotations — confidence-based proxy.",
}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import src.api.server as srv
    from src.utils.sku_registry import SKURegistry

    # Redirect checkpoint dir and registry to tmp_path so tests are isolated
    monkeypatch.setattr(srv, "_CHECKPOINT_DIR", str(tmp_path))
    monkeypatch.setattr(srv, "_registry", SKURegistry(str(tmp_path / "sku_registry.json")))

    # Replace the background pipeline task with a sync mock that completes instantly
    async def _mock_pipeline_task(job_id: str, sku_name: str, image_path: str) -> None:
        srv._registry.update(
            job_id,
            status="complete",
            stage=None,
            weights_path=str(tmp_path / f"{job_id}_weights.pt"),
            ewc_path=str(tmp_path / "ewc_state.pt"),
            report_path=str(tmp_path / job_id / "run_report.md"),
            metrics=MOCK_METRICS,
        )

    monkeypatch.setattr(srv, "_run_pipeline_task", _mock_pipeline_task)

    return TestClient(srv.app)


def test_onboard_returns_202_with_job_id(client):
    fake_image = io.BytesIO(b"fake_jpeg_bytes")
    resp = client.post(
        "/onboard?sku_name=TestSKU",
        files={"image": ("product.jpg", fake_image, "image/jpeg")},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert "job_id" in body
    assert body["sku_name"] == "TestSKU"
    assert body["status"] in {"queued", "running", "complete"}


def test_job_status_endpoint(client):
    fake_image = io.BytesIO(b"fake_jpeg_bytes")
    job_id = client.post(
        "/onboard?sku_name=TestSKU",
        files={"image": ("product.jpg", fake_image, "image/jpeg")},
    ).json()["job_id"]

    resp = client.get(f"/jobs/{job_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == job_id
    assert body["status"] == "complete"


def test_metrics_endpoint_returns_eval_mode(client):
    fake_image = io.BytesIO(b"fake_jpeg_bytes")
    job_id = client.post(
        "/onboard?sku_name=TestSKU",
        files={"image": ("product.jpg", fake_image, "image/jpeg")},
    ).json()["job_id"]

    resp = client.get(f"/skus/{job_id}/metrics")
    assert resp.status_code == 200, (
        f"Expected 200 — background task runs synchronously in TestClient, got {resp.status_code}"
    )
    body = resp.json()
    assert "map50" in body
    assert "eval_mode" in body


def test_unknown_job_returns_404(client):
    resp = client.get("/jobs/doesnotexist")
    assert resp.status_code == 404

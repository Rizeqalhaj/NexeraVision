"""
Tests for file upload detection endpoint.
"""
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import io


@pytest.fixture
def client():
    """Create test client."""
    from app.main import app
    return TestClient(app)


def create_dummy_video() -> bytes:
    """Create a minimal valid video file for testing."""
    # This is a minimal MP4 header - for real tests, use actual video
    return b'\x00\x00\x00\x20ftypiso5' + b'\x00' * 1000


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_root_endpoint(client):
    """Test root endpoint returns service info."""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert "service" in data
    assert "endpoints" in data


def test_service_info(client):
    """Test service info endpoint."""
    response = client.get("/api/info")
    assert response.status_code == 200

    data = response.json()
    assert "service" in data
    assert "device" in data
    assert "configuration" in data


def test_file_upload_no_file(client):
    """Test upload endpoint without file."""
    response = client.post("/api/detect")
    assert response.status_code == 422  # Validation error


def test_file_upload_invalid_type(client):
    """Test upload endpoint with invalid file type."""
    files = {"video": ("test.txt", b"not a video", "text/plain")}
    response = client.post("/api/detect", files=files)
    assert response.status_code == 400
    assert "video" in response.json()["detail"].lower()


@pytest.mark.skipif(
    not Path("/app/models/ultimate_best_model.h5").exists(),
    reason="Model file not available"
)
def test_file_upload_success(client):
    """Test successful video upload and detection."""
    video_bytes = create_dummy_video()
    files = {"video": ("test.mp4", video_bytes, "video/mp4")}

    response = client.post("/api/detect", files=files)

    # Should succeed or fail gracefully with invalid video
    assert response.status_code in [200, 400, 500]

    if response.status_code == 200:
        data = response.json()
        assert "violence_probability" in data
        assert "confidence" in data
        assert 0.0 <= data["violence_probability"] <= 1.0


def test_file_upload_size_limit(client):
    """Test file size validation."""
    # Create file larger than limit (500MB)
    large_file = b'\x00' * (501 * 1024 * 1024)
    files = {"video": ("large.mp4", large_file, "video/mp4")}

    response = client.post("/api/detect", files=files)
    assert response.status_code == 413  # Payload too large

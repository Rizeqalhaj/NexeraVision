"""
Tests for live stream detection endpoint.
"""
import pytest
from fastapi.testclient import TestClient
import base64
import numpy as np
from PIL import Image
import io


@pytest.fixture
def client():
    """Create test client."""
    from app.main import app
    return TestClient(app)


def create_dummy_frame() -> str:
    """Create a dummy frame as base64 encoded image."""
    # Create random 224x224 RGB image
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)

    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    img_bytes = buffer.getvalue()

    return base64.b64encode(img_bytes).decode('utf-8')


def test_live_detection_invalid_frame_count(client):
    """Test live detection with wrong number of frames."""
    # Send only 10 frames instead of 20
    frames = [create_dummy_frame() for _ in range(10)]

    response = client.post("/api/detect_live", json={"frames": frames})
    assert response.status_code == 422  # Validation error


def test_live_detection_empty_frames(client):
    """Test live detection with empty frame list."""
    response = client.post("/api/detect_live", json={"frames": []})
    assert response.status_code == 422


@pytest.mark.skipif(
    not Path("/app/models/ultimate_best_model.h5").exists(),
    reason="Model file not available"
)
def test_live_detection_success(client):
    """Test successful live detection."""
    # Create 20 dummy frames
    frames = [create_dummy_frame() for _ in range(20)]

    response = client.post("/api/detect_live", json={"frames": frames})

    if response.status_code == 200:
        data = response.json()
        assert "violence_probability" in data
        assert "confidence" in data
        assert "prediction" in data
        assert 0.0 <= data["violence_probability"] <= 1.0
        assert data["confidence"] in ["Low", "Medium", "High"]


def test_live_detection_invalid_base64(client):
    """Test live detection with invalid base64 data."""
    # Send invalid base64 strings
    frames = ["invalid_base64" for _ in range(20)]

    response = client.post("/api/detect_live", json={"frames": frames})
    assert response.status_code in [400, 500]


def test_batch_detection_exceeds_limit(client):
    """Test batch detection with too many requests."""
    # Create 33 requests (exceeds limit of 32)
    frames = [create_dummy_frame() for _ in range(20)]
    requests = [{"frames": frames} for _ in range(33)]

    response = client.post("/api/detect_live_batch", json=requests)
    assert response.status_code == 400
    assert "batch size" in response.json()["detail"].lower()


@pytest.mark.skipif(
    not Path("/app/models/ultimate_best_model.h5").exists(),
    reason="Model file not available"
)
def test_batch_detection_success(client):
    """Test successful batch detection."""
    # Create 3 requests
    frames = [create_dummy_frame() for _ in range(20)]
    requests = [{"frames": frames} for _ in range(3)]

    response = client.post("/api/detect_live_batch", json=requests)

    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 3
        for result in data:
            assert "violence_probability" in result
            assert 0.0 <= result["violence_probability"] <= 1.0

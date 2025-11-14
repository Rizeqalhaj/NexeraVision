"""
Tests for ViolenceDetector model class.
"""
import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def model_path():
    """Get model path."""
    return "/app/models/ultimate_best_model.h5"


@pytest.mark.skipif(
    not Path("/app/models/ultimate_best_model.h5").exists(),
    reason="Model file not available"
)
def test_model_loading(model_path):
    """Test model loads successfully."""
    from app.models.violence_detector import ViolenceDetector

    detector = ViolenceDetector(model_path)
    assert detector.model is not None


def test_model_not_found():
    """Test error handling when model file doesn't exist."""
    from app.models.violence_detector import ViolenceDetector

    with pytest.raises(FileNotFoundError):
        ViolenceDetector("/nonexistent/model.h5")


@pytest.mark.skipif(
    not Path("/app/models/ultimate_best_model.h5").exists(),
    reason="Model file not available"
)
def test_prediction_shape_validation(model_path):
    """Test prediction validates input shape."""
    from app.models.violence_detector import ViolenceDetector

    detector = ViolenceDetector(model_path)

    # Wrong shape (missing frames dimension)
    invalid_frames = np.random.rand(224, 224, 3).astype(np.float32)

    with pytest.raises(ValueError):
        detector.predict(invalid_frames)


@pytest.mark.skipif(
    not Path("/app/models/ultimate_best_model.h5").exists(),
    reason="Model file not available"
)
def test_prediction_success(model_path):
    """Test successful prediction."""
    from app.models.violence_detector import ViolenceDetector

    detector = ViolenceDetector(model_path)

    # Create valid input
    frames = np.random.rand(20, 224, 224, 3).astype(np.float32)

    result = detector.predict(frames)

    assert "violence_probability" in result
    assert "confidence" in result
    assert "prediction" in result
    assert 0.0 <= result["violence_probability"] <= 1.0
    assert result["confidence"] in ["Low", "Medium", "High"]


@pytest.mark.skipif(
    not Path("/app/models/ultimate_best_model.h5").exists(),
    reason="Model file not available"
)
def test_batch_prediction(model_path):
    """Test batch prediction."""
    from app.models.violence_detector import ViolenceDetector

    detector = ViolenceDetector(model_path)

    # Create batch of 3 videos
    batch = np.random.rand(3, 20, 224, 224, 3).astype(np.float32)

    results = detector.predict_batch(batch)

    assert len(results) == 3
    for result in results:
        assert "violence_probability" in result
        assert 0.0 <= result["violence_probability"] <= 1.0


@pytest.mark.skipif(
    not Path("/app/models/ultimate_best_model.h5").exists(),
    reason="Model file not available"
)
def test_model_info(model_path):
    """Test model info retrieval."""
    from app.models.violence_detector import ViolenceDetector

    detector = ViolenceDetector(model_path)
    info = detector.get_model_info()

    assert "model_path" in info
    assert "input_shape" in info
    assert "output_shape" in info
    assert "total_layers" in info

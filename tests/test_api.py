"""
Unit Tests for FastAPI Backend API Endpoints (Phase 7)
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_check_endpoint():
    """Verify GET /health returns HTTP 200 and status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_localize_endpoint_match_found():
    """Verify POST /localize returns valid JSON schema when match is located."""
    mock_result = MagicMock()
    mock_result.match_found = True
    mock_result.timestamp = 601.53
    mock_result.timestamp_hms = "00:10:01.530"
    mock_result.frame_number = 14422
    mock_result.extracted_dialogue_text = "you cannot come in here unannounced like"
    mock_result.confidence = 91.6
    mock_result.frame_image_path = Path("output/frames/frame_601_53.jpg")
    mock_result.width = 1280
    mock_result.height = 720
    mock_result.pipeline_duration_seconds = 3.405

    with patch("api.routes.localize_dialogue", return_value=mock_result) as mock_pipeline:
        payload = {
            "video_source": "output/248244667877.mp4",
            "dialogue_query": "you cannot come in here unannounced like",
        }
        response = client.post("/localize", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["match_found"] is True
        assert data["timestamp"]["seconds"] == 601.53
        assert data["timestamp"]["formatted"] == "00:10:01.530"
        assert data["frame_number"] == 14422
        assert data["dialogue"] == "you cannot come in here unannounced like"
        assert data["confidence"] == 91.6
        assert "frame_601_53.jpg" in data["frame_path"]
        assert data["width"] == 1280
        assert data["height"] == 720
        assert data["pipeline_duration_seconds"] == 3.405

        mock_pipeline.assert_called_once_with(
            video_url_or_path="output/248244667877.mp4",
            dialogue_query="you cannot come in here unannounced like",
            output_dir="output",
            mode="v2",
        )


def test_localize_endpoint_no_match():
    """Verify POST /localize returns match_found=False when dialogue query is not found."""
    mock_result = MagicMock()
    mock_result.match_found = False

    with patch("api.routes.localize_dialogue", return_value=mock_result):
        payload = {
            "video_source": "output/248244667877.mp4",
            "dialogue_query": "nonexistent dialogue quote",
        }
        response = client.post("/localize", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["match_found"] is False
        assert data["reason"] == "dialogue_not_found"


def test_localize_endpoint_validation_empty_query():
    """Verify POST /localize returns HTTP 400 Bad Request on empty inputs."""
    payload = {"video_source": "output/sample.mp4", "dialogue_query": "   "}
    response = client.post("/localize", json=payload)
    assert response.status_code == 400
    assert "dialogue_query parameter cannot be empty" in response.json()["detail"]

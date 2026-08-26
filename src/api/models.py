"""
FastAPI Request and Response Models — Video Dialogue Localization API
"""

from typing import Optional, Union, Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class LocalizeRequest(BaseModel):
    video_source: str = Field(..., description="Public video URL or local file path")
    dialogue_query: str = Field(..., description="Target spoken dialogue phrase")


class TimestampDetail(BaseModel):
    seconds: float
    formatted: str


class LocalizeMatchResponse(BaseModel):
    match_found: bool = True
    timestamp: TimestampDetail
    frame_number: Optional[int]
    dialogue: str
    confidence: float
    frame_path: str
    width: int = 1280
    height: int = 720
    pipeline_duration_seconds: float


class LocalizeNoMatchResponse(BaseModel):
    match_found: bool = False
    reason: str = "dialogue_not_found"


# Flexible response type for endpoint
LocalizeResponse = Union[LocalizeMatchResponse, LocalizeNoMatchResponse]

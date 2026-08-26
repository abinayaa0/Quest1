"""
FastAPI Routes — Health Check & Dialogue Localization Endpoints
"""

import logging
from typing import Union
from fastapi import APIRouter, HTTPException

from pipeline import localize_dialogue
from .models import (
    HealthResponse,
    LocalizeRequest,
    LocalizeMatchResponse,
    LocalizeNoMatchResponse,
    TimestampDetail,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """Health check endpoint returning system status."""
    return HealthResponse(status="ok")


@router.post(
    "/localize",
    response_model=Union[LocalizeMatchResponse, LocalizeNoMatchResponse],
    tags=["Localization"],
)
def localize_endpoint(req: LocalizeRequest):
    """
    Expose Video Dialogue Localization pipeline over REST API.
    Calls existing localize_dialogue(..., mode='v2') internally.
    """
    if not req.video_source or not req.video_source.strip():
        raise HTTPException(status_code=400, detail="video_source parameter cannot be empty")
    if not req.dialogue_query or not req.dialogue_query.strip():
        raise HTTPException(status_code=400, detail="dialogue_query parameter cannot be empty")

    try:
        result = localize_dialogue(
            video_url_or_path=req.video_source.strip(),
            dialogue_query=req.dialogue_query.strip(),
            output_dir="output",
            mode="v2",
        )

        if not result.match_found:
            return LocalizeNoMatchResponse(
                match_found=False,
                reason="dialogue_not_found"
            )

        ts_sec = round(result.timestamp, 2) if result.timestamp is not None else 0.0
        frame_path_str = str(result.frame_image_path) if result.frame_image_path else ""

        return LocalizeMatchResponse(
            match_found=True,
            timestamp=TimestampDetail(
                seconds=ts_sec,
                formatted=result.timestamp_hms,
            ),
            frame_number=result.frame_number,
            dialogue=result.extracted_dialogue_text or "",
            confidence=round(result.confidence, 1),
            frame_path=frame_path_str,
            width=result.width,
            height=result.height,
            pipeline_duration_seconds=round(result.pipeline_duration_seconds, 3),
        )
    except Exception as e:
        logger.error(f"Error during dialogue localization API call: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

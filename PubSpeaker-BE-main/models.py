"""Pydantic models for request/response validation."""

from pydantic import BaseModel
from typing import List, Dict, Optional


class AnalyzeRequest(BaseModel):
    """Request model for speech analysis endpoint."""
    audio_path: Optional[str] = None
    transcript: Optional[str] = None
    words: Optional[List[Dict]] = None

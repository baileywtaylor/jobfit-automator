from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


class JobPosting(BaseModel):
    filename: str
    raw_text: str

    # AI-enriched fields (optional)
    title: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    must_haves: List[str] = Field(default_factory=list)
    nice_to_haves: List[str] = Field(default_factory=list)
    responsibilities: List[str] = Field(default_factory=list)
    summary: Optional[str] = None

    # scoring
    score: Optional[float] = None
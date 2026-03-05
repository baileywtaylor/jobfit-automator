from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class JobPosting(BaseModel):
    filename: str
    raw_text: str

    # Basic AI-enriched fields
    title: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None

    must_haves: List[str] = Field(default_factory=list)
    nice_to_haves: List[str] = Field(default_factory=list)
    responsibilities: List[str] = Field(default_factory=list)

    summary: Optional[str] = None

    # Advanced extracted features (V2)

    # Role classification
    role_level: Optional[str] = None           # graduate | entry | junior | mid | senior
    program_type: Optional[str] = None         # grad_program | standard_role

    # Work environment
    work_mode: Optional[str] = None            # onsite | hybrid | remote

    # Technology stack
    tech_stack: List[str] = Field(default_factory=list)

    # Growth / learning signals
    growth_signals: List[str] = Field(default_factory=list)

    # Security signals
    citizenship_required: Optional[bool] = None
    clearance_required: Optional[bool] = None

    # Contract / job structure
    contract_type: Optional[str] = None        # permanent | contract | internship | unknown

    # Risk signals
    risk_flags: List[str] = Field(default_factory=list)

    # Evidence snippets supporting extracted features
    evidence: Optional[Dict[str, Any]] = None

    # Scoring results

    score: Optional[float] = None

    # category breakdown (filled later by scoring engine)
    score_breakdown: Optional[Dict[str, float]] = None

    # computed skill gaps
    missing_skills: List[str] = Field(default_factory=list)
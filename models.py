from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class JobPosting(BaseModel):
    filename: str
    raw_text: str

    # AI strategic evaluation
    structural_fit_summary: Optional[str] = None
    strategic_fit_summary: Optional[str] = None

    recommendation: Optional[str] = None   # strong_apply | apply | stretch_apply | low_priority | skip

    program_quality: Optional[str] = None      # strong | medium | weak | unknown
    role_substance: Optional[str] = None       # strong | medium | weak | unknown
    learning_environment: Optional[str] = None # strong | medium | weak | unknown
    trajectory_value: Optional[str] = None     # strong | medium | weak | unknown
    employer_signal: Optional[str] = None      # strong | medium | weak | unknown
    gap_severity: Optional[str] = None         # low | medium | high | unknown

    strategic_reasons: List[str] = Field(default_factory=list)
    caution_reasons: List[str] = Field(default_factory=list)
    ai_evaluation_evidence: Dict[str, Any] = Field(default_factory=dict)

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
    tech_tools: List[str] = Field(default_factory=list)
    tech_domains: List[str] = Field(default_factory=list)
    candidate_required_tools: List[str] = Field(default_factory=list)
    candidate_preferred_tools: List[str] = Field(default_factory=list)
    role_exposure_tools: List[str] = Field(default_factory=list)
    role_exposure_domains: List[str] = Field(default_factory=list)

    # Growth / learning signals
    growth_signals: List[str] = Field(default_factory=list)

    # Security signals
    citizenship_required: Optional[bool] = None
    clearance_required: Optional[bool] = None
    university_restriction_present: Optional[bool] = None
    required_university: Optional[str] = None

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
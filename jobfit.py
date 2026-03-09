"""
JobFit Automator
- Paste a job ad into the CLI
- Optional AI extraction with caching
- Optional second-pass AI strategic evaluation with caching
- Deterministic scoring against profile preferences
- Writes output/results.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from dotenv import load_dotenv
from openai import OpenAI

from models import JobPosting


logger = logging.getLogger("jobfit")

SCHEMA_VERSION = "v11"
AI_MODEL = "gpt-4.1-mini"


# -----------------------------
# Logging
# -----------------------------
def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


# -----------------------------
# General helpers
# -----------------------------
def normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def normalize_token(value: str) -> str:
    return normalize(value).replace("-", " ")


def load_profile(profile_path: Path) -> Dict[str, Any]:
    if not profile_path.exists():
        raise FileNotFoundError(f"{profile_path} not found.")
    with profile_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def hash_job(text: str) -> str:
    return hashlib.sha256((text + SCHEMA_VERSION).encode("utf-8")).hexdigest()


def hash_eval(job_text: str, profile: Dict[str, Any], job: JobPosting | None = None) -> str:
    profile_fragment = json.dumps(
        {
            "preferred_locations": profile.get("target", {}).get("preferred_locations", []),
            "preferred_work_modes": profile.get("target", {}).get("preferred_work_modes", []),
            "role_levels": profile.get("target", {}).get("role_levels", []),
            "preferred_role_clusters": profile.get("target", {}).get("preferred_role_clusters", {}),
            "company_preferences": profile.get("target", {}).get("company_preferences", {}),
            "growth_preferences": profile.get("target", {}).get("growth_preferences", {}),
        },
        sort_keys=True,
    )

    extracted_fragment = {}
    if job is not None:
        extracted_fragment = {
            "title": job.title,
            "company": job.company,
            "location": job.location,
            "role_level": job.role_level,
            "program_type": job.program_type,
            "work_mode": job.work_mode,
            "contract_type": job.contract_type,
            "tech_tools": job.tech_tools,
            "tech_domains": job.tech_domains,
            "candidate_required_tools": job.candidate_required_tools,
            "candidate_preferred_tools": job.candidate_preferred_tools,
            "role_exposure_tools": job.role_exposure_tools,
            "role_exposure_domains": job.role_exposure_domains,
            "must_haves": job.must_haves,
            "nice_to_haves": job.nice_to_haves,
            "responsibilities": job.responsibilities,
            "growth_signals": job.growth_signals,
            "citizenship_required": job.citizenship_required,
            "clearance_required": job.clearance_required,
            "university_restriction_present": job.university_restriction_present,
            "required_university": job.required_university,
            "risk_flags": job.risk_flags,
            "summary": job.summary,
        }

    return hashlib.sha256(
        (
            job_text
            + profile_fragment
            + json.dumps(extracted_fragment, sort_keys=True)
            + SCHEMA_VERSION
            + "eval_v3"
        ).encode("utf-8")
    ).hexdigest()


def build_openai_client() -> OpenAI | None:
    load_dotenv(dotenv_path=".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def recommendation_label(value: str | None) -> str:
    labels = {
        "strong_apply": "Strong Apply",
        "apply": "Apply",
        "stretch_apply": "Stretch Apply",
        "low_priority": "Low Priority",
        "skip": "Skip",
    }
    return labels.get((value or "").lower(), "Unknown")


# -----------------------------
# Profile / normalization helpers
# -----------------------------
def apply_synonym_map(tokens: List[str], profile: Dict[str, Any]) -> List[str]:
    synonym_map = profile.get("normalization", {}).get("synonym_map", {}) or {}
    normalized: List[str] = []

    for token in tokens:
        clean = normalize_token(token)
        mapped = normalize_token(synonym_map.get(clean, clean))
        if mapped not in normalized:
            normalized.append(mapped)

    return normalized


def get_profile_skill_buckets(profile: Dict[str, Any]) -> Dict[str, set[str]]:
    skills = profile.get("skills", {}) or {}

    strong = set(apply_synonym_map(skills.get("strong", []) or [], profile))
    working = set(apply_synonym_map(skills.get("working", []) or [], profile))
    basic = set(apply_synonym_map(skills.get("basic", []) or [], profile))

    return {
        "strong": strong,
        "working": working,
        "basic": basic,
        "all": strong | working | basic,
    }

def get_target_interest_tokens(profile: Dict[str, Any]) -> set[str]:
    """
    Build a set of normalized interest/domain tokens from the user's target profile.

    This expands internal role-cluster labels like:
    - software_engineering
    - security_engineering
    - cloud_platform

    into real matching tokens such as:
    - engineering
    - cybersecurity
    - cloud
    """
    target = profile.get("target", {}) or {}
    clusters = target.get("preferred_role_clusters", {}) or {}
    growth_prefs = target.get("growth_preferences", {}) or {}

    tokens: set[str] = set()

    cluster_expansions = {
        "software engineering": {
            "software engineering", "engineering", "software", "systems", "testing", "build"
        },
        "graduate software": {
            "software engineering", "engineering", "software", "graduate software"
        },
        "automation engineering": {
            "automation", "engineering", "systems", "testing"
        },
        "security engineering": {
            "cybersecurity", "cyber security", "security", "cyber", "engineering"
        },
        "cyber rotation": {
            "cybersecurity", "cyber security", "cyber", "security"
        },
        "cloud platform": {
            "cloud", "cloud platforms", "platform", "platforms", "engineering"
        },
        "devops": {
            "devops", "cloud", "automation", "platform", "engineering", "ci/cd"
        },
        "platform engineering": {
            "platform", "platforms", "engineering", "cloud", "systems"
        },
        "security rotation": {
            "cybersecurity", "cyber security", "cyber", "security"
        },
        "software rotation": {
            "software engineering", "engineering", "software", "systems"
        },
    }

    def add_value(value: str) -> None:
        norm = normalize_token(value)
        tokens.add(norm)

        expanded = cluster_expansions.get(norm, set())
        for item in expanded:
            tokens.add(normalize_token(item))

    for _, values in clusters.items():
        if isinstance(values, list):
            for value in values:
                add_value(value)
        elif isinstance(values, str):
            add_value(values)

    for _, values in growth_prefs.items():
        if isinstance(values, list):
            for value in values:
                add_value(value)
        elif isinstance(values, str):
            add_value(values)

    # Also include direct skill/domain signals from the profile itself
    skills = profile.get("skills", {}) or {}
    for bucket in ("strong", "working", "basic"):
        for value in skills.get(bucket, []) or []:
            add_value(value)

    return tokens

# -----------------------------
# Deterministic scoring
# -----------------------------
def score_role_fit(job: JobPosting, profile: Dict[str, Any]) -> tuple[float, List[str]]:
    rules = profile.get("scoring_rules", {}).get("role_fit", {}) or {}
    max_score = float(profile.get("weights", {}).get("role_fit", 25))
    role_level = (job.role_level or "unknown").lower()

    score = float(rules.get(role_level, rules.get("unknown", 0)))
    reasons = [f"Role level detected as '{role_level}'"]

    if job.program_type == "grad_program":
        score += 2
        reasons.append("Graduate program structure detected")

    return min(round(score, 1), max_score), reasons

def classify_tech_item(token: str) -> str:
    """
    Classify a normalized tech token into:
    - language
    - tool
    - domain
    - unknown
    """
    t = normalize_token(token)

    language_keywords = {
        "python", "java", "c#", "csharp", "cpp", "c++", "javascript",
        "typescript", "sql", "go", "rust", "php", "ruby"
    }

    tool_keywords = {
        "aws", "azure", "google cloud", "gcp", "docker", "kubernetes",
        "terraform", "react", "node", "node.js", "git", "ci/cd",
        "selenium", "blueprism", "blue prism", "uipath", "robocorp",
        "power bi", "tableau", "snowflake", "databricks", "postgres",
        "mysql", "mongodb", "linux", "openai api", "rest api"
    }

    domain_keywords = {
        "artificial intelligence", "ai", "machine learning",
        "generative ai", "natural language processing", "nlp",
        "optical character recognition", "ocr", "data analytics",
        "data science", "cloud", "cyber security", "cybersecurity",
        "automation", "digital transformation", "software engineering",
        "process improvement", "robotic process automation", "rpa",
        "time series forecasting", "forecasting"
    }

    if t in language_keywords:
        return "language"
    if t in tool_keywords:
        return "tool"
    if t in domain_keywords:
        return "domain"

    if any(x in t for x in ["language", "programming"]):
        return "language"
    if any(x in t for x in [
        "cloud", "ai", "machine learning", "analytics", "automation",
        "security", "transformation", "forecasting", "natural language",
        "optical character"
    ]):
        return "domain"
    if any(x in t for x in ["api", "platform", "framework", "tool", "prism", "uipath", "selenium", "azure", "google cloud"]):
        return "tool"

    return "unknown"


def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out

def get_target_interest_tokens(profile: Dict[str, Any]) -> set[str]:
    target = profile.get("target", {}) or {}
    clusters = target.get("preferred_role_clusters", {}) or {}

    tokens: set[str] = set()

    for _, values in clusters.items():
        if isinstance(values, list):
            for value in values:
                tokens.add(normalize_token(value))
        elif isinstance(values, str):
            tokens.add(normalize_token(values))

    growth_prefs = target.get("growth_preferences", {}) or {}
    for _, values in growth_prefs.items():
        if isinstance(values, list):
            for value in values:
                tokens.add(normalize_token(value))
        elif isinstance(values, str):
            tokens.add(normalize_token(values))

    company_prefs = target.get("company_preferences", {}) or {}
    for _, values in company_prefs.items():
        if isinstance(values, list):
            for value in values:
                tokens.add(normalize_token(value))
        elif isinstance(values, str):
            tokens.add(normalize_token(values))

    return tokens

def get_priority_domain_weights(profile: Dict[str, Any]) -> Dict[str, float]:
    """
    Maps high-value target domains to stronger exposure weights.
    These represent areas the candidate especially wants to build in.
    """
    return {
        "cybersecurity": 3.2,
        "cyber security": 3.2,
        "cloud": 3.0,
        "engineering": 2.8,
        "software engineering": 2.8,
        "technology strategy": 2.0,
        "data": 2.2,
        "data analysis": 2.0,
        "machine learning": 2.2,
        "ai": 2.4,
        "digital transformation": 1.8,
    }

def score_tech_match(job: JobPosting, profile: Dict[str, Any]) -> tuple[float, List[str], List[str]]:
    """
    Tech Match scoring model:

    Total: 30 points
    - 10 points: prerequisite fit
    - 20 points: exposure alignment / technical opportunity

    Design intent:
    - If a role has hard technical requirements the candidate does not meet,
      prerequisite fit stays low and actionable gaps are surfaced.
    - If a graduate role has little/no required tech but offers strong exposure
      aligned with the candidate's interests, the tech score should still be high.
    """
    max_score = float(profile.get("weights", {}).get("tech_match", 30))

    # Internal split
    prereq_max = 10.0
    exposure_max = 20.0

    required_tools = dedupe_preserve_order(
        apply_synonym_map(job.candidate_required_tools or [], profile)
    )
    preferred_tools = dedupe_preserve_order(
        apply_synonym_map(job.candidate_preferred_tools or [], profile)
    )
    exposure_tools = dedupe_preserve_order(
        apply_synonym_map(job.role_exposure_tools or [], profile)
    )
    exposure_domains = dedupe_preserve_order(
        apply_synonym_map(job.role_exposure_domains or [], profile)
    )

    # Backward-compatible fallback if the new fields are empty
    if not required_tools and not preferred_tools and not exposure_tools and not exposure_domains:
        legacy_tools = dedupe_preserve_order(apply_synonym_map(job.tech_tools or [], profile))
        legacy_domains = dedupe_preserve_order(apply_synonym_map(job.tech_domains or [], profile))

        if legacy_tools:
            required_tools = legacy_tools
        elif legacy_domains:
            exposure_domains = legacy_domains
        else:
            return (
                round(max_score * 0.4, 1),
                ["No clear tech signals extracted; assigning conservative partial score"],
                [],
            )

    buckets = get_profile_skill_buckets(profile)
    target_interest_tokens = get_target_interest_tokens(profile)

    reasons: List[str] = []

    matched_required_languages: List[str] = []
    matched_required_tools: List[str] = []
    matched_preferred: List[str] = []
    matched_exposure_tools: List[str] = []
    matched_exposure_domains: List[str] = []
    matched_interest_exposure: List[str] = []

    missing_required: List[str] = []
    missing_preferred: List[str] = []

    def bucket_strength(item: str) -> str | None:
        if item in buckets["strong"]:
            return "strong"
        if item in buckets["working"]:
            return "working"
        if item in buckets["basic"]:
            return "basic"
        return None

    def interest_match(item: str) -> bool:
        item_n = normalize_token(item)

        aliases = {
            "cyber security": "cybersecurity",
            "cloud platforms": "cloud",
            "data analysis": "data",
            "software engineering": "engineering",
            "platform engineering": "engineering",
        }
        item_canonical = aliases.get(item_n, item_n)

        if item_n in target_interest_tokens or item_canonical in target_interest_tokens:
            return True

        for tok in target_interest_tokens:
            tok_canonical = aliases.get(tok, tok)
            if (
                tok in item_n
                or item_n in tok
                or tok_canonical in item_canonical
                or item_canonical in tok_canonical
            ):
                return True

        return False

    def is_domain_like(item: str) -> bool:
        return classify_tech_item(item) == "domain"
    
    aligned_exposure_concepts: List[str] = []

    def add_aligned_exposure(item: str) -> None:
        canonical = normalize_token(item)
        aliases = {
            "cyber security": "cybersecurity",
            "cloud platforms": "cloud",
            "data analysis": "data",
            "software engineering": "engineering",
            "digital platforms": "digital transformation",
        }
        canonical = aliases.get(canonical, canonical)

        if canonical not in aligned_exposure_concepts:
            aligned_exposure_concepts.append(canonical)

    # Prevent obvious double-counting:
    # if an exposure "tool" is really a domain and already exists in exposure_domains,
    # keep it only in exposure_domains.
    exposure_domain_set = set(exposure_domains)
    filtered_exposure_tools: List[str] = []
    for item in exposure_tools:
        if is_domain_like(item) and item in exposure_domain_set:
            continue
        filtered_exposure_tools.append(item)
    exposure_tools = filtered_exposure_tools

    # -------------------------
    # 1) PREREQUISITE FIT (10)
    # -------------------------
    prereq_score = 0.0

    # Required tools: strongest signal
    for item in required_tools:
        item_type = classify_tech_item(item)
        strength = bucket_strength(item)

        if item_type == "language":
            if strength == "strong":
                prereq_score += 3.5
                matched_required_languages.append(item)
            elif strength == "working":
                prereq_score += 2.8
                matched_required_languages.append(item)
            elif strength == "basic":
                prereq_score += 1.8
                matched_required_languages.append(item)
            else:
                missing_required.append(item)
        else:
            if strength == "strong":
                prereq_score += 2.8
                matched_required_tools.append(item)
            elif strength == "working":
                prereq_score += 2.2
                matched_required_tools.append(item)
            elif strength == "basic":
                prereq_score += 1.4
                matched_required_tools.append(item)
            else:
                missing_required.append(item)

    # Preferred tools: softer signal
    for item in preferred_tools:
        strength = bucket_strength(item)

        if strength == "strong":
            prereq_score += 1.4
            matched_preferred.append(item)
        elif strength == "working":
            prereq_score += 1.0
            matched_preferred.append(item)
        elif strength == "basic":
            prereq_score += 0.6
            matched_preferred.append(item)
        else:
            missing_preferred.append(item)

    prereq_score = min(prereq_score, prereq_max)

        # -------------------------
    # 2) EXPOSURE ALIGNMENT (20)
    # -------------------------
    exposure_score = 0.0

    unique_interest_exposure: List[str] = []
    priority_weights = get_priority_domain_weights(profile)

    def canonicalize_domain(item: str) -> str:
        item_n = normalize_token(item)
        aliases = {
            "cyber security": "cybersecurity",
            "cloud platforms": "cloud",
            "data analysis": "data",
            "software engineering": "engineering",
        }
        return aliases.get(item_n, item_n)

    seen_exposure_concepts: set[str] = set()

    # Exposure tools: meaningful upside, but less important than domains
    for item in exposure_tools:
        canonical = canonicalize_domain(item)
        strength = bucket_strength(item)

        if canonical in seen_exposure_concepts:
            continue

        item_points = 0.0
        aligned = False

        if strength == "strong":
            item_points += 1.6
            matched_exposure_tools.append(item)
            aligned = True
        elif strength == "working":
            item_points += 1.3
            matched_exposure_tools.append(item)
            aligned = True
        elif strength == "basic":
            item_points += 0.9
            matched_exposure_tools.append(item)
            aligned = True

        if interest_match(item):
            item_points += 1.8
            aligned = True

        if canonical in priority_weights:
            item_points += priority_weights[canonical] * 0.45
            aligned = True

        if aligned:
            add_aligned_exposure(canonical)

        if item_points > 0:
            seen_exposure_concepts.add(canonical)

        exposure_score += item_points

    # Exposure domains: strongest signal for graduate-role attractiveness
    for item in exposure_domains:
            canonical = canonicalize_domain(item)

            if canonical in seen_exposure_concepts:
                continue

            item_points = 0.0
            strength = bucket_strength(item)
            aligned = False

            if strength is not None:
                item_points += 1.2
                matched_exposure_domains.append(item)
                aligned = True

            if interest_match(item):
                item_points += 2.6
                aligned = True

            if canonical in priority_weights:
                item_points += priority_weights[canonical]
                aligned = True

            if canonical in {"cybersecurity", "cloud", "engineering", "ai", "machine learning", "data"}:
                item_points += 0.8
                aligned = True

            if aligned:
                add_aligned_exposure(canonical)

            if item_points > 0:
                seen_exposure_concepts.add(canonical)

            exposure_score += item_points

    matched_interest_exposure = aligned_exposure_concepts

    # Breadth bonus for grad roles with strong aligned exposure
    aligned_exposure_count = len(seen_exposure_concepts)
    if job.role_level == "graduate" or job.program_type == "grad_program":
        if aligned_exposure_count >= 6:
            exposure_score += 3.5
            reasons.append("Graduate-role bonus applied for broad high-value technical exposure")
        elif aligned_exposure_count >= 4:
            exposure_score += 2.5
            reasons.append("Graduate-role bonus applied for strong aligned technical exposure")
        elif aligned_exposure_count >= 2:
            exposure_score += 1.5
            reasons.append("Graduate-role bonus applied for some aligned technical exposure")

        if job.learning_environment == "strong":
            exposure_score += 2.0
            reasons.append("Graduate-role adjustment applied for strong learning environment")
        elif job.learning_environment == "medium":
            exposure_score += 1.0
            reasons.append("Graduate-role adjustment applied for moderate learning environment")

    exposure_score = min(exposure_score, exposure_max)

    total_score = min(round(prereq_score + exposure_score, 1), max_score)

    if matched_required_languages:
        reasons.append(f"Required language matches: {', '.join(matched_required_languages)}")
    if matched_required_tools:
        reasons.append(f"Required tool matches: {', '.join(matched_required_tools)}")
    if matched_preferred:
        reasons.append(f"Preferred tool matches: {', '.join(matched_preferred)}")
    if matched_interest_exposure:
        reasons.append(
            f"Exposure strongly aligned with target interests: {', '.join(matched_interest_exposure)}"
        )
    else:
        if matched_exposure_tools:
            reasons.append(f"Relevant exposure-tool overlap: {', '.join(matched_exposure_tools)}")
        if matched_exposure_domains:
            reasons.append(f"Relevant exposure-domain overlap: {', '.join(matched_exposure_domains)}")
    if missing_required:
        reasons.append(f"Actionable required gaps: {', '.join(missing_required)}")
    if missing_preferred:
        reasons.append(f"Preferred but non-blocking gaps: {', '.join(missing_preferred)}")
    if exposure_tools and not matched_exposure_tools and not any(interest_match(x) for x in exposure_tools):
        reasons.append(
            f"Role exposure tools present but not treated as gaps: {', '.join(exposure_tools)}"
        )
    if exposure_domains and not matched_exposure_domains and not any(interest_match(x) for x in exposure_domains):
        reasons.append(
            f"Role exposure domains present but not treated as gaps: {', '.join(exposure_domains)}"
        )

    reasons.append(
        f"Tech scoring split: prerequisite fit {round(prereq_score, 1)}/{int(prereq_max)}, "
        f"exposure alignment {round(exposure_score, 1)}/{int(exposure_max)}"
    )

    # Only required tools are actionable gaps.
    missing_skills = missing_required
    return total_score, reasons, missing_skills

def score_location_fit(job: JobPosting, profile: Dict[str, Any]) -> tuple[float, List[str]]:
    rules = profile.get("scoring_rules", {}).get("location", {}) or {}

    location_text = normalize(job.location or "")
    company_text = normalize(job.company or "")
    raw_text = normalize(job.raw_text or "")
    work_mode = (job.work_mode or "unknown").lower()

    # Explicit Sydney mention
    if "sydney" in location_text:
        if work_mode == "hybrid":
            return float(rules.get("hybrid_sydney", 15)), ["Hybrid Sydney role"]
        return float(rules.get("sydney", 15)), ["Sydney-based role"]

    # Infer Sydney from employer / job text
    inferred_sydney_signals = [
        "university of sydney",
        "usyd",
        "sydney campus",
    ]

    if any(signal in company_text for signal in inferred_sydney_signals) or any(
        signal in raw_text for signal in inferred_sydney_signals
    ):
        if work_mode == "hybrid":
            return float(rules.get("hybrid_sydney", 15)), ["Inferred Sydney location from employer/job context"]
        return float(rules.get("sydney", 15)), ["Inferred Sydney location from employer/job context"]

    if work_mode == "remote":
        return float(rules.get("remote", 14)), ["Remote role"]

    if work_mode == "hybrid":
        return float(rules.get("australia_other_hybrid", 10)), ["Hybrid role outside Sydney or unspecified city"]

    if work_mode == "onsite":
        return float(rules.get("australia_other_onsite", 6)), ["Onsite role outside Sydney or unspecified city"]

    return float(rules.get("unknown", 8)), ["Location/work mode unclear"]


def score_company_quality(job: JobPosting, profile: Dict[str, Any]) -> tuple[float, List[str]]:
    """
    Company/program quality is now driven by structured AI judgments,
    not hardcoded company-name lists.
    """
    max_score = float(profile.get("weights", {}).get("company_quality", 15))

    score = 0.0
    reasons: List[str] = []

    if job.employer_signal == "strong":
        score += 5.0
        reasons.append("AI evaluation: strong employer signal")
    elif job.employer_signal == "medium":
        score += 3.0
        reasons.append("AI evaluation: moderate employer signal")
    elif job.employer_signal == "weak":
        score += 1.0
        reasons.append("AI evaluation: weak employer signal")

    if job.program_quality == "strong":
        score += 5.0
        reasons.append("AI evaluation: strong program quality")
    elif job.program_quality == "medium":
        score += 3.0
        reasons.append("AI evaluation: moderate program quality")
    elif job.program_quality == "weak":
        score += 1.0
        reasons.append("AI evaluation: weak program quality")

    if job.role_substance == "strong":
        score += 3.0
        reasons.append("AI evaluation: role appears substantive and meaningful")
    elif job.role_substance == "medium":
        score += 1.5
        reasons.append("AI evaluation: role appears reasonably substantive")
    elif job.role_substance == "weak":
        score += 0.5
        reasons.append("AI evaluation: role appears limited in substance")

    if job.learning_environment == "strong":
        score += 2.0
        reasons.append("AI evaluation: strong learning environment")
    elif job.learning_environment == "medium":
        score += 1.0
        reasons.append("AI evaluation: moderate learning environment")

    return min(round(score, 1), max_score), reasons


def score_growth(job: JobPosting, profile: Dict[str, Any]) -> tuple[float, List[str]]:
    max_score = float(profile.get("weights", {}).get("growth", 10))
    score = 0.0
    reasons: List[str] = []

    # Light deterministic layer for explicit signals extracted from the ad
    growth_items = [normalize_token(x) for x in (job.growth_signals or [])]
    growth_weights = {
        "mentorship": 2.0,
        "mentoring": 2.0,
        "training": 1.5,
        "coaching": 1.5,
        "certifications": 1.5,
        "career progression": 1.5,
        "career growth": 1.5,
        "rotations": 2.0,
        "rotation": 2.0,
        "support and development": 1.5,
        "learning & development platforms": 1.5,
        "professional skills building": 1.5,
    }

    for item in growth_items:
        for key, value in growth_weights.items():
            if key in item:
                score += value
                reasons.append(f"Growth signal detected: {item}")
                break

    if job.program_type == "grad_program":
        score += 1.5
        reasons.append("Structured graduate program detected")

    # Primary signal now comes from AI evaluation
    if job.learning_environment == "strong":
        score += 2.5
        reasons.append("AI evaluation: strong learning environment")
    elif job.learning_environment == "medium":
        score += 1.2
        reasons.append("AI evaluation: moderate learning environment")

    if job.trajectory_value == "strong":
        score += 2.0
        reasons.append("AI evaluation: strong long-term trajectory value")
    elif job.trajectory_value == "medium":
        score += 1.0
        reasons.append("AI evaluation: moderate long-term trajectory value")

    return min(round(score, 1), max_score), reasons


def score_risk(job: JobPosting, profile: Dict[str, Any]) -> tuple[float, List[str], List[str]]:
    """
    Risk remains deterministic, but ambiguous offsets now come from AI-evaluated
    employer/program quality rather than hardcoded employer lists.
    """
    max_score = float(profile.get("weights", {}).get("risk", 5))
    score = max_score
    reasons: List[str] = []
    flags: List[str] = []

    risk_flags = [normalize_token(x) for x in (job.risk_flags or [])]
    contract_type = (job.contract_type or "unknown").lower()
    location_text = normalize(job.location or "")
    work_mode = (job.work_mode or "unknown").lower()

    hard_no = [normalize_token(x) for x in profile.get("scoring_rules", {}).get("hard_no", []) or []]
    for flag in risk_flags:
        if flag in hard_no:
            flags.append(f"Hard no: {flag}")
            reasons.append(f"Hard rejection triggered by '{flag}'")
            return 0.0, reasons, flags

    if contract_type == "contract":
        if job.program_quality == "strong" and job.employer_signal == "strong":
            penalty = 0.5
            reasons.append("Contract penalty reduced due to strong AI-evaluated program/employer quality")
        elif job.program_quality in {"strong", "medium"} or job.employer_signal in {"strong", "medium"}:
            penalty = 1.0
            reasons.append("Small contract penalty due to positive AI-evaluated employer/program quality")
        else:
            penalty = 2.0
            reasons.append("Contract role penalty applied")

        score -= penalty
        flags.append("Contract role")

    if work_mode == "onsite" and "sydney" not in location_text and location_text:
        score -= 2
        flags.append("Onsite outside Sydney")
        reasons.append("Onsite role outside Sydney")

    if job.clearance_required:
        reasons.append("Clearance/background checks required, but no penalty applied")

    return max(round(score, 1), 0.0), reasons, flags

def university_matches_profile(required_university: str | None, profile: Dict[str, Any]) -> bool:
    """
    Compare extracted university restriction against the user's university.
    Only needs to recognize UTS aliases for now.
    """
    if not required_university:
        return False

    required = normalize(required_university)
    current_university = normalize(profile.get("eligibility", {}).get("current_university", ""))

    if current_university != "uts":
        return False

    uts_aliases = {
        "uts",
        "university of technology sydney",
    }

    return required in uts_aliases


def check_profile_eligibility(job: JobPosting, profile: Dict[str, Any]) -> tuple[bool, List[str], List[str]]:
    """
    Returns:
    - eligible: bool
    - reasons: explanatory notes
    - flags: blocking flags if any
    """
    reasons: List[str] = []
    flags: List[str] = []

    if job.university_restriction_present:
        required_university = (job.required_university or "").strip()

        # Only hard-block if a specific university was actually extracted
        if required_university:
            if university_matches_profile(required_university, profile):
                reasons.append("University restriction satisfied: role requires UTS and profile matches")
                return True, reasons, flags

            current_display = profile.get("eligibility", {}).get("current_university", "unknown")
            flags.append(f"Ineligible: role restricted to {required_university}, profile is {current_display}")
            reasons.append("University-specific eligibility restriction does not match profile")
            return False, reasons, flags

        # Restriction flag was raised but no university could be identified.
        # Do NOT hard block; treat as uncertain and allow normal scoring.
        reasons.append("University restriction signal detected but no specific university could be resolved")
        return True, reasons, flags

    return True, reasons, flags

def score_job_hybrid(job: JobPosting, profile: Dict[str, Any]) -> Dict[str, Any]:
    eligible, eligibility_reasons, eligibility_flags = check_profile_eligibility(job, profile)
    if not eligible:
        return {
            "score": 0.0,
            "score_breakdown": {
                "role_fit": 0.0,
                "tech_match": 0.0,
                "location": 0.0,
                "company_quality": 0.0,
                "growth": 0.0,
                "risk": 0.0,
            },
            "missing_skills": [],
            "flags": eligibility_flags,
            "reasons": {
                "role_fit": [],
                "tech_match": [],
                "location": [],
                "company_quality": [],
                "growth": [],
                "risk": eligibility_reasons,
            },
        }
    role_score, role_reasons = score_role_fit(job, profile)
    tech_score, tech_reasons, missing_skills = score_tech_match(job, profile)
    location_score, location_reasons = score_location_fit(job, profile)
    company_score, company_reasons = score_company_quality(job, profile)
    growth_score, growth_reasons = score_growth(job, profile)
    risk_score, risk_reasons, flags = score_risk(job, profile)

    hard_reject = any(flag.lower().startswith("hard no:") for flag in flags)
    total = 0.0 if hard_reject else round(
        role_score + tech_score + location_score + company_score + growth_score + risk_score,
        1
    )

    return {
        "score": total,
        "score_breakdown": {
            "role_fit": round(role_score, 1),
            "tech_match": round(tech_score, 1),
            "location": round(location_score, 1),
            "company_quality": round(company_score, 1),
            "growth": round(growth_score, 1),
            "risk": round(risk_score, 1),
        },
        "missing_skills": missing_skills,
        "flags": flags,
        "reasons": {
            "role_fit": role_reasons,
            "tech_match": tech_reasons,
            "location": location_reasons,
            "company_quality": company_reasons,
            "growth": growth_reasons,
            "risk": risk_reasons,
        },
    }


# -----------------------------
# AI extraction
# -----------------------------
def extract_job_with_ai(client: OpenAI, job: JobPosting, cache_dir: Path) -> Dict[str, Any]:
    cache_file = cache_dir / f"{hash_job(job.raw_text)}.json"

    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed reading extraction cache %s: %s. Regenerating.", cache_file.name, exc)

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title": {"type": ["string", "null"]},
            "company": {"type": ["string", "null"]},
            "location": {"type": ["string", "null"]},
            "role_level": {
                "type": ["string", "null"],
                "enum": ["graduate", "entry", "junior", "mid", "senior", "unknown", None],
            },
            "program_type": {
                "type": ["string", "null"],
                "enum": ["grad_program", "standard_role", "unknown", None],
            },
            "work_mode": {
                "type": ["string", "null"],
                "enum": ["onsite", "hybrid", "remote", "unknown", None],
            },
            "contract_type": {
                "type": ["string", "null"],
                "enum": ["permanent", "contract", "internship", "unknown", None],
            },
            "tech_tools": {"type": "array", "items": {"type": "string"}},
            "tech_domains": {"type": "array", "items": {"type": "string"}},
            "candidate_required_tools": {"type": "array", "items": {"type": "string"}},
            "candidate_preferred_tools": {"type": "array", "items": {"type": "string"}},
            "role_exposure_tools": {"type": "array", "items": {"type": "string"}},
            "role_exposure_domains": {"type": "array", "items": {"type": "string"}},
            "must_haves": {"type": "array", "items": {"type": "string"}},
            "nice_to_haves": {"type": "array", "items": {"type": "string"}},
            "responsibilities": {"type": "array", "items": {"type": "string"}},
            "growth_signals": {"type": "array", "items": {"type": "string"}},
            "citizenship_required": {"type": ["boolean", "null"]},
            "clearance_required": {"type": ["boolean", "null"]},
            "university_restriction_present": {"type": ["boolean", "null"]},
            "required_university": {"type": ["string", "null"]},
            "risk_flags": {"type": "array", "items": {"type": "string"}},
            "evidence": {
                "type": ["object", "null"],
                "additionalProperties": False,
                "properties": {
                    "role_level": {"type": "array", "items": {"type": "string"}},
                    "program_type": {"type": "array", "items": {"type": "string"}},
                    "work_mode": {"type": "array", "items": {"type": "string"}},
                    "contract_type": {"type": "array", "items": {"type": "string"}},
                    "tech_tools": {"type": "array", "items": {"type": "string"}},
                    "tech_domains": {"type": "array", "items": {"type": "string"}},
                    "candidate_required_tools": {"type": "array", "items": {"type": "string"}},
                    "candidate_preferred_tools": {"type": "array", "items": {"type": "string"}},
                    "role_exposure_tools": {"type": "array", "items": {"type": "string"}},
                    "role_exposure_domains": {"type": "array", "items": {"type": "string"}},
                    "growth_signals": {"type": "array", "items": {"type": "string"}},
                    "citizenship_required": {"type": "array", "items": {"type": "string"}},
                    "clearance_required": {"type": "array", "items": {"type": "string"}},
                    "university_restriction_present": {"type": "array", "items": {"type": "string"}},
                    "required_university": {"type": "array", "items": {"type": "string"}},
                    "risk_flags": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "role_level",
                    "program_type",
                    "work_mode",
                    "contract_type",
                    "tech_tools",
                    "tech_domains",
                    "candidate_required_tools",
                    "candidate_preferred_tools",
                    "role_exposure_tools",
                    "role_exposure_domains",
                    "growth_signals",
                    "citizenship_required",
                    "clearance_required",
                    "university_restriction_present",
                    "required_university",
                    "risk_flags",
                ],
            },
            "summary": {"type": ["string", "null"]},
        },
        "required": [
            "title",
            "company",
            "location",
            "role_level",
            "program_type",
            "work_mode",
            "contract_type",
            "tech_tools",
            "tech_domains",
            "candidate_required_tools",
            "candidate_preferred_tools",
            "role_exposure_tools",
            "role_exposure_domains",
            "must_haves",
            "nice_to_haves",
            "responsibilities",
            "growth_signals",
            "citizenship_required",
            "clearance_required",
            "university_restriction_present",
            "required_university",
            "risk_flags",
            "evidence",
            "summary",
        ],
    }

    prompt = (
        "Extract structured information from this job advertisement.\n"
        "Return valid JSON only.\n\n"
        "Rules:\n"
        "- If title/company/location are not explicitly present, return null.\n"
        "- role_level must be one of: graduate, entry, junior, mid, senior, unknown.\n"
        "- program_type must be one of: grad_program, standard_role, unknown.\n"
        "- work_mode must be one of: onsite, hybrid, remote, unknown.\n"
        "- contract_type must be one of: permanent, contract, internship, unknown.\n"
        "- If role_level, program_type, work_mode, or contract_type cannot be determined, return unknown.\n"
        "- tech_tools should include all concrete programming languages, frameworks, cloud platforms, products, services, and software tools explicitly mentioned anywhere in the ad.\n"
        "- tech_domains should include all broader technical areas or capability domains explicitly mentioned anywhere in the ad, such as ai, machine learning, cloud, automation, analytics, cybersecurity, devops.\n"
        "- candidate_required_tools should include only technologies the candidate is explicitly required or expected to already know before applying.\n"
        "- candidate_preferred_tools should include only technologies listed as preferred, desirable, advantageous, or nice-to-have.\n"
        "- role_exposure_tools should include concrete tools/platforms the candidate would work with or learn in the role, but which are not clearly stated as prerequisites.\n"
        "- role_exposure_domains should include broader areas the role touches, such as ai, cloud, automation, analytics, cybersecurity, or machine learning, when they describe the work, team, projects, or learning exposure rather than applicant prerequisites.\n"
        "- Do not place a technology in candidate_required_tools unless the ad clearly frames it as a requirement, prior experience expectation, applicant skill, qualification, or selection criterion.\n"
        "- Do not place a technology in candidate_preferred_tools unless the ad clearly frames it as preferred, desirable, advantageous, or nice-to-have.\n"
        "- If a technology is only mentioned in duties, project descriptions, team context, rotation descriptions, or what the candidate will work on after joining, place it in role_exposure_tools or role_exposure_domains instead.\n"
        "- Programming languages must go in tech_tools, and may also appear in candidate_required_tools or candidate_preferred_tools if the ad explicitly frames them that way.\n"
        "- tech_tools must only include concrete technologies, languages, frameworks, products, cloud platforms, services, or software tools.\n"
        "- tech_domains must include broader concepts, capability areas, and methods such as automation, ai, machine learning, analytics, forecasting, time series forecasting, natural language processing, and optical character recognition.\n"
        "- Do not place conceptual areas or methods like time series forecasting into tech_tools.\n"
        "- Normalize technologies to short canonical names where reasonable.\n"
        "- must_haves and nice_to_haves should be short skill phrases.\n"
        "- responsibilities should be concise bullet-like strings.\n"
        "- growth_signals should include only clear development signals such as mentorship, rotations, training, certifications, coaching, career progression.\n"
        "- citizenship_required should be true only if explicitly required.\n"
        "- clearance_required should be true only if explicitly required.\n"
        "- university_restriction_present should be true only if the role explicitly restricts applicants to students or graduates of a named university.\n"
        "- required_university should be the university name as written in the ad when such a restriction exists, otherwise null.\n"
        "- If the job says it is only open to students or graduates of a specific university, extract that university into required_university.\n"
        "- Do not normalize required_university to a code; keep the university name close to the wording in the ad.\n"
        "- Do not set university_restriction_present to true merely because the ad lists suitable degrees or fields of study; only set it to true if applicants are explicitly limited to a named university.\n"
        "- risk_flags should only include objective concerns explicitly supported by the ad.\n"
        "- evidence should contain short supporting snippets from the ad.\n"
        "- summary should be 1-3 sentences.\n"
        "- Do not invent details.\n\n"
        "JOB AD:\n"
        f"{job.raw_text}"
    )

    try:
        response = client.responses.create(
            model=AI_MODEL,
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "job_extraction",
                    "strict": True,
                    "schema": schema,
                }
            },
        )
        data = json.loads(response.output_text)
        cache_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return data
    except Exception as exc:
        logger.error("AI extraction failed for %s: %s", job.filename, exc)
        return {}


# -----------------------------
# AI strategic evaluation
# -----------------------------
def evaluate_job_with_ai(
    client: OpenAI,
    job: JobPosting,
    profile: Dict[str, Any],
    cache_dir: Path,
) -> Dict[str, Any]:
    cache_file = cache_dir / f"{hash_eval(job.raw_text, profile, job)}.eval.json"

    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed reading evaluation cache %s: %s. Regenerating.", cache_file.name, exc)

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "structural_fit_summary": {"type": ["string", "null"]},
            "strategic_fit_summary": {"type": ["string", "null"]},
            "recommendation": {
                "type": "string",
                "enum": ["strong_apply", "apply", "stretch_apply", "low_priority", "skip"],
            },
            "program_quality": {
                "type": "string",
                "enum": ["strong", "medium", "weak", "unknown"],
            },
            "role_substance": {
                "type": "string",
                "enum": ["strong", "medium", "weak", "unknown"],
            },
            "learning_environment": {
                "type": "string",
                "enum": ["strong", "medium", "weak", "unknown"],
            },
            "trajectory_value": {
                "type": "string",
                "enum": ["strong", "medium", "weak", "unknown"],
            },
            "employer_signal": {
                "type": "string",
                "enum": ["strong", "medium", "weak", "unknown"],
            },
            "gap_severity": {
                "type": "string",
                "enum": ["low", "medium", "high", "unknown"],
            },
            "strategic_reasons": {
                "type": "array",
                "items": {"type": "string"},
            },
            "caution_reasons": {
                "type": "array",
                "items": {"type": "string"},
            },
            "evidence": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "program_quality": {"type": "array", "items": {"type": "string"}},
                    "learning_environment": {"type": "array", "items": {"type": "string"}},
                    "trajectory_value": {"type": "array", "items": {"type": "string"}},
                    "recommendation": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "program_quality",
                    "learning_environment",
                    "trajectory_value",
                    "recommendation",
                ],
            },
        },
        "required": [
            "structural_fit_summary",
            "strategic_fit_summary",
            "recommendation",
            "program_quality",
            "role_substance",
            "learning_environment",
            "trajectory_value",
            "employer_signal",
            "gap_severity",
            "strategic_reasons",
            "caution_reasons",
            "evidence",
        ],
    }

    profile_context = {
        "preferred_locations": profile.get("target", {}).get("preferred_locations", []),
        "preferred_work_modes": profile.get("target", {}).get("preferred_work_modes", []),
        "role_levels": profile.get("target", {}).get("role_levels", []),
        "preferred_role_clusters": profile.get("target", {}).get("preferred_role_clusters", {}),
        "company_preferences": profile.get("target", {}).get("company_preferences", {}),
        "growth_preferences": profile.get("target", {}).get("growth_preferences", {}),
        "eligibility": profile.get("eligibility", {}),
    }

    extracted_context = {
        "title": job.title,
        "company": job.company,
        "location": job.location,
        "role_level": job.role_level,
        "program_type": job.program_type,
        "work_mode": job.work_mode,
        "contract_type": job.contract_type,
        "tech_tools": job.tech_tools,
        "tech_domains": job.tech_domains,
        "candidate_required_tools": job.candidate_required_tools,
        "candidate_preferred_tools": job.candidate_preferred_tools,
        "role_exposure_tools": job.role_exposure_tools,
        "role_exposure_domains": job.role_exposure_domains,
        "must_haves": job.must_haves,
        "nice_to_haves": job.nice_to_haves,
        "responsibilities": job.responsibilities,
        "growth_signals": job.growth_signals,
        "citizenship_required": job.citizenship_required,
        "clearance_required": job.clearance_required,
        "university_restriction_present": job.university_restriction_present,
        "required_university": job.required_university,
        "risk_flags": job.risk_flags,
        "summary": job.summary,
    }

    prompt = (
        "Evaluate this job advertisement for career fit.\n"
        "Return valid JSON only.\n\n"
        "You are not scoring numerically. You are making structured judgments that will later be converted into deterministic points.\n\n"
        "Candidate preferences:\n"
        f"{json.dumps(profile_context, indent=2)}\n\n"
        "Extracted job context:\n"
        f"{json.dumps(extracted_context, indent=2)}\n\n"
        "Rules:\n"
        "- structural_fit_summary should briefly explain whether the role matches hard preferences such as location, role level, work mode, and eligibility.\n"
        "- strategic_fit_summary should explain whether the role is genuinely strong for early-career growth.\n"
        "- recommendation must be one of: strong_apply, apply, stretch_apply, low_priority, skip.\n"
        "- program_quality reflects how structured and well-supported the program/role appears.\n"
        "- role_substance reflects whether the work sounds real and meaningful.\n"
        "- learning_environment reflects evidence of mentoring, training, coaching, certifications, and support.\n"
        "- trajectory_value reflects whether the role builds useful long-term career capital.\n"
        "- employer_signal reflects reputation/signalling value for an early-career applicant.\n"
        "- gap_severity should be LOW when missing experience appears learnable within the role or normal for a graduate hire.\n"
        "- Broad domains such as ai, cloud, data analytics, digital transformation, and machine learning should NOT automatically be treated as hard disqualifying gaps.\n"
        "- For graduate roles, distinguish between true blockers and learnable exposure areas.\n"
        "- strategic_reasons should be concise, high-value reasons the role is attractive.\n"
        "- caution_reasons should be concise and realistic.\n"
        "- evidence should contain short snippets or references from the ad supporting your judgment.\n"
        "- Do not invent facts not supported by the ad.\n\n"
        "JOB AD:\n"
        f"{job.raw_text}"
    )

    try:
        response = client.responses.create(
            model=AI_MODEL,
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "job_evaluation",
                    "strict": True,
                    "schema": schema,
                }
            },
        )
        data = json.loads(response.output_text)
        cache_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return data
    except Exception as exc:
        logger.error("AI evaluation failed for %s: %s", job.filename, exc)
        return {}


def enrich_jobs_with_ai(
    jobs: List[JobPosting],
    client: OpenAI,
    cache_dir: Path,
    profile: Dict[str, Any],
) -> None:
    for job in jobs:
        ai_data = extract_job_with_ai(client=client, job=job, cache_dir=cache_dir)
        if ai_data:
            job.title = ai_data.get("title")
            job.company = ai_data.get("company")
            job.location = ai_data.get("location")
            job.must_haves = ai_data.get("must_haves", []) or []
            job.nice_to_haves = ai_data.get("nice_to_haves", []) or []
            job.responsibilities = ai_data.get("responsibilities", []) or []
            job.summary = ai_data.get("summary")
            job.role_level = ai_data.get("role_level")
            job.program_type = ai_data.get("program_type")
            job.work_mode = ai_data.get("work_mode")
            job.tech_tools = ai_data.get("tech_tools", []) or []
            job.tech_domains = ai_data.get("tech_domains", []) or []
            job.candidate_required_tools = ai_data.get("candidate_required_tools", []) or []
            job.candidate_preferred_tools = ai_data.get("candidate_preferred_tools", []) or []
            job.role_exposure_tools = ai_data.get("role_exposure_tools", []) or []
            job.role_exposure_domains = ai_data.get("role_exposure_domains", []) or []
            job.growth_signals = ai_data.get("growth_signals", []) or []
            job.citizenship_required = ai_data.get("citizenship_required")
            job.clearance_required = ai_data.get("clearance_required")
            job.university_restriction_present = ai_data.get("university_restriction_present")
            job.required_university = ai_data.get("required_university")
            job.contract_type = ai_data.get("contract_type")
            job.risk_flags = ai_data.get("risk_flags", []) or []
            job.evidence = ai_data.get("evidence")

        eval_data = evaluate_job_with_ai(
            client=client,
            job=job,
            profile=profile,
            cache_dir=cache_dir,
        )
        if eval_data:
            job.structural_fit_summary = eval_data.get("structural_fit_summary")
            job.strategic_fit_summary = eval_data.get("strategic_fit_summary")
            job.recommendation = eval_data.get("recommendation")
            job.program_quality = eval_data.get("program_quality")
            job.role_substance = eval_data.get("role_substance")
            job.learning_environment = eval_data.get("learning_environment")
            job.trajectory_value = eval_data.get("trajectory_value")
            job.employer_signal = eval_data.get("employer_signal")
            job.gap_severity = eval_data.get("gap_severity")
            job.strategic_reasons = eval_data.get("strategic_reasons", []) or []
            job.caution_reasons = eval_data.get("caution_reasons", []) or []
            job.ai_evaluation_evidence = eval_data.get("evidence", {}) or {}


# -----------------------------
# Output
# -----------------------------
def write_results_md(
    scored: List[Tuple[JobPosting, Dict[str, Any]]],
    out_dir: Path,
    top_n: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.md"
    run_date = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines: List[str] = [
        "# JobFit Analysis",
        f"Run date: {run_date}",
        f"Jobs processed: {len(scored)}",
        "",
    ]

    if not scored:
        lines.append("No jobs were scored.")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path

    for idx, (job, result) in enumerate(scored[:top_n], start=1):
        title = job.title or job.filename.replace(".txt", "")
        company = job.company or "Unknown"
        location = job.location or "Unknown"
        breakdown = result.get("score_breakdown") or {}
        flags = result.get("flags") or []
        reasons = result.get("reasons") or {}
        missing_skills = job.missing_skills or []

        lines.append(f"## {idx}) {title} — {company}")
        lines.append(f"**Score:** {job.score}/100")
        lines.append(f"**Recommendation:** {recommendation_label(job.recommendation)}")
        ineligible_flags = [flag for flag in flags if flag.lower().startswith("ineligible:")]
        if ineligible_flags:
            lines.append("### Eligibility Issue")
            for flag in ineligible_flags:
                lines.append(f"- {flag}")
            lines.append("")
        lines.append(f"**Location:** {location}")
        lines.append(f"**Work mode:** {job.work_mode or 'Unknown'}")
        lines.append(f"**Role level:** {job.role_level or 'Unknown'}")
        lines.append(f"**Program type:** {job.program_type or 'Unknown'}")
        lines.append("")

        lines.append("### Structural Fit")
        lines.append(job.structural_fit_summary or "No structured fit summary available.")
        lines.append("")

        lines.append("### Why This Role Is Strategically Strong")
        if job.strategic_fit_summary:
            lines.append(job.strategic_fit_summary)
            lines.append("")
        if job.strategic_reasons:
            for reason in job.strategic_reasons:
                lines.append(f"- {reason}")
        else:
            lines.append("- No strategic reasons available")
        lines.append("")

        lines.append("### Main Gaps / Cautions")
        if job.caution_reasons:
            for reason in job.caution_reasons:
                lines.append(f"- {reason}")
        else:
            lines.append("- None identified")
        lines.append("")

        lines.append("### Score Breakdown")
        lines.append(f"- Role Fit: {breakdown.get('role_fit', 0)}/25")
        lines.append(f"- Tech Match: {breakdown.get('tech_match', 0)}/30")
        lines.append(f"- Location: {breakdown.get('location', 0)}/15")
        lines.append(f"- Company/Program: {breakdown.get('company_quality', 0)}/15")
        lines.append(f"- Growth: {breakdown.get('growth', 0)}/10")
        lines.append(f"- Risk: {breakdown.get('risk', 0)}/5")
        lines.append("")

        lines.append("### Actionable Gaps")
        if missing_skills:
            for skill in missing_skills:
                lines.append(f"- {skill}")
        else:
            lines.append("- None identified")
        lines.append("")

        lines.append("### Flags")
        if flags:
            for flag in flags:
                lines.append(f"- {flag}")
        else:
            lines.append("- None")
        lines.append("")

        lines.append("### Why this scored the way it did")
        for category_label, key in [
            ("Role Fit", "role_fit"),
            ("Tech Match", "tech_match"),
            ("Location", "location"),
            ("Company/Program", "company_quality"),
            ("Growth", "growth"),
            ("Risk", "risk"),
        ]:
            lines.append(f"**{category_label}**")
            category_reasons = reasons.get(key, []) or []
            if category_reasons:
                for reason in category_reasons:
                    lines.append(f"- {reason}")
            else:
                lines.append("- No specific notes")
            lines.append("")

        lines.append("---")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JobFit Automator")
    parser.add_argument("--profile", type=Path, default=Path("profile.yaml"), help="Path to profile.yaml")
    parser.add_argument("--out", type=Path, default=Path("output"), help="Output folder")
    parser.add_argument("--cache", type=Path, default=Path("cache"), help="Cache folder")
    parser.add_argument("--top", type=int, default=10, help="Top N jobs to include in report")
    parser.add_argument("--no-ai", action="store_true", help="Disable AI enrichment")
    parser.add_argument("--verbose", action="store_true", help="Verbose logs")
    return parser.parse_args()


def read_pasted_job() -> str:
    print("\nPaste the job description below.")
    print("Type ::end on a new line when finished.\n")

    lines: List[str] = []
    while True:
        line = input()
        if line.strip() == "::end":
            break
        lines.append(line)

    return "\n".join(lines).strip()


def print_job_result(job: JobPosting, result: Dict[str, Any], verbose: bool) -> None:
    title = job.title or job.filename.replace(".txt", "")
    company = job.company or "Unknown"
    location = job.location or "Unknown"
    breakdown = result.get("score_breakdown") or {}
    flags = result.get("flags") or []
    reasons = result.get("reasons") or {}

    print(f"\nJob: {title} — {company}")
    print(f"Score: {job.score}/100")
    print(f"Recommendation: {recommendation_label(job.recommendation)}")

    ineligible_flags = [flag for flag in flags if flag.lower().startswith("ineligible:")]
    if ineligible_flags:
        print("\nEligibility issue")
        for flag in ineligible_flags:
            print(f"- {flag}")

        print("\nWhy it was rejected")
        for reason in reasons.get("risk", []) or []:
            print(f"- {reason}")
        return

    print(f"\nStructural Fit")
    if job.structural_fit_summary:
        print(job.structural_fit_summary)
    else:
        print("No structured fit summary available.")

    print(f"\nStrategic Fit")
    if job.strategic_fit_summary:
        print(job.strategic_fit_summary)
    else:
        print("No strategic fit summary available.")

    print(f"\nMain cautions")
    if job.caution_reasons:
        for reason in job.caution_reasons:
            print(f"- {reason}")
    else:
        print("- None identified")

    print(f"\nActionable Gaps")
    if job.missing_skills:
        for skill in job.missing_skills:
            print(f"- {skill}")
    else:
        print("- None identified")

    print("\nBreakdown")
    print(
        f"Role {breakdown.get('role_fit', 0)} | "
        f"Tech {breakdown.get('tech_match', 0)} | "
        f"Location {breakdown.get('location', 0)} | "
        f"Company {breakdown.get('company_quality', 0)} | "
        f"Growth {breakdown.get('growth', 0)} | "
        f"Risk {breakdown.get('risk', 0)}"
    )

    if verbose:
        print("\nWhy this role is strategically strong")
        if job.strategic_reasons:
            for reason in job.strategic_reasons:
                print(f"- {reason}")
        else:
            print("- No strategic reasons available")

        print("\nFlags")
        if flags:
            for flag in flags:
                print(f"- {flag}")
        else:
            print("- None")

        print("\nWhy this scored the way it did")
        for category_label, key in [
            ("Role Fit", "role_fit"),
            ("Tech Match", "tech_match"),
            ("Location", "location"),
            ("Company/Program", "company_quality"),
            ("Growth", "growth"),
            ("Risk", "risk"),
        ]:
            category_reasons = reasons.get(key, []) or []
            print(f"{category_label}:")
            if category_reasons:
                for reason in category_reasons:
                    print(f"  - {reason}")
            else:
                print("  - No specific notes")

        print("\nRaw extracted fields")
        print(f"- Location: {location}")
        print(f"- Work mode: {job.work_mode or 'Unknown'}")
        print(f"- Role level: {job.role_level or 'Unknown'}")
        print(f"- Program type: {job.program_type or 'Unknown'}")
        print(f"- Contract type: {job.contract_type or 'Unknown'}")
        print(f"- Tech tools: {job.tech_tools}")
        print(f"- Tech domains: {job.tech_domains}")
        print(f"- Candidate required tools: {job.candidate_required_tools}")
        print(f"- Candidate preferred tools: {job.candidate_preferred_tools}")
        print(f"- Role exposure tools: {job.role_exposure_tools}")
        print(f"- Role exposure domains: {job.role_exposure_domains}")
        print(f"- Growth signals: {job.growth_signals}")
        print(f"- Clearance required: {job.clearance_required}")
        print(f"- Citizenship required: {job.citizenship_required}")
        print(f"- University restriction present: {job.university_restriction_present}")
        print(f"- Required university: {job.required_university}")
        print(f"- AI evidence: {job.ai_evaluation_evidence}")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    args.cache.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    profile = load_profile(args.profile)
    job_text = read_pasted_job()

    if not job_text:
        logger.error("No job text provided.")
        return

    jobs = [JobPosting(filename="pasted_job", raw_text=job_text)]
    client = None if args.no_ai else build_openai_client()

    if client is None:
        logger.info("AI enrichment disabled (no key or --no-ai).")
    else:
        logger.info("AI enrichment enabled (cached).")
        enrich_jobs_with_ai(jobs, client=client, cache_dir=args.cache, profile=profile)

    scored: List[Tuple[JobPosting, Dict[str, Any]]] = []
    for job in jobs:
        result = score_job_hybrid(job, profile)
        if any(flag.lower().startswith("ineligible:") for flag in result.get("flags", [])):
            job.recommendation = "skip"
        job.score = result["score"]
        job.score_breakdown = result["score_breakdown"]
        job.missing_skills = result["missing_skills"]
        scored.append((job, result))

    scored.sort(key=lambda x: (x[0].score or 0, x[0].filename), reverse=True)

    for job, result in scored:
        print_job_result(job, result, verbose=args.verbose)

    report_path = write_results_md(scored, out_dir=args.out, top_n=args.top)
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
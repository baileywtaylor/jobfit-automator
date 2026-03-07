"""
JobFit Automator (V1)
- Ingest job ads from jobs_inbox/*.txt
- (Optional) AI-extract structured fields + summary with caching
- Deterministic keyword scoring
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


# -----------------------------
# Logging
# -----------------------------
logger = logging.getLogger("jobfit")


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


# -----------------------------
# Constants
# -----------------------------
SCHEMA_VERSION = "v2"
AI_MODEL = "gpt-4.1-mini"


# -----------------------------
# Helpers
# -----------------------------
def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def load_profile(profile_path: Path) -> Dict[str, Any]:
    if not profile_path.exists():
        raise FileNotFoundError(f"{profile_path} not found.")
    with profile_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_job_files(jobs_dir: Path) -> List[JobPosting]:
    if not jobs_dir.exists():
        logger.warning("%s folder not found.", jobs_dir)
        return []

    job_files = sorted(jobs_dir.glob("*.txt"))
    if not job_files:
        logger.warning("No job files found in %s.", jobs_dir)
        return []

    jobs: List[JobPosting] = []
    for file_path in job_files:
        try:
            content = file_path.read_text(encoding="utf-8").strip()
            if not content:
                logger.warning("%s is empty. Skipping.", file_path.name)
                continue
            jobs.append(JobPosting(filename=file_path.name, raw_text=content))
        except Exception as exc:
            logger.error("Failed to read %s: %s", file_path.name, exc)

    return jobs


# -----------------------------
# Scoring (hybrid deterministic V2)
# -----------------------------
def normalize_token(value: str) -> str:
    return normalize(value).replace("-", " ")


def apply_synonym_map(tokens: List[str], profile: Dict[str, Any]) -> List[str]:
    synonym_map = profile.get("normalization", {}).get("synonym_map", {}) or {}
    normalized: List[str] = []

    for token in tokens:
        clean = normalize_token(token)
        mapped = synonym_map.get(clean, clean)
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


def score_role_fit(job: JobPosting, profile: Dict[str, Any]) -> tuple[float, List[str]]:
    rules = profile.get("scoring_rules", {}).get("role_fit", {}) or {}
    role_level = (job.role_level or "unknown").lower()

    score = float(rules.get(role_level, rules.get("unknown", 0)))
    reasons = [f"Role level detected as '{role_level}'"]

    if job.program_type == "grad_program":
        score += 2
        reasons.append("Graduate program structure detected")

    return min(score, float(profile.get("weights", {}).get("role_fit", 25))), reasons


def score_tech_match(job: JobPosting, profile: Dict[str, Any]) -> tuple[float, List[str], List[str]]:
    max_score = float(profile.get("weights", {}).get("tech_match", 30))
    job_stack = apply_synonym_map(job.tech_stack or [], profile)
    if not job_stack:
        return max_score * 0.35, ["No clear tech stack extracted; assigning conservative partial score"], []

    buckets = get_profile_skill_buckets(profile)

    strong_hits = [t for t in job_stack if t in buckets["strong"]]
    working_hits = [t for t in job_stack if t in buckets["working"]]
    basic_hits = [t for t in job_stack if t in buckets["basic"]]
    missing = [t for t in job_stack if t not in buckets["all"]]

    matched_points = (
        len(strong_hits) * 1.0 +
        len(working_hits) * 0.7 +
        len(basic_hits) * 0.4
    )
    possible_points = max(len(job_stack), 1)
    ratio = min(matched_points / possible_points, 1.0)
    score = round(max_score * ratio, 1)

    reasons: List[str] = []
    if strong_hits:
        reasons.append(f"Strong matches: {', '.join(strong_hits)}")
    if working_hits:
        reasons.append(f"Working matches: {', '.join(working_hits)}")
    if basic_hits:
        reasons.append(f"Basic matches: {', '.join(basic_hits)}")
    if missing:
        reasons.append(f"Missing extracted skills: {', '.join(missing)}")

    return score, reasons, missing


def score_location_fit(job: JobPosting, profile: Dict[str, Any]) -> tuple[float, List[str]]:
    rules = profile.get("scoring_rules", {}).get("location", {}) or {}
    location_text = normalize(job.location or "")
    work_mode = (job.work_mode or "unknown").lower()

    reasons: List[str] = []

    if "sydney" in location_text and work_mode == "hybrid":
        return float(rules.get("hybrid_sydney", 15)), ["Hybrid Sydney role"]
    if "sydney" in location_text:
        return float(rules.get("sydney", 15)), ["Sydney-based role"]
    if work_mode == "remote":
        return float(rules.get("remote", 14)), ["Remote role"]
    if work_mode == "hybrid":
        return float(rules.get("australia_other_hybrid", 10)), ["Hybrid role outside Sydney or unspecified city"]
    if work_mode == "onsite":
        return float(rules.get("australia_other_onsite", 6)), ["Onsite role outside Sydney or unspecified city"]

    reasons.append("Location/work mode unclear")
    return float(rules.get("unknown", 8)), reasons


def score_growth(job: JobPosting, profile: Dict[str, Any]) -> tuple[float, List[str]]:
    max_score = float(profile.get("weights", {}).get("growth", 10))
    growth_items = [normalize_token(x) for x in (job.growth_signals or [])]

    reasons: List[str] = []
    score = 0.0

    growth_weights = {
        "mentorship": 3.0,
        "mentoring": 3.0,
        "rotations": 4.0,
        "rotation": 4.0,
        "training": 2.0,
        "coaching": 2.0,
        "career progression": 2.0,
        "career growth": 2.0,
        "support and development": 2.0,
        "learning & development platforms": 2.0,
        "professional skills building": 2.0,
        "security rotation": 2.0,
        "software rotation": 2.0,
    }

    for item in growth_items:
        for key, value in growth_weights.items():
            if key in item:
                score += value
                reasons.append(f"Growth signal detected: {item}")
                break

    if job.program_type == "grad_program":
        score += 2
        reasons.append("Structured graduate program detected")

    return min(round(score, 1), max_score), reasons


def score_company_quality(job: JobPosting, profile: Dict[str, Any]) -> tuple[float, List[str]]:
    max_score = float(profile.get("weights", {}).get("company_quality", 15))
    company = normalize(job.company or "")
    text = normalize(job.raw_text)

    score = 0.0
    reasons: List[str] = []

    reputable_signals = [
        "microsoft", "google", "amazon", "atlassian", "kpmg", "deloitte",
        "ey", "pwc", "accenture", "lockheed", "arup", "government",
        "department", "commbank", "westpac", "nab", "anz"
    ]
    maturity_signals = [
        "graduate program", "mentoring", "coaching", "training",
        "hybrid", "engineering lifecycle", "learning & development",
        "real client work", "performance development"
    ]

    if any(signal in company for signal in reputable_signals):
        score += 8
        reasons.append("Reputable employer signal detected")
    elif any(signal in text for signal in ["global", "enterprise", "consultancy", "government"]):
        score += 6
        reasons.append("Broad reputable/enterprise signal detected")

    maturity_hits = [s for s in maturity_signals if s in text]
    if maturity_hits:
        score += min(len(maturity_hits), 4) * 1.5
        reasons.append(f"Engineering maturity / structure signals: {', '.join(maturity_hits[:4])}")

    return min(round(score, 1), max_score), reasons


def score_risk(job: JobPosting, profile: Dict[str, Any]) -> tuple[float, List[str], List[str]]:
    max_score = float(profile.get("weights", {}).get("risk", 5))
    score = max_score
    reasons: List[str] = []
    flags: List[str] = []

    risk_flags = [normalize_token(x) for x in (job.risk_flags or [])]
    company = normalize(job.company or "")
    contract_type = (job.contract_type or "unknown").lower()
    location_text = normalize(job.location or "")
    work_mode = (job.work_mode or "unknown").lower()

    hard_no = [normalize_token(x) for x in profile.get("scoring_rules", {}).get("hard_no", []) or []]
    for flag in risk_flags:
        if flag in hard_no:
            flags.append(f"Hard no: {flag}")
            reasons.append(f"Hard rejection triggered by '{flag}'")
            return 0.0, reasons, flags

    # Contract penalty is contextual
    if contract_type == "contract":
        if any(x in company for x in ["microsoft", "google", "amazon", "atlassian"]):
            penalty = 0
            reasons.append("Contract role offset by major reputable employer")
        elif any(x in company for x in ["kpmg", "deloitte", "ey", "pwc", "accenture", "lockheed", "arup"]):
            penalty = 1
            reasons.append("Small contract penalty due to reputable employer")
        else:
            penalty = 2
            reasons.append("Contract role penalty applied")
        score -= penalty
        flags.append("Contract role")

    # Onsite outside Sydney penalty
    if work_mode == "onsite" and "sydney" not in location_text and location_text:
        score -= 2
        flags.append("Onsite outside Sydney")
        reasons.append("Onsite role outside Sydney")

    # Clearance is not a penalty for you
    if job.clearance_required:
        reasons.append("Clearance required, but no penalty applied")

    return max(round(score, 1), 0.0), reasons, flags


def score_job_hybrid(job: JobPosting, profile: Dict[str, Any]) -> Dict[str, Any]:
    role_score, role_reasons = score_role_fit(job, profile)
    tech_score, tech_reasons, missing_skills = score_tech_match(job, profile)
    location_score, location_reasons = score_location_fit(job, profile)
    company_score, company_reasons = score_company_quality(job, profile)
    growth_score, growth_reasons = score_growth(job, profile)
    risk_score, risk_reasons, flags = score_risk(job, profile)

    hard_reject = any(flag.lower().startswith("hard no:") for flag in flags)

    if hard_reject:
        total = 0.0
    else:
        total = round(
            role_score + tech_score + location_score + company_score + growth_score + risk_score,
            1
        )

    score_breakdown = {
        "role_fit": round(role_score, 1),
        "tech_match": round(tech_score, 1),
        "location": round(location_score, 1),
        "company_quality": round(company_score, 1),
        "growth": round(growth_score, 1),
        "risk": round(risk_score, 1),
    }

    reasons = {
        "role_fit": role_reasons,
        "tech_match": tech_reasons,
        "location": location_reasons,
        "company_quality": company_reasons,
        "growth": growth_reasons,
        "risk": risk_reasons,
    }

    return {
        "score": total,
        "score_breakdown": score_breakdown,
        "missing_skills": missing_skills,
        "flags": flags,
        "reasons": reasons,
    }


# -----------------------------
# AI Extraction (cached)
# -----------------------------
def hash_job(text: str) -> str:
    return hashlib.sha256((text + SCHEMA_VERSION).encode("utf-8")).hexdigest()


def build_openai_client() -> OpenAI | None:
    load_dotenv(dotenv_path=".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def extract_job_with_ai(
    client: OpenAI,
    job: JobPosting,
    cache_dir: Path,
) -> Dict[str, Any]:
    """
    AI-extract structured fields from a job ad.
    Uses cache/sha256(job_text + schema_version).json to avoid repeat calls.
    Returns {} if extraction fails.
    """
    job_hash = hash_job(job.raw_text)
    cache_file = cache_dir / f"{job_hash}.json"

    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Corrupted cache file %s. Regenerating.", cache_file.name)
        except Exception as exc:
            logger.warning("Failed reading cache file %s: %s. Regenerating.", cache_file.name, exc)

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

            "tech_stack": {"type": "array", "items": {"type": "string"}},
            "must_haves": {"type": "array", "items": {"type": "string"}},
            "nice_to_haves": {"type": "array", "items": {"type": "string"}},
            "responsibilities": {"type": "array", "items": {"type": "string"}},
            "growth_signals": {"type": "array", "items": {"type": "string"}},

            "citizenship_required": {"type": ["boolean", "null"]},
            "clearance_required": {"type": ["boolean", "null"]},

            "risk_flags": {"type": "array", "items": {"type": "string"}},

        "evidence": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "properties": {
                "role_level": {"type": "array", "items": {"type": "string"}},
                "program_type": {"type": "array", "items": {"type": "string"}},
                "work_mode": {"type": "array", "items": {"type": "string"}},
                "contract_type": {"type": "array", "items": {"type": "string"}},
                "tech_stack": {"type": "array", "items": {"type": "string"}},
                "growth_signals": {"type": "array", "items": {"type": "string"}},
                "citizenship_required": {"type": "array", "items": {"type": "string"}},
                "clearance_required": {"type": "array", "items": {"type": "string"}},
                "risk_flags": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "role_level",
                "program_type",
                "work_mode",
                "contract_type",
                "tech_stack",
                "growth_signals",
                "citizenship_required",
                "clearance_required",
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
            "tech_stack",
            "must_haves",
            "nice_to_haves",
            "responsibilities",
            "growth_signals",
            "citizenship_required",
            "clearance_required",
            "risk_flags",
            "evidence",
            "summary",
        ],
    }

    prompt = (
        "Extract structured information from this job advertisement.\n"
        "Return valid JSON only.\n\n"

        "Rules:\n"
        "- If title/company/location not present, return null.\n"
        "- role_level must be one of: graduate, entry, junior, mid, senior, unknown.\n"
        "- program_type must be one of: grad_program, standard_role, unknown.\n"
        "- work_mode must be one of: onsite, hybrid, remote, unknown.\n"
        "- contract_type must be one of: permanent, contract, internship, unknown.\n"
        "- If role_level, program_type, work_mode, or contract_type cannot be determined from the ad, return unknown.\n"
        "- tech_stack should be a normalized list of technologies/tools explicitly mentioned or clearly implied.\n"
        '- Normalize technologies to short canonical names where possible, e.g. "Amazon Web Services" -> "aws", '
        '"Node.js" -> "node", "JavaScript" -> "javascript", "CI/CD" -> "ci/cd".\n'
        "- must_haves and nice_to_haves should be short skill phrases.\n"
        "- responsibilities should be concise bullet-like strings.\n"
        "- growth_signals should include only clear development signals such as mentorship, rotations, training, "
        "certifications, career progression, security_rotation, software_rotation.\n"
        "- citizenship_required should be true only if the ad explicitly requires citizenship or a specific nationality.\n"
        "- clearance_required should be true only if the ad explicitly mentions security clearance or eligibility for clearance.\n"
        "- contract_type should reflect whether the role is permanent, contract, internship, or unknown.\n"
        "- risk_flags should only include objective concerns explicitly supported by the ad, such as unpaid, "
        "commission_only, or extreme_hours.\n"
        "- evidence should contain short supporting quotes/snippets from the job ad for key extracted fields when available.\n"
        "- summary should be 1-3 sentences.\n"
        "- Do not invent details that are not present in the job ad.\n\n"

        "Suggested evidence structure:\n"
        "{\n"
        '  "role_level": ["snippet"],\n'
        '  "program_type": ["snippet"],\n'
        '  "work_mode": ["snippet"],\n'
        '  "contract_type": ["snippet"],\n'
        '  "tech_stack": ["snippet"],\n'
        '  "growth_signals": ["snippet"],\n'
        '  "citizenship_required": ["snippet"],\n'
        '  "clearance_required": ["snippet"],\n'
        '  "risk_flags": ["snippet"]\n'
        "}\n\n"

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


def enrich_jobs_with_ai(
    jobs: List[JobPosting],
    client: OpenAI,
    cache_dir: Path,
) -> None:
    """
    Mutates JobPosting objects in-place with AI extracted fields.
    """
    for job in jobs:
        ai_data = extract_job_with_ai(client=client, job=job, cache_dir=cache_dir)
        if not ai_data:
            continue

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

        job.tech_stack = ai_data.get("tech_stack", []) or []
        job.growth_signals = ai_data.get("growth_signals", []) or []

        job.citizenship_required = ai_data.get("citizenship_required")
        job.clearance_required = ai_data.get("clearance_required")

        job.contract_type = ai_data.get("contract_type")
        job.risk_flags = ai_data.get("risk_flags", []) or []

        job.evidence = ai_data.get("evidence")


# -----------------------------
# Report output
# -----------------------------
def write_results_md(
    scored: List[Tuple[JobPosting, Dict[str, Any]]],
    profile: Dict[str, Any],
    out_dir: Path,
    top_n: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.md"
    run_date = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines: List[str] = []
    lines.append("# JobFit Analysis")
    lines.append(f"Run date: {run_date}")
    lines.append(f"Jobs processed: {len(scored)}")
    lines.append("")

    if not scored:
        lines.append("No jobs were scored.")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path

    for idx, (job, result) in enumerate(scored[:top_n], start=1):
        title = job.title or job.filename.replace(".txt", "")
        company = job.company or "Unknown"
        location = job.location or "Unknown"

        breakdown = job.score_breakdown or {}
        flags = result.get("flags") or []
        reasons = result.get("reasons") or {}
        missing_skills = job.missing_skills or []

        lines.append(f"## {idx}) {title} — {company}")
        lines.append(f"**Score:** {job.score}/100")
        lines.append(f"**Location:** {location}")
        lines.append(f"**Work mode:** {job.work_mode or 'Unknown'}")
        lines.append(f"**Role level:** {job.role_level or 'Unknown'}")
        lines.append(f"**Program type:** {job.program_type or 'Unknown'}")
        lines.append("")

        lines.append("### Breakdown")
        lines.append(f"- Role Fit: {breakdown.get('role_fit', 0)}/25")
        lines.append(f"- Tech Match: {breakdown.get('tech_match', 0)}/30")
        lines.append(f"- Location: {breakdown.get('location', 0)}/15")
        lines.append(f"- Company/Program: {breakdown.get('company_quality', 0)}/15")
        lines.append(f"- Growth: {breakdown.get('growth', 0)}/10")
        lines.append(f"- Risk: {breakdown.get('risk', 0)}/5")
        lines.append("")

        lines.append("### Summary")
        if job.summary:
            lines.append(job.summary.strip())
        else:
            preview = job.raw_text[:300].replace("\n", " ").strip()
            lines.append(preview + ("..." if len(job.raw_text) > 300 else ""))
        lines.append("")

        lines.append("### Missing Skills")
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
            category_reasons = reasons.get(key, []) or []
            lines.append(f"**{category_label}**")
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
    parser = argparse.ArgumentParser(description="JobFit Automator (V1)")
    parser.add_argument("--inbox", type=Path, default=Path("jobs_inbox"), help="Folder containing *.txt job ads")
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

    lines = []

    while True:
        line = input()
        if line.strip() == "::end":
            break
        lines.append(line)

    return "\n".join(lines)

# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    args.cache.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    logger.info("JobFit Automator — V1 (Rules + Optional AI Enrichment)")
    profile = load_profile(args.profile)

    job_text = read_pasted_job()

    job = JobPosting(
        filename="pasted_job",
        raw_text=job_text
    )

    jobs = [job]

    client = None if args.no_ai else build_openai_client()

    if client is None:
        logger.info("AI enrichment disabled (no key or --no-ai).")
    else:
        logger.info("AI enrichment enabled (cached).")
        enrich_jobs_with_ai(jobs, client=client, cache_dir=args.cache)

    scored: List[Tuple[JobPosting, Dict[str, Any]]] = []

    for job in jobs:
        result = score_job_hybrid(job, profile)
        job.score = result["score"]
        job.score_breakdown = result["score_breakdown"]
        job.missing_skills = result["missing_skills"]
        scored.append((job, result))

    for i, (job, result) in enumerate(scored, start=1):
        title = job.title or job.filename.replace(".txt", "")
        company = job.company or "Unknown"
        location = job.location or "Unknown"
        breakdown = job.score_breakdown or {}
        flags = result.get("flags") or []
        reasons = result.get("reasons") or {}

        print(f"\nJob: {title} — {company}")
        print(f"Score: {job.score}/100")
        print(f"Location: {location}")
        print(f"Work mode: {job.work_mode or 'Unknown'}")
        print(f"Role level: {job.role_level or 'Unknown'}")
        print(f"Program type: {job.program_type or 'Unknown'}")

        print("\nBreakdown")
        print(f"- Role Fit: {breakdown.get('role_fit', 0)}/25")
        print(f"- Tech Match: {breakdown.get('tech_match', 0)}/30")
        print(f"- Location: {breakdown.get('location', 0)}/15")
        print(f"- Company/Program: {breakdown.get('company_quality', 0)}/15")
        print(f"- Growth: {breakdown.get('growth', 0)}/10")
        print(f"- Risk: {breakdown.get('risk', 0)}/5")

    print("\nMissing Skills")
    if job.missing_skills:
        for skill in job.missing_skills:
            print(f"- {skill}")
    else:
        print("- None identified")

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

    if args.verbose:
        print("\nRaw extracted fields")
        print(f"- Tech stack: {job.tech_stack}")
        print(f"- Growth signals: {job.growth_signals}")
        print(f"- Contract type: {job.contract_type}")
        print(f"- Clearance required: {job.clearance_required}")
        print(f"- Citizenship required: {job.citizenship_required}")

    scored.sort(key=lambda x: (x[0].score or 0, x[0].filename), reverse=True)

    report_path = write_results_md(scored, profile, out_dir=args.out, top_n=args.top)
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
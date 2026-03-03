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
SCHEMA_VERSION = "v1"
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
# Scoring (deterministic rules)
# -----------------------------
def score_job_rules(job: JobPosting, profile: Dict[str, Any]) -> Dict[str, Any]:
    text = normalize(job.raw_text)

    scoring = profile.get("scoring", {})
    must = [k.lower() for k in scoring.get("must_have_keywords", [])]
    nice = [k.lower() for k in scoring.get("nice_to_have_keywords", [])]

    constraints = profile.get("constraints", {})
    exclude_phrases = [p.lower() for p in constraints.get("exclude_if_contains", [])]

    flags: List[str] = []
    for phrase in exclude_phrases:
        if phrase and phrase in text:
            flags.append(f"EXCLUDE: contains '{phrase}'")

    must_hits = [k for k in must if k and k in text]
    nice_hits = [k for k in nice if k and k in text]

    # forgiving scoring
    must_score = min(len(must_hits) * 12, 60)  # up to 60
    nice_score = min(len(nice_hits) * 5, 30)   # up to 30
    base_score = 10                             # baseline
    score = round(base_score + must_score + nice_score, 1)

    # If we found an exclude flag, force low score (still include in output)
    if flags:
        score = min(score, 15.0)

    return {
        "score": score,
        "must_hits": must_hits,
        "nice_hits": nice_hits,
        "flags": flags,
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
            "must_haves": {"type": "array", "items": {"type": "string"}},
            "nice_to_haves": {"type": "array", "items": {"type": "string"}},
            "responsibilities": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": ["string", "null"]},
        },
        "required": [
            "title",
            "company",
            "location",
            "must_haves",
            "nice_to_haves",
            "responsibilities",
            "summary",
        ],
    }

    prompt = (
        "Extract structured information from this job advertisement.\n"
        "Rules:\n"
        "- If title/company/location not present, return null.\n"
        "- must_haves and nice_to_haves should be short skill phrases.\n"
        "- responsibilities should be concise bullet-like strings.\n"
        "- summary should be 1–3 sentences.\n\n"
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

    scoring = profile.get("scoring", {})
    must = scoring.get("must_have_keywords", [])
    nice = scoring.get("nice_to_have_keywords", [])

    lines: List[str] = []
    lines.append("# JobFit Automator — Ranked Shortlist")
    lines.append(f"Run date: {run_date}")
    lines.append(f"Jobs processed: {len(scored)}")
    lines.append(f"Top N: {top_n}")
    lines.append("")
    lines.append("## Profile snapshot")
    lines.append(f"- Must-have keywords: {', '.join(must[:10])}{'...' if len(must) > 10 else ''}")
    lines.append(f"- Nice-to-have keywords: {', '.join(nice[:10])}{'...' if len(nice) > 10 else ''}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Top matches")
    lines.append("")

    for idx, (job, result) in enumerate(scored[:top_n], start=1):
        title = job.title or job.filename.replace(".txt", "")
        company = job.company or "Unknown"
        location = job.location or "Unknown"

        lines.append(f"### {idx}) {title} — {company} (Score: {job.score}/100)")
        lines.append(f"**Location:** {location}")
        lines.append("")

        lines.append("**Summary**")
        if job.summary:
            lines.append(job.summary.strip())
        else:
            preview = job.raw_text[:300].replace("\n", " ").strip()
            lines.append(preview + ("..." if len(job.raw_text) > 300 else ""))
        lines.append("")

        must_hits = result.get("must_hits", [])
        nice_hits = result.get("nice_hits", [])
        lines.append(f"**Must hits:** {', '.join(must_hits) if must_hits else 'None'}")
        lines.append(f"**Nice hits:** {', '.join(nice_hits) if nice_hits else 'None'}")

        flags = result.get("flags") or []
        if flags:
            lines.append(f"**Flags:** {', '.join(flags)}")

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
    jobs = load_job_files(args.inbox)

    logger.info("Total jobs loaded: %d", len(jobs))
    if not jobs:
        return

    # AI enrichment (cached) unless disabled
    client = None if args.no_ai else build_openai_client()
    if client is None:
        logger.info("AI enrichment disabled (no key or --no-ai).")
    else:
        logger.info("AI enrichment enabled (cached).")
        enrich_jobs_with_ai(jobs, client=client, cache_dir=args.cache)

    scored: List[Tuple[JobPosting, Dict[str, Any]]] = []
    for job in jobs:
        result = score_job_rules(job, profile)
        job.score = result["score"]
        scored.append((job, result))

    # Stable sort: score desc, then filename asc
    scored.sort(key=lambda x: (x[0].score or 0, x[0].filename), reverse=True)

    # Minimal console output (clean)
    for i, (job, result) in enumerate(scored, start=1):
        preview_src = job.summary if job.summary else job.raw_text[:160]
        preview = preview_src.replace("\n", " ").strip()
        print(f"{i}) {job.filename} — Score: {job.score}/100")
        if args.verbose:
            print(f"   Must hits: {result.get('must_hits', [])}")
            print(f"   Nice hits: {result.get('nice_hits', [])}")
            if result.get("flags"):
                print(f"   Flags: {result.get('flags')}")
            print(f"   Preview: {preview}...\n")

    report_path = write_results_md(scored, profile, out_dir=args.out, top_n=args.top)
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
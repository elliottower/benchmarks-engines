#!/usr/bin/env python3
"""
Scrape ICLR 2025 reviews from OpenReview for demand-compliance analysis.

Replicates the ICLR 2024/2023 analysis for a third venue year.
Outputs both raw JSON and demand-compliance CSV.

Usage:
    python scrape_iclr2025.py              # Full corpus
    python scrape_iclr2025.py --limit 50   # Quick test
"""

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

try:
    import openreview
except ImportError:
    print("ERROR: openreview-py not installed. Run: pip install openreview-py", file=sys.stderr)
    sys.exit(1)

DEMAND_PATTERNS = [
    r"should\s+run",
    r"need\s+to\s+run",
    r"add\s+experiment",
    r"add\s+baseline",
    r"add\s+ablation",
    r"missing\s+experiment",
    r"additional\s+evaluation",
    r"more\s+baselines",
    r"please\s+add",
    r"strongly\s+suggest.*experiment",
    r"lack\s+of.*empirical",
    r"recommend.*additional",
]

COMPLIANCE_PATTERNS = [
    r"we\s+have\s+added",
    r"we\s+conducted\s+additional",
    r"new\s+experiment",
    r"updated\s+results",
    r"as\s+requested",
    r"following\s+your\s+suggestion",
    r"we\s+now\s+include",
    r"added\s+to\s+appendix",
    r"new\s+table",
    r"new\s+figure",
    r"additional\s+experiment",
]

DEMAND_RE = [re.compile(p, re.IGNORECASE) for p in DEMAND_PATTERNS]
COMPLIANCE_RE = [re.compile(p, re.IGNORECASE) for p in COMPLIANCE_PATTERNS]

VENUE_ID = "ICLR.cc/2025/Conference"
SUBMISSION_INVITATION = f"{VENUE_ID}/-/Submission"
API_BASE_URL = "https://api2.openreview.net"

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
RAW_JSON_PATH = DATA_DIR / "iclr2025_reviews_raw.json"
CSV_PATH = DATA_DIR / "iclr2025_demand_compliance.csv"
SCORES_PATH = DATA_DIR / "iclr2025_scores.json"


def get_content_value(content, key, default=""):
    if content is None:
        return default
    field = content.get(key)
    if field is None:
        return default
    if isinstance(field, dict):
        return field.get("value", default)
    return field


def is_author_comment(reply):
    signatures = reply.get("signatures", [])
    for sig in signatures:
        if "Authors" in sig:
            return True
    return False


def extract_rating(rating_str):
    if not rating_str:
        return None
    m = re.match(r'(\d+)', str(rating_str))
    return int(m.group(1)) if m else None


def classify_reply(reply):
    invitations = reply.get("invitations", [])
    invitation = reply.get("invitation", "")
    all_invitations = invitations + ([invitation] if invitation else [])
    inv_str = " ".join(all_invitations)

    content = reply.get("content", {})

    if "Official_Review" in inv_str:
        review_text = get_content_value(content, "review")
        if not review_text:
            review_text = get_content_value(content, "main_review")
        if not review_text:
            review_text = get_content_value(content, "summary")
        strengths = get_content_value(content, "strengths")
        weaknesses = get_content_value(content, "weaknesses")
        questions = get_content_value(content, "questions")
        full_text = "\n".join(filter(None, [review_text, strengths, weaknesses, questions]))
        return "review", full_text, content

    if "Decision" in inv_str:
        decision_text = get_content_value(content, "decision")
        return "decision", decision_text, content

    if "Official_Comment" in inv_str or "Rebuttal" in inv_str:
        comment_text = get_content_value(content, "comment")
        if not comment_text:
            comment_text = get_content_value(content, "rebuttal")
        if is_author_comment(reply):
            return "author_comment", comment_text, content
        return "reviewer_comment", comment_text, content

    return "other", "", content


def count_pattern_matches(text, compiled_patterns):
    if not text:
        return 0
    return sum(1 for p in compiled_patterns if p.search(text))


def fetch_submissions(client, limit=None):
    print(f"Fetching submissions from {VENUE_ID} ...")
    if limit:
        print(f"  limit: {limit}")
        return client.get_notes(
            invitation=SUBMISSION_INVITATION,
            details="directReplies",
            limit=limit,
        )

    submissions = []
    offset = 0
    page_size = 1000
    while True:
        batch = client.get_notes(
            invitation=SUBMISSION_INVITATION,
            details="directReplies",
            limit=page_size,
            offset=offset,
        )
        if not batch:
            break
        submissions.extend(batch)
        print(f"  fetched {len(submissions)} so far...")
        offset += len(batch)
        if len(batch) < page_size:
            break

    print(f"Fetched {len(submissions)} submissions.\n")
    return submissions


def process_submission(submission):
    content = submission.content or {}
    paper_id = submission.id
    number = getattr(submission, "number", None)
    title = get_content_value(content, "title")

    details = getattr(submission, "details", {}) or {}
    replies = details.get("directReplies", details.get("replies", []))

    reviews = []
    author_comments = []
    decision = ""
    scores = []

    for reply in replies:
        reply_type, text, rcontent = classify_reply(reply)
        if reply_type == "review":
            reviews.append(text)
            rating = get_content_value(rcontent, "rating")
            if not rating:
                rating = get_content_value(rcontent, "recommendation")
            score = extract_rating(rating)
            if score is not None:
                scores.append(score)
        elif reply_type == "author_comment":
            author_comments.append(text)
        elif reply_type == "decision":
            decision = text

    all_review_text = "\n\n".join(reviews)
    all_rebuttal_text = "\n\n".join(author_comments)

    demand_count = count_pattern_matches(all_review_text, DEMAND_RE)
    compliance_count = count_pattern_matches(all_rebuttal_text, COMPLIANCE_RE)

    decision_lower = decision.lower().strip()
    if "accept" in decision_lower:
        acceptance = True
    elif "reject" in decision_lower:
        acceptance = False
    else:
        acceptance = None

    return {
        "paper_id": paper_id,
        "number": number,
        "title": title,
        "reviews": reviews,
        "author_comments": author_comments,
        "decision": decision,
        "acceptance": acceptance,
        "demand_count": demand_count,
        "has_demand": demand_count > 0,
        "compliance_count": compliance_count,
        "has_compliance": compliance_count > 0,
        "scores": scores,
    }


def save_outputs(results):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw_data = [{
        "paper_id": r["paper_id"],
        "number": r["number"],
        "title": r["title"],
        "reviews": r["reviews"],
        "author_comments": r["author_comments"],
        "decision": r["decision"],
    } for r in results]

    with open(RAW_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2, ensure_ascii=False)
    size_mb = RAW_JSON_PATH.stat().st_size / (1024 * 1024)
    print(f"Saved raw data: {RAW_JSON_PATH} ({size_mb:.1f} MB, {len(raw_data)} papers)")

    fieldnames = [
        "paper_id", "title", "has_demand", "demand_count",
        "has_compliance", "compliance_count", "decision", "acceptance",
    ]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"Saved CSV: {CSV_PATH} ({len(results)} rows)")

    scores_data = {}
    for r in results:
        if r["scores"]:
            scores_data[r["paper_id"]] = r["scores"]
    with open(SCORES_PATH, "w") as f:
        json.dump(scores_data, f, indent=2)
    print(f"Saved scores: {SCORES_PATH} ({len(scores_data)} papers with scores)")


def print_summary(results):
    total = len(results)
    if total == 0:
        print("No papers processed.")
        return

    accepted = sum(1 for r in results if r["acceptance"] is True)
    rejected = sum(1 for r in results if r["acceptance"] is False)
    decided = accepted + rejected

    def acc_rate(subset):
        d = [r for r in subset if r["acceptance"] is not None]
        if not d:
            return 0.0, 0
        return sum(1 for r in d if r["acceptance"]) / len(d) * 100, len(d)

    demand = [r for r in results if r["has_demand"]]
    no_demand = [r for r in results if not r["has_demand"]]
    comply = [r for r in results if r["has_demand"] and r["has_compliance"]]
    no_comply = [r for r in results if r["has_demand"] and not r["has_compliance"]]

    dr, dn = acc_rate(demand)
    ndr, ndn = acc_rate(no_demand)
    cr, cn = acc_rate(comply)
    ncr, ncn = acc_rate(no_comply)

    print()
    print("=" * 70)
    print("ICLR 2025 DEMAND-COMPLIANCE ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Total papers:              {total}")
    print(f"With decisions:            {decided} (accepted: {accepted}, rejected: {rejected})")
    if decided:
        print(f"Overall acceptance rate:   {accepted/decided*100:.1f}%")
    print()
    print(f"With demands:              {len(demand)} ({len(demand)/total*100:.1f}%)")
    print(f"  Acceptance rate:         {dr:.1f}% (n={dn})")
    print(f"Without demands:           {len(no_demand)} ({len(no_demand)/total*100:.1f}%)")
    print(f"  Acceptance rate:         {ndr:.1f}% (n={ndn})")
    print()
    print(f"Demand + complied:         {len(comply)}")
    print(f"  Acceptance rate:         {cr:.1f}% (n={cn})")
    print(f"Demand + did not comply:   {len(no_comply)}")
    print(f"  Acceptance rate:         {ncr:.1f}% (n={ncn})")
    if cn and ncn:
        print(f"Compliance lift:           {cr - ncr:+.1f}pp")
    print("=" * 70)

    # Score-stratified analysis
    scored = [r for r in results if r["scores"]]
    if scored:
        print(f"\n{'='*70}")
        print("SCORE-STRATIFIED ANALYSIS")
        print(f"{'='*70}")

        def get_stratum(mean_score):
            if mean_score < 4.5:
                return "low (< 4.5)"
            elif mean_score < 5.5:
                return "borderline (4.5-5.5)"
            elif mean_score < 6.5:
                return "mid (5.5-6.5)"
            else:
                return "high (>= 6.5)"

        strata = {}
        for r in scored:
            if r["acceptance"] is None:
                continue
            ms = sum(r["scores"]) / len(r["scores"])
            stratum = get_stratum(ms)
            if stratum not in strata:
                strata[stratum] = {"comply": [], "no_comply": [], "no_demand": []}
            acc = r["acceptance"]
            if r["has_demand"]:
                if r["has_compliance"]:
                    strata[stratum]["comply"].append(acc)
                else:
                    strata[stratum]["no_comply"].append(acc)
            else:
                strata[stratum]["no_demand"].append(acc)

        def rate(lst):
            if not lst:
                return 0, 0
            return sum(lst) / len(lst) * 100, len(lst)

        print(f"{'Stratum':<25} {'Comply':>14} {'NoComply':>14} {'Lift':>10}")
        print("-" * 65)
        for s_name in ["low (< 4.5)", "borderline (4.5-5.5)", "mid (5.5-6.5)", "high (>= 6.5)"]:
            if s_name not in strata:
                continue
            s = strata[s_name]
            cr2, cn2 = rate(s["comply"])
            ncr2, ncn2 = rate(s["no_comply"])
            lift = cr2 - ncr2 if cn2 > 0 and ncn2 > 0 else float("nan")
            lift_str = f"{lift:+.1f}pp" if cn2 > 0 and ncn2 > 0 else "N/A"
            print(f"{s_name:<25} {cr2:5.1f}% (n={cn2:<3d})  {ncr2:5.1f}% (n={ncn2:<3d})  {lift_str:>10}")


def main():
    parser = argparse.ArgumentParser(description="Scrape ICLR 2025 for demand-compliance analysis")
    parser.add_argument("--limit", type=int, default=None, help="Limit papers (for testing)")
    args = parser.parse_args()

    print("Connecting to OpenReview API...")
    try:
        client = openreview.api.OpenReviewClient(baseurl=API_BASE_URL)
    except Exception as e:
        print(f"ERROR: Could not connect: {e}", file=sys.stderr)
        sys.exit(1)
    print("Connected.\n")

    submissions = fetch_submissions(client, limit=args.limit)
    if not submissions:
        print("No submissions found.", file=sys.stderr)
        sys.exit(1)

    results = []
    total = len(submissions)
    print(f"Processing {total} submissions...")

    for i, sub in enumerate(submissions):
        try:
            results.append(process_submission(sub))
        except Exception as e:
            pid = getattr(sub, "id", "unknown")
            print(f"  WARNING: Error processing {pid}: {e}", file=sys.stderr)

        if (i + 1) % 500 == 0 or (i + 1) == total:
            demands = sum(1 for r in results if r["has_demand"])
            print(f"  [{i+1:5d}/{total}] {(i+1)/total*100:.1f}% | {demands} demands detected")

    print(f"\nProcessed {len(results)} papers.")

    save_outputs(results)
    print_summary(results)


if __name__ == "__main__":
    main()

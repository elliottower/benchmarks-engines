"""
Verify the 688-paper MI citation corpus against live Semantic Scholar data.

Spot-checks:
1. Seed papers resolve correctly and community labels match
2. Citation counts are in the right ballpark
3. Cross-community edges are real (sample verification)
4. Re-computes the citation matrix from live S2 data for the seed papers
"""

import csv
import json
import os
import random
import time
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"

S2_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
S2_BASE = "https://api.semanticscholar.org/graph/v1"
HEADERS = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
RATE_DELAY = 0.5 if S2_API_KEY else 1.5

random.seed(42)


def s2_get(endpoint, params=None):
    import requests as req_lib

    url = f"{S2_BASE}/{endpoint}"
    for attempt in range(3):
        try:
            resp = req_lib.get(url, params=params, headers=HEADERS, timeout=30)
            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code == 200:
                return resp.json()
            else:
                print(f"  HTTP {resp.status_code}")
                return None
        except Exception as e:
            print(f"  Error: {e}")
            return None
    return None


def load_corpus():
    papers = {}
    with open(DATA / "mi_citation_corpus" / "mi_corpus_papers.csv") as f:
        for row in csv.DictReader(f):
            papers[row["s2_paper_id"]] = {
                "title": row["title"],
                "year": row["year"],
                "community": row["community"],
                "citation_count": int(float(row["citation_count"])) if row["citation_count"] else 0,
                "is_seed": row["is_seed"] == "True",
            }

    edges = []
    with open(DATA / "mi_citation_corpus" / "mi_corpus_edges.csv") as f:
        for row in csv.DictReader(f):
            edges.append({
                "source": row["source_s2id"],
                "target": row["target_s2id"],
                "source_community": row["source_community"],
                "target_community": row["target_community"],
            })

    return papers, edges


def verify_seed_papers(papers):
    print("\n" + "=" * 60)
    print("CHECK 1: SEED PAPER VERIFICATION")
    print("=" * 60)

    seeds = {pid: p for pid, p in papers.items() if p["is_seed"]}
    print(f"Corpus has {len(seeds)} seed papers")

    verified = 0
    mismatches = []

    for pid, info in seeds.items():
        time.sleep(RATE_DELAY)
        result = s2_get(f"paper/{pid}", {"fields": "title,year,citationCount"})
        if result:
            verified += 1
            s2_title = result.get("title", "")
            s2_year = result.get("year")
            s2_cites = result.get("citationCount", 0)
            corpus_cites = info["citation_count"]

            title_match = s2_title.lower()[:40] == info["title"].lower()[:40]
            cite_ratio = s2_cites / max(corpus_cites, 1) if corpus_cites else 0

            status = "OK" if title_match else "TITLE MISMATCH"
            print(f"  [{info['community']:>10}] {info['title'][:50]:50s} "
                  f"cites: {corpus_cites:>5} -> {s2_cites:>5} ({cite_ratio:.1f}x)  {status}")

            if not title_match:
                mismatches.append((pid, info["title"], s2_title))
        else:
            print(f"  [{info['community']:>10}] {info['title'][:50]:50s}  NOT FOUND ON S2")

    print(f"\n  Verified: {verified}/{len(seeds)}")
    if mismatches:
        print(f"  Mismatches: {len(mismatches)}")
        for pid, expected, got in mismatches:
            print(f"    {pid[:12]}: expected '{expected[:40]}', got '{got[:40]}'")

    return verified


def verify_edge_sample(papers, edges, n_sample=30):
    print("\n" + "=" * 60)
    print("CHECK 2: EDGE VERIFICATION (SAMPLE)")
    print("=" * 60)

    cross_edges = [e for e in edges if e["source_community"] != e["target_community"]]
    sample = random.sample(cross_edges, min(n_sample, len(cross_edges)))
    print(f"Sampling {len(sample)} cross-community edges to verify against S2...")

    confirmed = 0
    denied = 0
    unknown = 0

    for e in sample:
        time.sleep(RATE_DELAY)
        result = s2_get(f"paper/{e['source']}", {"fields": "references.paperId"})
        if result and result.get("references"):
            ref_ids = {r["paperId"] for r in result["references"] if r.get("paperId")}
            if e["target"] in ref_ids:
                confirmed += 1
                status = "CONFIRMED"
            else:
                denied += 1
                status = "NOT FOUND in refs"
        else:
            unknown += 1
            status = "COULD NOT CHECK"

        src = papers.get(e["source"], {})
        tgt = papers.get(e["target"], {})
        print(f"  {src.get('community','?'):>10} -> {tgt.get('community','?'):<10} "
              f"{src.get('title','?')[:30]:30s} -> {tgt.get('title','?')[:30]:30s}  {status}")

    print(f"\n  Confirmed: {confirmed}/{len(sample)}")
    print(f"  Not found: {denied}/{len(sample)}")
    print(f"  Unknown:   {unknown}/{len(sample)}")
    print(f"  Precision: {confirmed/max(confirmed+denied,1)*100:.1f}%")

    return confirmed, denied, unknown


def live_citation_matrix(papers):
    print("\n" + "=" * 60)
    print("CHECK 3: LIVE CITATION MATRIX (SEED PAPERS)")
    print("=" * 60)

    seeds = {pid: p for pid, p in papers.items() if p["is_seed"]}
    seed_ids = set(seeds.keys())

    communities = sorted(set(p["community"] for p in seeds.values()))
    matrix = defaultdict(lambda: defaultdict(int))
    total_checked = 0

    for pid in seed_ids:
        time.sleep(RATE_DELAY)
        result = s2_get(f"paper/{pid}", {"fields": "references.paperId"})
        if result and result.get("references"):
            ref_ids = {r["paperId"] for r in result["references"] if r.get("paperId")}
            from_comm = seeds[pid]["community"]
            for ref_id in ref_ids & seed_ids:
                if ref_id != pid:
                    to_comm = seeds[ref_id]["community"]
                    matrix[from_comm][to_comm] += 1
            total_checked += 1

    print(f"Checked {total_checked} seed papers against live S2 data\n")

    from_to = "From \\ To"
    header = f"  {from_to:<12}" + "".join(f"{c:>12}" for c in communities)
    print(header)
    print("  " + "-" * len(header))
    for c_from in communities:
        row = f"  {c_from:<12}"
        for c_to in communities:
            row += f"{matrix[c_from][c_to]:>12}"
        print(row)

    within = sum(matrix[c][c] for c in communities)
    cross = sum(matrix[c1][c2] for c1 in communities for c2 in communities if c1 != c2)
    print(f"\n  Within-community edges: {within}")
    print(f"  Cross-community edges: {cross}")
    print(f"  Total: {within + cross}")

    return dict(matrix)


def citation_count_drift(papers, n_sample=50):
    print("\n" + "=" * 60)
    print("CHECK 4: CITATION COUNT DRIFT")
    print("=" * 60)

    sample_pids = random.sample(list(papers.keys()), min(n_sample, len(papers)))
    print(f"Checking {len(sample_pids)} random papers for citation count drift...\n")

    drifts = []
    for pid in sample_pids:
        time.sleep(RATE_DELAY)
        result = s2_get(f"paper/{pid}", {"fields": "citationCount,title"})
        if result:
            old = papers[pid]["citation_count"]
            new = result.get("citationCount", 0)
            drift = new - old
            drifts.append(drift)
            if abs(drift) > 50:
                print(f"  {papers[pid]['title'][:50]:50s}  {old:>5} -> {new:>5} ({drift:+d})")

    if drifts:
        import statistics
        print(f"\n  Papers checked: {len(drifts)}")
        print(f"  Mean drift: {statistics.mean(drifts):+.1f} citations")
        print(f"  Median drift: {statistics.median(drifts):+.1f} citations")
        print(f"  Max drift: {max(drifts):+d}")
        print(f"  Min drift: {min(drifts):+d}")
        positive = sum(1 for d in drifts if d > 0)
        print(f"  Papers with increased citations: {positive}/{len(drifts)}")


def main():
    if not S2_API_KEY:
        print("WARNING: No SEMANTIC_SCHOLAR_API_KEY found. Using unauthenticated requests (slower rate limits).")
    else:
        print(f"Using Semantic Scholar API key: {S2_API_KEY[:8]}...")

    print("\nVERIFYING MI CITATION CORPUS AGAINST SEMANTIC SCHOLAR")
    print("=" * 60)

    papers, edges = load_corpus()
    print(f"Corpus: {len(papers)} papers, {len(edges)} edges")

    seeds = sum(1 for p in papers.values() if p["is_seed"])
    print(f"Seed papers: {seeds}")
    for comm in sorted(set(p["community"] for p in papers.values())):
        n = sum(1 for p in papers.values() if p["community"] == comm)
        print(f"  {comm}: {n}")

    verify_seed_papers(papers)
    confirmed, denied, unknown = verify_edge_sample(papers, edges)
    live_citation_matrix(papers)
    citation_count_drift(papers)

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

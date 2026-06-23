"""
Experiment: Author overlap across MI communities.

Uses Semantic Scholar API to get author lists for corpus papers,
then checks whether the same researchers publish across communities.
If authors cross boundaries but citations don't, that's a different
kind of fragmentation than fully separate tribes.
"""

import csv
import json
import os
import time
from collections import defaultdict, Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"

S2_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
S2_BASE = "https://api.semanticscholar.org/graph/v1"
HEADERS = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
RATE_DELAY = 0.3 if S2_API_KEY else 1.5


def s2_get(paper_id):
    import requests
    url = f"{S2_BASE}/paper/{paper_id}"
    params = {"fields": "authors.authorId,authors.name"}
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code == 200:
                return resp.json()
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
                "community": row["community"],
                "is_seed": row["is_seed"] == "True",
            }
    return papers


def main():
    print("EXPERIMENT: AUTHOR OVERLAP ACROSS MI COMMUNITIES")
    print("=" * 60)

    papers = load_corpus()
    seed_papers = {pid: p for pid, p in papers.items() if p["is_seed"]}

    # First do seeds (small, fast), then sample from full corpus
    print(f"\nPhase 1: Fetching authors for {len(seed_papers)} seed papers...")

    author_communities = defaultdict(set)  # author_id -> set of communities
    author_names = {}  # author_id -> name
    paper_authors = {}  # paper_id -> list of author_ids
    community_authors = defaultdict(set)  # community -> set of author_ids

    for pid, info in seed_papers.items():
        time.sleep(RATE_DELAY)
        result = s2_get(pid)
        if result and result.get("authors"):
            aids = []
            for a in result["authors"]:
                if a.get("authorId"):
                    aid = a["authorId"]
                    author_communities[aid].add(info["community"])
                    author_names[aid] = a.get("name", "Unknown")
                    community_authors[info["community"]].add(aid)
                    aids.append(aid)
            paper_authors[pid] = aids

    print(f"  Found {len(author_names)} unique authors across seed papers")

    # Now sample from full corpus
    import random
    random.seed(42)
    non_seed = {pid: p for pid, p in papers.items() if not p["is_seed"]}
    sample_size = min(150, len(non_seed))
    sample_pids = random.sample(list(non_seed.keys()), sample_size)

    print(f"\nPhase 2: Fetching authors for {sample_size} sampled corpus papers...")
    fetched = 0
    for i, pid in enumerate(sample_pids):
        time.sleep(RATE_DELAY)
        result = s2_get(pid)
        if result and result.get("authors"):
            aids = []
            for a in result["authors"]:
                if a.get("authorId"):
                    aid = a["authorId"]
                    author_communities[aid].add(papers[pid]["community"])
                    author_names[aid] = a.get("name", "Unknown")
                    community_authors[papers[pid]["community"]].add(aid)
                    aids.append(aid)
            paper_authors[pid] = aids
            fetched += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{sample_size} fetched...")

    print(f"  Fetched {fetched}/{sample_size}")

    # Analysis
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    total_authors = len(author_names)
    multi_comm = {aid: comms for aid, comms in author_communities.items() if len(comms) > 1}

    print(f"\nTotal unique authors: {total_authors}")
    print(f"Authors in multiple communities: {len(multi_comm)} ({len(multi_comm)/max(total_authors,1)*100:.1f}%)")

    # Breakdown
    print(f"\nAuthors per community:")
    comms = sorted(community_authors.keys())
    for c in comms:
        print(f"  {c}: {len(community_authors[c])} authors")

    # Pairwise author overlap
    print(f"\nPairwise author overlap:")
    print(f"  {'Pair':<25} {'Shared':>8} {'Jaccard':>10}")
    print("  " + "-" * 45)
    for i, c1 in enumerate(comms):
        for c2 in comms[i+1:]:
            shared = community_authors[c1] & community_authors[c2]
            union = community_authors[c1] | community_authors[c2]
            jaccard = len(shared) / len(union) if union else 0
            print(f"  {c1} <-> {c2:<14} {len(shared):>8} {jaccard:>9.3f}")

    # List bridge authors
    if multi_comm:
        print(f"\nBridge authors (publish in 2+ communities):")
        bridge_sorted = sorted(multi_comm.items(), key=lambda x: -len(x[1]))
        for aid, comms_set in bridge_sorted[:20]:
            name = author_names.get(aid, "Unknown")
            print(f"  {name:<30} {', '.join(sorted(comms_set))}")

    # Key metric: do authors cross but citations don't?
    print(f"\n{'='*60}")
    print("KEY FINDING")
    print(f"{'='*60}")
    overlap_rate = len(multi_comm) / max(total_authors, 1) * 100
    print(f"Author overlap rate: {overlap_rate:.1f}%")
    if overlap_rate > 10:
        print("Authors DO cross community boundaries frequently.")
        print("Combined with low cross-citation density, this suggests")
        print("fragmentation is in *reading habits*, not *personnel*.")
    elif overlap_rate > 3:
        print("Moderate author overlap suggests partial community bridging.")
    else:
        print("Very low author overlap — communities are genuinely separate tribes.")

    # Save
    output = DATA / "experiment_author_overlap.json"
    results = {
        "total_authors": total_authors,
        "multi_community_authors": len(multi_comm),
        "overlap_rate_pct": round(overlap_rate, 1),
        "community_author_counts": {c: len(a) for c, a in community_authors.items()},
        "bridge_authors": [
            {"name": author_names[aid], "communities": sorted(comms_set)}
            for aid, comms_set in sorted(multi_comm.items(), key=lambda x: -len(x[1]))[:30]
        ],
    }
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()

"""
Experiment: Co-citation analysis across MI communities.

Checks which papers from different communities get cited *together*
by downstream work. High co-citation + low direct citation =
"same intellectual space, no dialogue."

Uses Semantic Scholar API to get citing papers for seeds, then checks
which seed pairs co-occur in the same reference lists.
"""

import csv
import json
import os
import time
from collections import defaultdict, Counter
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"

S2_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
S2_BASE = "https://api.semanticscholar.org/graph/v1"
HEADERS = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
RATE_DELAY = 0.3 if S2_API_KEY else 1.5


def s2_get(endpoint, params=None):
    import requests
    url = f"{S2_BASE}/{endpoint}"
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
    print("EXPERIMENT: CO-CITATION ANALYSIS")
    print("=" * 60)

    papers = load_corpus()
    seeds = {pid: p for pid, p in papers.items() if p["is_seed"]}
    seed_ids = set(seeds.keys())

    print(f"Seed papers: {len(seeds)}")
    for c in sorted(set(p["community"] for p in seeds.values())):
        n = sum(1 for p in seeds.values() if p["community"] == c)
        print(f"  {c}: {n}")

    # For each seed paper, get its citing papers (who cites it)
    print(f"\nFetching citing papers for each seed...")
    cited_by = defaultdict(set)  # seed_pid -> set of citing paper IDs

    for pid, info in seeds.items():
        time.sleep(RATE_DELAY)
        result = s2_get(f"paper/{pid}", {"fields": "citations.paperId", "limit": "500"})
        if result and result.get("citations"):
            citers = {c["paperId"] for c in result["citations"] if c.get("paperId")}
            cited_by[pid] = citers
            print(f"  [{info['community']:>10}] {info['title'][:45]:45s} cited by {len(citers)} papers")
        else:
            print(f"  [{info['community']:>10}] {info['title'][:45]:45s} no citations found")

    # Co-citation: for each pair of seeds, count papers that cite both
    print(f"\n{'='*60}")
    print("CO-CITATION COUNTS (papers citing both seeds in pair)")
    print(f"{'='*60}")

    cocite_counts = {}
    seed_list = sorted(seed_ids)

    for i, p1 in enumerate(seed_list):
        for p2 in seed_list[i+1:]:
            shared = cited_by[p1] & cited_by[p2]
            if shared:
                cocite_counts[(p1, p2)] = len(shared)

    # Sort by count
    sorted_pairs = sorted(cocite_counts.items(), key=lambda x: -x[1])

    # Separate cross vs within
    cross_pairs = [(pair, count) for pair, count in sorted_pairs
                   if seeds[pair[0]]["community"] != seeds[pair[1]]["community"]]
    within_pairs = [(pair, count) for pair, count in sorted_pairs
                    if seeds[pair[0]]["community"] == seeds[pair[1]]["community"]]

    print(f"\nTop cross-community co-cited pairs:")
    for (p1, p2), count in cross_pairs[:15]:
        c1 = seeds[p1]["community"]
        c2 = seeds[p2]["community"]
        t1 = seeds[p1]["title"][:35]
        t2 = seeds[p2]["title"][:35]
        print(f"  {count:>4} papers cite both: [{c1}] {t1} + [{c2}] {t2}")

    print(f"\nTop within-community co-cited pairs:")
    for (p1, p2), count in within_pairs[:10]:
        c1 = seeds[p1]["community"]
        t1 = seeds[p1]["title"][:35]
        t2 = seeds[p2]["title"][:35]
        print(f"  {count:>4} papers cite both: [{c1}] {t1} + [{c1}] {t2}")

    # Aggregate stats
    total_cross_cocite = sum(c for _, c in cross_pairs)
    total_within_cocite = sum(c for _, c in within_pairs)

    print(f"\n{'='*60}")
    print("AGGREGATE CO-CITATION STATS")
    print(f"{'='*60}")
    print(f"Cross-community co-cited pairs: {len(cross_pairs)}")
    print(f"Within-community co-cited pairs: {len(within_pairs)}")
    print(f"Total cross co-citations: {total_cross_cocite}")
    print(f"Total within co-citations: {total_within_cocite}")
    if total_cross_cocite + total_within_cocite > 0:
        print(f"Cross share: {total_cross_cocite/(total_cross_cocite+total_within_cocite)*100:.1f}%")

    # Pairwise community co-citation
    print(f"\nCo-citation by community pair:")
    comm_pairs = defaultdict(int)
    comm_pair_n = defaultdict(int)
    for (p1, p2), count in sorted_pairs:
        c1 = seeds[p1]["community"]
        c2 = seeds[p2]["community"]
        key = tuple(sorted([c1, c2]))
        comm_pairs[key] += count
        comm_pair_n[key] += 1

    for key in sorted(comm_pairs.keys()):
        print(f"  {key[0]} <-> {key[1]}: {comm_pairs[key]} total co-citations across {comm_pair_n[key]} pairs")

    # Key finding
    print(f"\n{'='*60}")
    print("KEY FINDING")
    print(f"{'='*60}")
    if cross_pairs:
        top_cross = cross_pairs[0]
        p1, p2 = top_cross[0]
        print(f"Strongest cross-community co-citation: {top_cross[1]} papers cite both")
        print(f"  [{seeds[p1]['community']}] {seeds[p1]['title'][:50]}")
        print(f"  [{seeds[p2]['community']}] {seeds[p2]['title'][:50]}")
    if total_cross_cocite > total_within_cocite * 0.3:
        print("\nSubstantial cross-community co-citation despite low direct citation.")
        print("Downstream papers read both communities — the seeds themselves don't.")
    else:
        print("\nCo-citation is also dominated by within-community pairs.")
        print("The fragmentation extends to downstream readership too.")

    # Save
    output = DATA / "experiment_cocitation.json"
    results = {
        "cross_pairs": len(cross_pairs),
        "within_pairs": len(within_pairs),
        "total_cross_cocitations": total_cross_cocite,
        "total_within_cocitations": total_within_cocite,
        "top_cross_pairs": [
            {
                "paper1": {"title": seeds[p1]["title"], "community": seeds[p1]["community"]},
                "paper2": {"title": seeds[p2]["title"], "community": seeds[p2]["community"]},
                "cocitation_count": count,
            }
            for (p1, p2), count in cross_pairs[:20]
        ],
    }
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()

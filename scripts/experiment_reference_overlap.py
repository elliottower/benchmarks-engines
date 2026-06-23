"""
Experiment: Reference overlap across MI communities.

Do SAE and causal papers cite the same foundational work without
citing each other? Shared ancestors + no sibling citations is the
strongest version of the proof-cultures claim.

Uses Semantic Scholar API to get reference lists for seed papers,
then computes Jaccard similarity of reference sets across communities.
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
    print("EXPERIMENT: REFERENCE OVERLAP ACROSS COMMUNITIES")
    print("=" * 60)

    papers = load_corpus()
    seeds = {pid: p for pid, p in papers.items() if p["is_seed"]}
    seed_ids = set(seeds.keys())

    print(f"Seed papers: {len(seeds)}")

    # Fetch reference lists for all seeds
    print(f"\nFetching reference lists from Semantic Scholar...")
    paper_refs = {}  # pid -> set of referenced paper IDs

    for pid, info in seeds.items():
        time.sleep(RATE_DELAY)
        result = s2_get(f"paper/{pid}", {"fields": "references.paperId,references.title"})
        if result and result.get("references"):
            refs = {}
            for r in result["references"]:
                if r.get("paperId"):
                    refs[r["paperId"]] = r.get("title", "Unknown")
            paper_refs[pid] = refs
            print(f"  [{info['community']:>10}] {info['title'][:45]:45s} {len(refs)} refs")
        else:
            paper_refs[pid] = {}
            print(f"  [{info['community']:>10}] {info['title'][:45]:45s} no refs")

    # Aggregate references by community
    community_refs = defaultdict(set)  # community -> set of paper IDs referenced
    for pid, refs in paper_refs.items():
        comm = seeds[pid]["community"]
        community_refs[comm].update(refs.keys())

    comms = sorted(community_refs.keys())
    print(f"\nReferences per community:")
    for c in comms:
        print(f"  {c}: {len(community_refs[c])} unique referenced papers")

    # Pairwise reference overlap
    print(f"\n{'='*60}")
    print("PAIRWISE REFERENCE OVERLAP")
    print(f"{'='*60}")
    print(f"\n  {'Pair':<25} {'Shared':>8} {'Union':>8} {'Jaccard':>10}")
    print("  " + "-" * 55)

    for i, c1 in enumerate(comms):
        for c2 in comms[i+1:]:
            shared = community_refs[c1] & community_refs[c2]
            union = community_refs[c1] | community_refs[c2]
            jaccard = len(shared) / len(union) if union else 0
            print(f"  {c1} <-> {c2:<14} {len(shared):>8} {len(union):>8} {jaccard:>9.3f}")

    # Find the most-shared references (cited by multiple communities)
    ref_communities = defaultdict(set)  # ref_pid -> set of communities that reference it
    ref_titles = {}
    for pid, refs in paper_refs.items():
        comm = seeds[pid]["community"]
        for ref_id, ref_title in refs.items():
            ref_communities[ref_id].add(comm)
            ref_titles[ref_id] = ref_title

    bridge_refs = {rid: comms_set for rid, comms_set in ref_communities.items()
                   if len(comms_set) >= 2}

    # Exclude seed papers themselves (they're the papers we're studying)
    external_bridges = {rid: comms_set for rid, comms_set in bridge_refs.items()
                        if rid not in seed_ids}

    print(f"\n{'='*60}")
    print(f"SHARED ANCESTORS (referenced by 2+ communities, excluding seeds)")
    print(f"{'='*60}")
    print(f"Total shared references: {len(external_bridges)}")

    # Count how many seeds in each community cite each bridge ref
    bridge_cite_counts = {}
    for rid in external_bridges:
        counts = defaultdict(int)
        for pid, refs in paper_refs.items():
            if rid in refs:
                counts[seeds[pid]["community"]] += 1
        bridge_cite_counts[rid] = counts

    # Sort by total citations across communities
    sorted_bridges = sorted(external_bridges.items(),
                           key=lambda x: -sum(bridge_cite_counts[x[0]].values()))

    # Show by number of communities
    three_comm = [(rid, comms_set) for rid, comms_set in sorted_bridges if len(comms_set) >= 3]
    two_comm = [(rid, comms_set) for rid, comms_set in sorted_bridges if len(comms_set) == 2]

    if three_comm:
        print(f"\nReferenced by ALL 3 communities ({len(three_comm)} papers):")
        for rid, comms_set in three_comm[:15]:
            title = ref_titles.get(rid, "Unknown")[:60]
            counts = bridge_cite_counts[rid]
            count_str = ", ".join(f"{c}:{counts[c]}" for c in sorted(counts))
            print(f"  {title:60s} ({count_str})")

    if two_comm:
        print(f"\nReferenced by 2 communities ({len(two_comm)} papers, showing top 15):")
        for rid, comms_set in two_comm[:15]:
            title = ref_titles.get(rid, "Unknown")[:60]
            counts = bridge_cite_counts[rid]
            count_str = ", ".join(f"{c}:{counts[c]}" for c in sorted(counts))
            pair = " <-> ".join(sorted(comms_set))
            print(f"  {title:60s} ({pair}: {count_str})")

    # Key finding
    print(f"\n{'='*60}")
    print("KEY FINDING")
    print(f"{'='*60}")

    sae_refs = community_refs.get("sae", set())
    causal_refs = community_refs.get("causal", set())
    circuits_refs = community_refs.get("circuits", set())

    sae_causal_shared = sae_refs & causal_refs
    sae_causal_union = sae_refs | causal_refs
    sae_causal_jaccard = len(sae_causal_shared) / len(sae_causal_union) if sae_causal_union else 0

    print(f"SAE and Causal share {len(sae_causal_shared)} common references (Jaccard={sae_causal_jaccard:.3f})")

    if three_comm:
        print(f"{len(three_comm)} papers are cited by all three communities")
        print("These shared ancestors show the communities have common intellectual roots")
        print("but have diverged in their citation practices — reading the same foundations")
        print("without reading each other's extensions of those foundations.")

    # Save
    output = DATA / "experiment_reference_overlap.json"
    results = {
        "community_ref_counts": {c: len(refs) for c, refs in community_refs.items()},
        "pairwise_jaccard": {},
        "shared_ancestors_all_3": len(three_comm),
        "shared_ancestors_2": len(two_comm),
        "top_shared_ancestors": [
            {"title": ref_titles.get(rid, "Unknown"), "communities": sorted(comms_set),
             "cite_counts": dict(bridge_cite_counts[rid])}
            for rid, comms_set in sorted_bridges[:30]
        ],
    }
    for i, c1 in enumerate(comms):
        for c2 in comms[i+1:]:
            shared = community_refs[c1] & community_refs[c2]
            union = community_refs[c1] | community_refs[c2]
            results["pairwise_jaccard"][f"{c1}__{c2}"] = round(len(shared) / len(union), 4) if union else 0
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()

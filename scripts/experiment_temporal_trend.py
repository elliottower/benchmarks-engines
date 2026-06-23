"""
Experiment: Temporal trend of cross-community citation density.

Uses Semantic Scholar API to get publication years for all papers,
then computes cross-citation density by year to see if fragmentation
is increasing, stable, or decreasing over time.
"""

import csv
import json
import os
import time
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"

S2_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
S2_BASE = "https://api.semanticscholar.org/graph/v1"
HEADERS = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
RATE_DELAY = 0.3 if S2_API_KEY else 1.5


def load_corpus():
    papers = {}
    with open(DATA / "mi_citation_corpus" / "mi_corpus_papers.csv") as f:
        for row in csv.DictReader(f):
            year = row["year"]
            try:
                year = int(float(year)) if year else None
            except ValueError:
                year = None
            papers[row["s2_paper_id"]] = {
                "title": row["title"],
                "year": year,
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


def main():
    print("EXPERIMENT: TEMPORAL TREND OF CITATION FRAGMENTATION")
    print("=" * 60)

    papers, edges = load_corpus()

    # Get source paper years
    source_years = {}
    for e in edges:
        src = e["source"]
        if src in papers and papers[src]["year"]:
            source_years[src] = papers[src]["year"]

    # Group edges by source paper year
    edges_by_year = defaultdict(list)
    for e in edges:
        src = e["source"]
        if src in source_years:
            edges_by_year[source_years[src]].append(e)

    years = sorted(y for y in edges_by_year.keys() if y >= 2019)

    print(f"\nPapers with years: {len(source_years)}/{len(papers)}")
    print(f"Years with edges: {min(years)}-{max(years)}")

    # Papers by year and community
    print(f"\n{'Year':>6} {'Papers':>8} {'Edges':>8} {'Cross%':>8} {'Cross-d':>10} {'Within-d':>10} {'Ratio':>8}")
    print("-" * 70)

    yearly_data = []

    for year in years:
        year_edges = edges_by_year[year]
        year_papers = {e["source"] for e in year_edges} | {e["target"] for e in year_edges}
        year_papers = {p for p in year_papers if p in papers}

        cross = sum(1 for e in year_edges if e["source_community"] != e["target_community"])
        within = sum(1 for e in year_edges if e["source_community"] == e["target_community"])
        total = cross + within

        # Compute densities
        comm_sizes = defaultdict(int)
        for pid in year_papers:
            comm_sizes[papers[pid]["community"]] += 1

        cross_possible = 0
        within_possible = 0
        comms = sorted(comm_sizes.keys())
        for i, c1 in enumerate(comms):
            for j, c2 in enumerate(comms):
                n1 = comm_sizes[c1]
                n2 = comm_sizes[c2]
                if c1 == c2:
                    within_possible += n1 * (n2 - 1)
                else:
                    cross_possible += n1 * n2

        cross_density = cross / cross_possible if cross_possible > 0 else 0
        within_density = within / within_possible if within_possible > 0 else 0
        ratio = within_density / cross_density if cross_density > 0 else float("inf")
        cross_pct = cross / total * 100 if total > 0 else 0

        print(f"{year:>6} {len(year_papers):>8} {total:>8} {cross_pct:>7.1f}% {cross_density*100:>9.4f}% {within_density*100:>9.4f}% {ratio:>7.1f}x")

        yearly_data.append({
            "year": year,
            "n_papers": len(year_papers),
            "n_edges": total,
            "cross_edges": cross,
            "within_edges": within,
            "cross_pct": round(cross_pct, 1),
            "cross_density": round(cross_density * 100, 4),
            "within_density": round(within_density * 100, 4),
            "ratio": round(ratio, 2),
        })

    # Community growth over time
    print(f"\n\nCOMMUNITY SIZE BY YEAR (cumulative papers)")
    print(f"{'Year':>6}", end="")
    all_comms = sorted(set(p["community"] for p in papers.values()))
    for c in all_comms:
        print(f" {c:>12}", end="")
    print()
    print("-" * (6 + 13 * len(all_comms)))

    for year in years:
        print(f"{year:>6}", end="")
        for c in all_comms:
            n = sum(1 for p in papers.values() if p["community"] == c and p["year"] and p["year"] <= year)
            print(f" {n:>12}", end="")
        print()

    # Pairwise cross-density trends
    print(f"\n\nPAIRWISE CROSS-DENSITY BY YEAR")
    pairs = [("sae", "causal"), ("sae", "circuits"), ("causal", "circuits")]
    print(f"{'Year':>6}", end="")
    for c1, c2 in pairs:
        print(f" {c1[:3]}>{c2[:3]:>8}", end="")
        print(f" {c2[:3]}>{c1[:3]:>8}", end="")
    print()
    print("-" * 70)

    for year in years:
        year_edges = edges_by_year[year]
        print(f"{year:>6}", end="")
        for c1, c2 in pairs:
            fwd = sum(1 for e in year_edges if e["source_community"] == c1 and e["target_community"] == c2)
            rev = sum(1 for e in year_edges if e["source_community"] == c2 and e["target_community"] == c1)
            print(f" {fwd:>8} {rev:>8}", end="")
        print()

    # Save results
    output = DATA / "experiment_temporal_trend.json"
    with open(output, "w") as f:
        json.dump({"yearly_data": yearly_data}, f, indent=2)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()

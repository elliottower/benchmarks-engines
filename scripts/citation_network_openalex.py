"""
Citation network analysis for MI proof-culture fragmentation using OpenAlex.
OpenAlex has much better rate limits than Semantic Scholar for this use case.
"""
import requests
import json
import time
import sys
from collections import defaultdict, Counter
from itertools import combinations

OPENALEX_BASE = "https://api.openalex.org"
MAILTO = "elliot@elliottower.ai"

SEED_PAPERS = {
    "SAE": [
        ("Cunningham et al 2023", "Sparse Autoencoders Find Highly Interpretable Features in Language Models"),
        ("Bricken et al 2023", "Towards Monosemanticity Decomposing Language Models With Dictionary Learning"),
        ("Templeton et al 2024", "Scaling Monosemanticity Extracting Interpretable Features from Claude 3 Sonnet"),
        ("Paulo et al 2024", "Automatically Interpreting Millions of Features in Large Language Models"),
        ("Paulo Belrose 2025", "Sparse Autoencoders Trained on the Same Data Learn Different Features"),
        ("Karvonen et al 2025", "SAEBench Comprehensive Benchmark Sparse Autoencoders"),
        ("Gao et al 2024", "Scaling and Evaluating Sparse Autoencoders"),
        ("Lieberum et al 2024", "Gemma Scope Open Sparse Autoencoders"),
        ("Engels et al 2024", "Decomposing the Dark Matter of Sparse Autoencoders"),
        ("Chanin et al 2024", "Absorption Studying Feature Splitting Absorption Sparse Autoencoders"),
        ("Rajamanoharan et al 2024", "Improving Dictionary Learning with Gated Sparse Autoencoders"),
        ("Marks et al 2024", "Sparse Feature Circuits Discovering Editing Interpretable Causal Graphs"),
    ],
    "Causal": [
        ("Geiger et al 2021", "Causal Abstractions of Neural Networks"),
        ("Geiger et al 2022", "Inducing Causal Structure for Interpretable Neural Networks"),
        ("Geiger et al 2023 DAS", "Finding Alignments Between Interpretable Causal Variables Distributed Neural Representations"),
        ("Wu et al 2024", "Interpretability at Scale Identifying Causal Mechanisms in Alpaca"),
        ("Huang et al 2024", "RAVEL Evaluating Interpretability Methods on Disentangling Language Model Representations"),
        ("Goldowsky-Dill et al 2023", "Localizing Model Behavior with Path Patching"),
        ("Jenner et al 2024", "The Causal Abstraction Activation Patching Cycle"),
    ],
    "Circuits": [
        ("Olah et al 2020", "Zoom In An Introduction to Circuits"),
        ("Elhage et al 2021", "Mathematical Framework for Transformer Circuits"),
        ("Olsson et al 2022", "In-context Learning and Induction Heads"),
        ("Wang et al 2022", "Interpretability in the Wild Circuit for Indirect Object Identification GPT-2"),
        ("Elhage et al 2022", "Toy Models of Superposition"),
        ("Conmy et al 2023", "Towards Automated Circuit Discovery for Mechanistic Interpretability"),
        ("Nanda et al 2023", "Progress measures for grokking via mechanistic interpretability"),
        ("Hanna et al 2023", "How does GPT-2 compute greater-than"),
        ("Gould et al 2024", "Successor Heads Recurring Interpretable Attention Heads"),
    ],
}


def search_openalex(title):
    """Search OpenAlex for a paper by title. Returns work dict or None."""
    url = f"{OPENALEX_BASE}/works"
    params = {
        "search": title,
        "mailto": MAILTO,
        "per_page": 3,
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            if results:
                return results[0]
    except Exception as e:
        print(f"  Error: {e}")
    return None


def get_references(work_id):
    """Get referenced works for a given OpenAlex work ID."""
    url = f"{OPENALEX_BASE}/works/{work_id}"
    params = {"mailto": MAILTO}
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("referenced_works", [])
    except Exception as e:
        print(f"  Error getting refs: {e}")
    return []


def main():
    print("=" * 60)
    print("MI CITATION NETWORK (OpenAlex)")
    print("=" * 60)

    # Phase 1: Resolve seed papers
    print("\n[Phase 1] Resolving seed papers...")
    papers = {}  # openalex_id -> {label, community, title, year, venue}
    community_ids = defaultdict(set)

    for community, seeds in SEED_PAPERS.items():
        print(f"\n  --- {community} ({len(seeds)} seeds) ---")
        for label, title in seeds:
            time.sleep(0.15)
            result = search_openalex(title)
            if result:
                oa_id = result["id"].split("/")[-1]
                papers[oa_id] = {
                    "label": label,
                    "community": community,
                    "title": result.get("title", title)[:80],
                    "year": result.get("publication_year"),
                    "venue": result.get("primary_location", {}).get("source", {}).get("display_name", "") if result.get("primary_location") and result["primary_location"].get("source") else "",
                    "citation_count": result.get("cited_by_count", 0),
                }
                community_ids[community].add(oa_id)
                print(f"    + {label} -> {oa_id} ({result.get('publication_year')}, {papers[oa_id]['citation_count']} cites)")
            else:
                print(f"    - {label} NOT FOUND")

    total = sum(len(v) for v in community_ids.values())
    print(f"\n  Resolved {total}/{sum(len(v) for v in SEED_PAPERS.values())} papers")

    # Phase 2: Get reference lists
    print("\n[Phase 2] Fetching reference lists...")
    ref_graph = {}  # oa_id -> set of referenced oa_ids
    all_seed_ids = set(papers.keys())

    for oa_id, info in papers.items():
        time.sleep(0.15)
        refs = get_references(oa_id)
        ref_ids = set()
        for ref_url in refs:
            ref_oa_id = ref_url.split("/")[-1] if isinstance(ref_url, str) else ""
            if ref_oa_id in all_seed_ids:
                ref_ids.add(ref_oa_id)
        ref_graph[oa_id] = ref_ids
        n_total = len(refs)
        n_internal = len(ref_ids)
        if n_internal > 0:
            print(f"    {info['label']}: {n_total} refs total, {n_internal} to seed papers")

    # Phase 3: Build citation matrix
    print("\n[Phase 3] Citation matrix...")
    matrix = defaultdict(lambda: defaultdict(int))
    edge_list = []

    for citing_id, cited_ids in ref_graph.items():
        for cited_id in cited_ids:
            if cited_id == citing_id:
                continue
            from_comm = papers[citing_id]["community"]
            to_comm = papers[cited_id]["community"]
            matrix[from_comm][to_comm] += 1
            edge_list.append({
                "from": papers[citing_id]["label"],
                "from_community": from_comm,
                "to": papers[cited_id]["label"],
                "to_community": to_comm,
            })

    comms = ["SAE", "Causal", "Circuits"]
    print(f"\n  {'FROM / TO':<15} {'SAE':>8} {'Causal':>8} {'Circuits':>10} {'Total':>8}")
    print("  " + "-" * 52)
    for c in comms:
        row = [matrix[c][c2] for c2 in comms]
        print(f"  {c:<15} {row[0]:>8} {row[1]:>8} {row[2]:>10} {sum(row):>8}")
    print("  " + "-" * 52)
    col_totals = [sum(matrix[c][c2] for c in comms) for c2 in comms]
    print(f"  {'Total In':<15} {col_totals[0]:>8} {col_totals[1]:>8} {col_totals[2]:>10}")

    # Compute metrics
    total_edges = sum(matrix[c1][c2] for c1 in comms for c2 in comms)
    within = sum(matrix[c][c] for c in comms)
    between = total_edges - within

    print(f"\n  Total edges: {total_edges}")
    print(f"  Within-community: {within} ({100*within/max(total_edges,1):.1f}%)")
    print(f"  Between-community: {between} ({100*between/max(total_edges,1):.1f}%)")

    # Pairwise densities
    print("\n  Pairwise cross-community densities:")
    for c1, c2 in [("SAE", "Causal"), ("Causal", "SAE"), ("SAE", "Circuits"), ("Circuits", "SAE"), ("Causal", "Circuits"), ("Circuits", "Causal")]:
        n1, n2 = len(community_ids[c1]), len(community_ids[c2])
        possible = n1 * n2
        actual = matrix[c1][c2]
        print(f"    {c1} -> {c2}: {actual}/{possible} possible = {actual/max(possible,1):.3f}")

    # Bridge papers
    print("\n  Bridge papers (cited by 2+ communities):")
    cited_by = defaultdict(set)
    for citing_id, cited_ids in ref_graph.items():
        comm = papers[citing_id]["community"]
        for cited_id in cited_ids:
            if cited_id != citing_id:
                cited_by[cited_id].add(comm)

    for pid, citing_comms in sorted(cited_by.items(), key=lambda x: -len(x[1])):
        if len(citing_comms) >= 2:
            info = papers.get(pid, {})
            print(f"    {info.get('label', pid)} [{info.get('community', '?')}] <- {sorted(citing_comms)}")

    # Edge details
    print("\n  All cross-community edges:")
    for e in sorted(edge_list, key=lambda x: (x["from_community"], x["to_community"])):
        if e["from_community"] != e["to_community"]:
            print(f"    {e['from']} [{e['from_community']}] -> {e['to']} [{e['to_community']}]")

    # Co-citation analysis
    print("\n  Co-citation pairs (papers cited together):")
    cocite = defaultdict(int)
    for citing_id, cited_ids in ref_graph.items():
        seed_refs = cited_ids - {citing_id}
        for p1, p2 in combinations(seed_refs, 2):
            pair = tuple(sorted([p1, p2]))
            cocite[pair] += 1

    top_cocite = sorted(cocite.items(), key=lambda x: -x[1])[:15]
    for (p1, p2), count in top_cocite:
        l1 = papers.get(p1, {}).get("label", p1[:12])
        l2 = papers.get(p2, {}).get("label", p2[:12])
        c1 = papers.get(p1, {}).get("community", "?")
        c2 = papers.get(p2, {}).get("community", "?")
        cross = "CROSS" if c1 != c2 else "same"
        print(f"    {count}x: {l1}[{c1}] + {l2}[{c2}] ({cross})")

    # Save
    results = {
        "metadata": {
            "date": "2026-06-21",
            "source": "OpenAlex",
            "n_papers": total,
            "n_edges": total_edges,
        },
        "papers": {pid: info for pid, info in papers.items()},
        "citation_matrix": {c1: {c2: matrix[c1][c2] for c2 in comms} for c1 in comms},
        "metrics": {
            "total_edges": total_edges,
            "within": within,
            "between": between,
            "within_pct": round(100 * within / max(total_edges, 1), 1),
            "sae_causal_cross": matrix["SAE"]["Causal"] + matrix["Causal"]["SAE"],
            "sae_n": len(community_ids["SAE"]),
            "causal_n": len(community_ids["Causal"]),
            "circuits_n": len(community_ids["Circuits"]),
        },
        "edges": edge_list,
        "bridge_papers": [
            {"label": papers.get(pid, {}).get("label", ""), "community": papers.get(pid, {}).get("community", ""), "cited_by": list(comms_set)}
            for pid, comms_set in cited_by.items() if len(comms_set) >= 2
        ],
    }

    out = "/Users/elliottower/Downloads/mc_iayn/data/mi_citation_network_openalex.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {out}")


if __name__ == "__main__":
    main()

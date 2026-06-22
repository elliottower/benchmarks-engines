"""
Deep citation network analysis for MI proof-culture fragmentation.

Strategy:
1. Start with seed papers for each community (expanded from original 27)
2. Query Semantic Scholar for full reference lists
3. Build the citation graph among all MI papers found
4. Compute modularity, clustering, bridge papers
5. Do co-citation analysis (which papers are cited together by the same downstream papers)
6. Venue analysis (where does each community publish?)
"""

import requests
import json
import time
import sys
from collections import defaultdict, Counter
from itertools import combinations

# Semantic Scholar API
S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "title,year,venue,authors,references,citations,externalIds"
RATE_LIMIT_DELAY = 1.1  # seconds between requests

# Seed papers per community - expanded set
SEED_PAPERS = {
    "SAE": [
        # Core SAE methodology
        ("Cunningham et al 2023", "Sparse Autoencoders Find Highly Interpretable Features in Language Models"),
        ("Bricken et al 2023", "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"),
        ("Templeton et al 2024", "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet"),
        ("Paulo et al 2024", "Automatically Interpreting Millions of Features in Large Language Models"),
        ("Paulo & Belrose 2025", "Sparse Autoencoders Trained on the Same Data Learn Different Features"),
        ("Karvonen et al 2025", "SAEBench: A Comprehensive Benchmark for Sparse Autoencoders"),
        ("Gao et al 2024", "Scaling and Evaluating Sparse Autoencoders"),
        ("Lieberum et al 2024", "Gemma Scope: Open Sparse Autoencoders Everywhere All At Once"),
        ("Engels et al 2024", "Decomposing the Dark Matter of Sparse Autoencoders"),
        ("Chanin et al 2024", "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders"),
        ("Rajamanoharan et al 2024", "Improving Dictionary Learning with Gated Sparse Autoencoders"),
        ("Marks et al 2024", "Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models"),
    ],
    "Causal": [
        # Causal abstraction / DAS / interchange interventions
        ("Geiger et al 2021", "Causal Abstractions of Neural Networks"),
        ("Geiger et al 2022", "Inducing Causal Structure for Interpretable Neural Networks"),
        ("Geiger et al 2023", "Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations"),
        ("Geiger et al 2024", "Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations"),
        ("Wu et al 2024", "Interpretability at Scale: Identifying Causal Mechanisms in Alpaca"),
        ("Huang et al 2024", "RAVEL: Evaluating Interpretability Methods on Disentangling Language Model Representations"),
        ("Goldowsky-Dill et al 2023", "Localizing Model Behavior with Path Patching"),
        ("Sutter et al 2025", "The Non-Linear Representation Dilemma"),
        ("Jenner et al 2024", "The Causal Abstraction/Activation Patching Cycle"),
        ("Davies et al 2023", "Unifying Causal Abstraction with Game-Theoretic Accountability"),
    ],
    "Circuits": [
        # Circuit discovery / activation patching / subgraph identification
        ("Olah et al 2020", "Zoom In: An Introduction to Circuits"),
        ("Elhage et al 2021", "A Mathematical Framework for Transformer Circuits"),
        ("Olsson et al 2022", "In-context Learning and Induction Heads"),
        ("Wang et al 2022", "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small"),
        ("Elhage et al 2022", "Toy Models of Superposition"),
        ("Conmy et al 2023", "Towards Automated Circuit Discovery for Mechanistic Interpretability"),
        ("Nanda et al 2023a", "Progress measures for grokking via mechanistic interpretability"),
        ("Nanda et al 2023b", "Attribution Patching: Activation Patching At Industrial Scale"),
        ("Hanna et al 2023", "How does GPT-2 compute greater-than?"),
        ("Gould et al 2024", "Successor Heads: Recurring, Interpretable Attention Heads In The Wild"),
    ],
}


def search_paper(title):
    """Search Semantic Scholar for a paper by title."""
    url = f"{S2_BASE}/paper/search"
    params = {"query": title, "fields": S2_FIELDS, "limit": 3}
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            print(f"  Rate limited, waiting 10s...")
            time.sleep(10)
            resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("data"):
                return data["data"][0]
    except Exception as e:
        print(f"  Error searching: {e}")
    return None


def get_paper_details(paper_id):
    """Get full paper details including references."""
    url = f"{S2_BASE}/paper/{paper_id}"
    params = {"fields": "title,year,venue,authors,references.paperId,references.title,citations.paperId,citations.title,externalIds,citationCount"}
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            print(f"  Rate limited, waiting 10s...")
            time.sleep(10)
            resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"  Error fetching details: {e}")
    return None


def main():
    print("=" * 60)
    print("MI CITATION NETWORK: DEEP ANALYSIS")
    print("=" * 60)

    # Phase 1: Resolve all seed papers to Semantic Scholar IDs
    print("\n[Phase 1] Resolving seed papers...")
    papers_db = {}  # paper_id -> {community, title, year, venue, refs, cites, ...}
    community_ids = defaultdict(set)  # community -> set of paper_ids

    for community, papers in SEED_PAPERS.items():
        print(f"\n  --- {community} ({len(papers)} seeds) ---")
        for label, title in papers:
            time.sleep(RATE_LIMIT_DELAY)
            result = search_paper(title)
            if result and result.get("paperId"):
                pid = result["paperId"]
                papers_db[pid] = {
                    "label": label,
                    "title": result.get("title", title),
                    "year": result.get("year"),
                    "venue": result.get("venue", ""),
                    "community": community,
                }
                community_ids[community].add(pid)
                print(f"    ✓ {label} -> {pid[:12]}... ({result.get('year')})")
            else:
                print(f"    ✗ {label} -- NOT FOUND")

    total_found = sum(len(v) for v in community_ids.values())
    print(f"\n  Resolved {total_found}/{sum(len(v) for v in SEED_PAPERS.values())} seed papers")

    # Phase 2: Get reference lists for all found papers
    print("\n[Phase 2] Fetching reference lists...")
    ref_graph = defaultdict(set)  # citing_id -> set of cited_ids
    all_referenced = set()

    for pid, info in list(papers_db.items()):
        time.sleep(RATE_LIMIT_DELAY)
        details = get_paper_details(pid)
        if details and details.get("references"):
            refs = [r["paperId"] for r in details["references"] if r.get("paperId")]
            ref_graph[pid] = set(refs)
            all_referenced.update(refs)
            info["citation_count"] = details.get("citationCount", 0)
            print(f"    {info['label']}: {len(refs)} references, {info['citation_count']} citations")
        else:
            print(f"    {info['label']}: no references found")

    # Phase 3: Build citation matrix among seed papers
    print("\n[Phase 3] Building intra-network citation matrix...")
    all_seed_ids = set(papers_db.keys())

    # For each paper, which other seed papers does it cite?
    citation_edges = []  # (from_id, to_id)
    for citing_id, refs in ref_graph.items():
        for cited_id in refs:
            if cited_id in all_seed_ids and cited_id != citing_id:
                citation_edges.append((citing_id, cited_id))

    print(f"  Found {len(citation_edges)} citation edges among seed papers")

    # Build the matrix
    matrix = defaultdict(lambda: defaultdict(int))
    edge_details = defaultdict(list)
    for citing_id, cited_id in citation_edges:
        from_comm = papers_db[citing_id]["community"]
        to_comm = papers_db[cited_id]["community"]
        matrix[from_comm][to_comm] += 1
        edge_details[f"{from_comm}->{to_comm}"].append(
            f"  {papers_db[citing_id]['label']} -> {papers_db[cited_id]['label']}"
        )

    print("\n  Citation Flow Matrix:")
    header_label = "FROM \\ TO"
    print(f"  {header_label:<12} {'SAE':>8} {'Causal':>8} {'Circuits':>10} {'Total':>8}")
    print("  " + "-" * 50)
    for comm in ["SAE", "Causal", "Circuits"]:
        row_total = sum(matrix[comm][c] for c in ["SAE", "Causal", "Circuits"])
        print(f"  {comm:<12} {matrix[comm]['SAE']:>8} {matrix[comm]['Causal']:>8} {matrix[comm]['Circuits']:>10} {row_total:>8}")
    print("  " + "-" * 50)
    col_totals = [sum(matrix[c][to] for c in ["SAE", "Causal", "Circuits"]) for to in ["SAE", "Causal", "Circuits"]]
    print(f"  {'Total In':<12} {col_totals[0]:>8} {col_totals[1]:>8} {col_totals[2]:>10}")

    # Phase 4: Network metrics
    print("\n[Phase 4] Computing network metrics...")
    import networkx as nx

    G = nx.DiGraph()
    for pid, info in papers_db.items():
        G.add_node(pid, community=info["community"], label=info["label"])

    for citing_id, cited_id in citation_edges:
        G.add_edge(citing_id, cited_id)

    # Within vs between community edges
    within = sum(1 for u, v in G.edges() if papers_db[u]["community"] == papers_db[v]["community"])
    between = sum(1 for u, v in G.edges() if papers_db[u]["community"] != papers_db[v]["community"])
    total_edges = within + between

    print(f"  Total edges: {total_edges}")
    print(f"  Within-community: {within} ({100*within/max(total_edges,1):.1f}%)")
    print(f"  Between-community: {between} ({100*between/max(total_edges,1):.1f}%)")

    # Modularity (using community assignments)
    # Convert to undirected for modularity
    G_undir = G.to_undirected()
    communities_partition = []
    for comm in ["SAE", "Causal", "Circuits"]:
        communities_partition.append(community_ids[comm])

    if len(G_undir.edges()) > 0:
        try:
            mod = nx.algorithms.community.modularity(G_undir, communities_partition)
            print(f"  Modularity (Q): {mod:.3f}")
            print(f"    (Q > 0.3 = strong community structure; Q > 0.5 = very strong)")
        except:
            print("  Modularity: could not compute")

    # Density within vs between
    for comm in ["SAE", "Causal", "Circuits"]:
        n = len(community_ids[comm])
        possible = n * (n - 1)
        actual_within = sum(1 for u, v in G.edges() if papers_db.get(u, {}).get("community") == comm and papers_db.get(v, {}).get("community") == comm)
        density = actual_within / max(possible, 1)
        print(f"  {comm} within-density: {actual_within}/{possible} = {density:.3f}")

    # Cross-community densities
    print("\n  Cross-community densities:")
    for c1, c2 in [("SAE", "Causal"), ("SAE", "Circuits"), ("Causal", "Circuits")]:
        n1 = len(community_ids[c1])
        n2 = len(community_ids[c2])
        possible = n1 * n2  # directed: c1 -> c2
        actual = matrix[c1][c2]
        density_fwd = actual / max(possible, 1)
        actual_rev = matrix[c2][c1]
        density_rev = actual_rev / max(possible, 1)
        print(f"  {c1} -> {c2}: {actual}/{possible} = {density_fwd:.4f}")
        print(f"  {c2} -> {c1}: {actual_rev}/{possible} = {density_rev:.4f}")

    # Phase 5: Bridge papers (papers cited by multiple communities)
    print("\n[Phase 5] Bridge papers (cited by 2+ communities)...")
    cited_by_comms = defaultdict(set)  # paper_id -> set of communities that cite it
    for citing_id, cited_id in citation_edges:
        comm = papers_db[citing_id]["community"]
        cited_by_comms[cited_id].add(comm)

    bridge_papers = [(pid, comms) for pid, comms in cited_by_comms.items() if len(comms) >= 2]
    bridge_papers.sort(key=lambda x: -len(x[1]))

    for pid, comms in bridge_papers:
        info = papers_db.get(pid, {})
        label = info.get("label", pid[:20])
        print(f"  {label} [{info.get('community', '?')}] cited by: {', '.join(sorted(comms))}")

    # Phase 6: Co-citation analysis
    # Which pairs of seed papers are most frequently co-cited by the same downstream papers?
    print("\n[Phase 6] Co-citation analysis (which papers appear together in reference lists)...")
    cocitation = defaultdict(int)
    for citing_id, refs in ref_graph.items():
        seed_refs = refs & all_seed_ids
        for p1, p2 in combinations(seed_refs, 2):
            pair = tuple(sorted([p1, p2]))
            cocitation[pair] += 1

    # Top co-cited pairs
    top_pairs = sorted(cocitation.items(), key=lambda x: -x[1])[:15]
    print(f"  Top co-cited pairs (within our seed set):")
    for (p1, p2), count in top_pairs:
        l1 = papers_db.get(p1, {}).get("label", p1[:15])
        l2 = papers_db.get(p2, {}).get("label", p2[:15])
        c1 = papers_db.get(p1, {}).get("community", "?")
        c2 = papers_db.get(p2, {}).get("community", "?")
        cross = "CROSS" if c1 != c2 else "within"
        print(f"    {count}x: {l1} [{c1}] + {l2} [{c2}] -- {cross}")

    # Cross-community co-citations
    cross_cocite = [(pair, count) for pair, count in cocitation.items()
                    if papers_db.get(pair[0], {}).get("community") != papers_db.get(pair[1], {}).get("community")]
    cross_cocite.sort(key=lambda x: -x[1])
    print(f"\n  Cross-community co-citations: {len(cross_cocite)} pairs")
    print(f"  Total cross co-citation count: {sum(c for _, c in cross_cocite)}")
    within_cocite = [(pair, count) for pair, count in cocitation.items()
                     if papers_db.get(pair[0], {}).get("community") == papers_db.get(pair[1], {}).get("community")]
    print(f"  Within-community co-citations: {len(within_cocite)} pairs")
    print(f"  Total within co-citation count: {sum(c for _, c in within_cocite)}")

    # Phase 7: Venue analysis
    print("\n[Phase 7] Venue/publication patterns...")
    venue_by_comm = defaultdict(list)
    for pid, info in papers_db.items():
        venue = info.get("venue", "unknown")
        if venue:
            venue_by_comm[info["community"]].append(venue)

    for comm in ["SAE", "Causal", "Circuits"]:
        venues = venue_by_comm[comm]
        venue_counts = Counter(venues)
        print(f"\n  {comm} venues:")
        for v, c in venue_counts.most_common(5):
            print(f"    {c}x {v}")

    # Phase 8: Save results
    print("\n[Phase 8] Saving results...")

    results = {
        "metadata": {
            "analysis_date": "2026-06-21",
            "description": "Deep citation network analysis of MI subcommunities",
            "n_seed_papers": total_found,
            "n_citation_edges": total_edges,
            "methodology": "Semantic Scholar API reference list retrieval for expanded seed set. Network metrics computed with NetworkX.",
        },
        "citation_matrix": {c1: {c2: matrix[c1][c2] for c2 in ["SAE", "Causal", "Circuits"]} for c1 in ["SAE", "Causal", "Circuits"]},
        "network_metrics": {
            "total_edges": total_edges,
            "within_community": within,
            "between_community": between,
            "within_pct": round(100 * within / max(total_edges, 1), 1),
        },
        "bridge_papers": [
            {"label": papers_db.get(pid, {}).get("label", ""), "community": papers_db.get(pid, {}).get("community", ""), "cited_by_communities": list(comms)}
            for pid, comms in bridge_papers
        ],
        "cross_community_cocitations": len(cross_cocite),
        "within_community_cocitations": len(within_cocite),
        "edge_details": {k: v for k, v in edge_details.items()},
        "papers_resolved": {pid: {"label": info["label"], "community": info["community"], "year": info.get("year"), "venue": info.get("venue", "")} for pid, info in papers_db.items()},
    }

    output_path = "/Users/elliottower/Downloads/mc_iayn/data/mi_citation_network_deep.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {output_path}")

    # Summary for the paper
    print("\n" + "=" * 60)
    print("SUMMARY FOR PAPER")
    print("=" * 60)
    print(f"""
Across {total_found} canonical MI papers (SAE: {len(community_ids['SAE'])}, Causal: {len(community_ids['Causal'])}, Circuits: {len(community_ids['Circuits'])}):

Citation flow:
  SAE -> Causal:     {matrix['SAE']['Causal']} edges
  Causal -> SAE:     {matrix['Causal']['SAE']} edges
  SAE -> Circuits:   {matrix['SAE']['Circuits']} edges
  Causal -> Circuits: {matrix['Causal']['Circuits']} edges
  Circuits -> SAE:   {matrix['Circuits']['SAE']} edges
  Circuits -> Causal: {matrix['Circuits']['Causal']} edges

Within vs between: {within} within ({100*within/max(total_edges,1):.0f}%) vs {between} between ({100*between/max(total_edges,1):.0f}%)

Bridge papers cited by multiple communities: {len(bridge_papers)}

KEY FINDING: SAE <-> Causal cross-citation = {matrix['SAE']['Causal'] + matrix['Causal']['SAE']} total edges
(out of {len(community_ids['SAE'])} * {len(community_ids['Causal'])} = {len(community_ids['SAE']) * len(community_ids['Causal'])} possible directed pairs)
Cross-density = {(matrix['SAE']['Causal'] + matrix['Causal']['SAE']) / max(2 * len(community_ids['SAE']) * len(community_ids['Causal']), 1):.4f}
""")


if __name__ == "__main__":
    main()

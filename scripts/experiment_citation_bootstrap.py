"""
Experiment 2: Bootstrap/permutation test on citation cross-density
for the MI proof-cultures claim.

Tests whether the observed low cross-community citation density (1.4%
for the canonical 27-paper set, and the 688-paper corpus) is
distinguishable from chance by shuffling community labels.

Also tests sensitivity to corpus boundary by computing density at
multiple granularities.
"""

import csv
import json
import random
import statistics
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"

SEED = 42
N_PERMUTATIONS = 10_000
random.seed(SEED)


def load_canonical_network():
    path = DATA / "mi_citation_network_final.json"
    with open(path) as f:
        data = json.load(f)
    return data


def load_corpus():
    papers_path = DATA / "mi_citation_corpus" / "mi_corpus_papers.csv"
    edges_path = DATA / "mi_citation_corpus" / "mi_corpus_edges.csv"

    papers = {}
    with open(papers_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            papers[row["s2_paper_id"]] = {
                "title": row["title"],
                "year": row["year"],
                "community": row["community"],
                "citation_count": int(float(row["citation_count"])) if row["citation_count"] else 0,
                "is_seed": row["is_seed"] == "True",
            }

    edges = []
    with open(edges_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            edges.append({
                "source": row["source_s2id"],
                "target": row["target_s2id"],
                "source_community": row["source_community"],
                "target_community": row["target_community"],
            })

    return papers, edges


def compute_density_matrix(papers: dict, edges):
    communities = sorted(set(p["community"] for p in papers.values()))
    comm_papers = {c: [pid for pid, p in papers.items() if p["community"] == c]
                   for c in communities}

    matrix = {}
    for c_from in communities:
        for c_to in communities:
            n_from = len(comm_papers[c_from])
            n_to = len(comm_papers[c_to])
            if c_from == c_to:
                possible = n_from * (n_to - 1)
            else:
                possible = n_from * n_to
            actual = sum(1 for e in edges
                        if e["source_community"] == c_from
                        and e["target_community"] == c_to)
            density = actual / possible if possible > 0 else 0
            matrix[(c_from, c_to)] = {
                "n_edges": actual,
                "possible": possible,
                "density": density,
            }

    return matrix, communities


def compute_cross_density(papers: dict, edges):
    cross_edges = sum(1 for e in edges if e["source_community"] != e["target_community"])
    within_edges = sum(1 for e in edges if e["source_community"] == e["target_community"])

    communities = set(p["community"] for p in papers.values())
    comm_sizes = defaultdict(int)
    for p in papers.values():
        comm_sizes[p["community"]] += 1

    total_cross_possible = 0
    total_within_possible = 0
    comms = sorted(communities)
    for i, c1 in enumerate(comms):
        for j, c2 in enumerate(comms):
            n1 = comm_sizes[c1]
            n2 = comm_sizes[c2]
            if c1 == c2:
                total_within_possible += n1 * (n2 - 1)
            else:
                total_cross_possible += n1 * n2

    cross_density = cross_edges / total_cross_possible if total_cross_possible > 0 else 0
    within_density = within_edges / total_within_possible if total_within_possible > 0 else 0

    return {
        "cross_edges": cross_edges,
        "within_edges": within_edges,
        "total_edges": cross_edges + within_edges,
        "cross_possible": total_cross_possible,
        "within_possible": total_within_possible,
        "cross_density": cross_density,
        "within_density": within_density,
        "ratio": within_density / cross_density if cross_density > 0 else float("inf"),
    }


def permutation_test(papers: dict, edges, n_perms: int = N_PERMUTATIONS):
    observed = compute_cross_density(papers, edges)
    observed_cross = observed["cross_density"]
    observed_ratio = observed["ratio"]

    paper_ids = list(papers.keys())
    original_communities = {pid: papers[pid]["community"] for pid in paper_ids}
    community_list = [original_communities[pid] for pid in paper_ids]

    count_lower_cross = 0
    count_higher_ratio = 0
    perm_cross_densities = []
    perm_ratios = []

    for _ in range(n_perms):
        shuffled = list(community_list)
        random.shuffle(shuffled)
        shuffled_map = {pid: shuffled[i] for i, pid in enumerate(paper_ids)}

        shuffled_edges = []
        for e in edges:
            src = e["source"]
            tgt = e["target"]
            if src in shuffled_map and tgt in shuffled_map:
                shuffled_edges.append({
                    "source": src,
                    "target": tgt,
                    "source_community": shuffled_map[src],
                    "target_community": shuffled_map[tgt],
                })

        perm_result = compute_cross_density(
            {pid: {**papers[pid], "community": shuffled_map[pid]} for pid in paper_ids if pid in shuffled_map},
            shuffled_edges,
        )
        perm_cross_densities.append(perm_result["cross_density"])
        perm_ratios.append(perm_result["ratio"])

        if perm_result["cross_density"] <= observed_cross:
            count_lower_cross += 1
        if perm_result["ratio"] >= observed_ratio:
            count_higher_ratio += 1

    p_cross = count_lower_cross / n_perms
    p_ratio = count_higher_ratio / n_perms

    return {
        "observed_cross_density": observed_cross,
        "observed_within_density": observed["within_density"],
        "observed_ratio": observed_ratio,
        "p_value_cross": p_cross,
        "p_value_ratio": p_ratio,
        "perm_cross_mean": statistics.mean(perm_cross_densities),
        "perm_cross_std": statistics.stdev(perm_cross_densities),
        "perm_cross_ci_lo": sorted(perm_cross_densities)[int(0.025 * n_perms)],
        "perm_cross_ci_hi": sorted(perm_cross_densities)[int(0.975 * n_perms)],
    }


def bootstrap_cross_density(papers: dict, edges, n_boot: int = N_PERMUTATIONS):
    observed = compute_cross_density(papers, edges)

    paper_ids = list(papers.keys())
    pid_set = set(paper_ids)

    boot_cross = []
    boot_within = []
    boot_ratios = []

    for _ in range(n_boot):
        sample_pids = set(random.choices(paper_ids, k=len(paper_ids)))
        sample_papers = {pid: papers[pid] for pid in sample_pids}
        sample_edges = [e for e in edges if e["source"] in sample_pids and e["target"] in sample_pids]

        if not sample_edges:
            continue

        result = compute_cross_density(sample_papers, sample_edges)
        boot_cross.append(result["cross_density"])
        boot_within.append(result["within_density"])
        if result["cross_density"] > 0:
            boot_ratios.append(result["ratio"])

    boot_cross.sort()
    boot_within.sort()
    boot_ratios.sort()

    n = len(boot_cross)
    nr = len(boot_ratios)

    return {
        "cross_density_ci": (boot_cross[int(0.025*n)], boot_cross[int(0.975*n)]) if n > 40 else None,
        "within_density_ci": (boot_within[int(0.025*n)], boot_within[int(0.975*n)]) if n > 40 else None,
        "ratio_ci": (boot_ratios[int(0.025*nr)], boot_ratios[int(0.975*nr)]) if nr > 40 else None,
        "n_valid": n,
    }


def sensitivity_analysis(papers: dict, edges):
    """Test stability across corpus boundaries: seed-only, seed+1hop, full."""
    print(f"\n{'='*70}")
    print("SENSITIVITY: CORPUS BOUNDARY")
    print(f"{'='*70}")

    # Full corpus
    full = compute_cross_density(papers, edges)
    print(f"\n  Full corpus ({len(papers)} papers, {len(edges)} edges):")
    print(f"    Cross-density: {full['cross_density']*100:.4f}%")
    print(f"    Within-density: {full['within_density']*100:.4f}%")
    print(f"    Ratio (within/cross): {full['ratio']:.1f}x")

    # Seed papers only
    seed_papers = {pid: p for pid, p in papers.items() if p["is_seed"]}
    seed_pids = set(seed_papers.keys())
    seed_edges = [e for e in edges
                  if e["source"] in seed_pids and e["target"] in seed_pids]
    if seed_papers and seed_edges:
        seed_result = compute_cross_density(seed_papers, seed_edges)
        print(f"\n  Seed papers only ({len(seed_papers)} papers, {len(seed_edges)} edges):")
        print(f"    Cross-density: {seed_result['cross_density']*100:.4f}%")
        print(f"    Within-density: {seed_result['within_density']*100:.4f}%")
        print(f"    Ratio (within/cross): {seed_result['ratio']:.1f}x")
    else:
        print(f"\n  Seed papers only: {len(seed_papers)} papers, {len(seed_edges)} edges (too few)")

    # High-citation papers only (top quartile)
    citation_threshold = sorted(
        (p["citation_count"] for p in papers.values()), reverse=True
    )[len(papers) // 4]
    hi_papers = {pid: p for pid, p in papers.items()
                 if p["citation_count"] >= citation_threshold}
    hi_pids = set(hi_papers.keys())
    hi_edges = [e for e in edges if e["source"] in hi_pids and e["target"] in hi_pids]
    if hi_papers and hi_edges:
        hi_result = compute_cross_density(hi_papers, hi_edges)
        print(f"\n  High-citation papers (top 25%, >= {citation_threshold} cites, "
              f"{len(hi_papers)} papers, {len(hi_edges)} edges):")
        print(f"    Cross-density: {hi_result['cross_density']*100:.4f}%")
        print(f"    Within-density: {hi_result['within_density']*100:.4f}%")
        print(f"    Ratio (within/cross): {hi_result['ratio']:.1f}x")


def pairwise_density_table(papers, edges, communities):
    print(f"\n{'='*70}")
    print("PAIRWISE CITATION DENSITY MATRIX")
    print(f"{'='*70}")

    matrix, _ = compute_density_matrix(papers, edges)

    # Header
    from_to = 'From \\ To'
    header = f"{from_to:<15}" + "".join(f"{c:>15}" for c in communities)
    print(f"\n  {header}")
    print("  " + "-" * len(header))

    for c_from in communities:
        row = f"{c_from:<15}"
        for c_to in communities:
            m = matrix[(c_from, c_to)]
            density_pct = m["density"] * 100
            row += f"  {density_pct:>8.4f}% ({m['n_edges']:>3})"
        print(f"  {row}")


def main():
    print("EXPERIMENT 2: CITATION CROSS-DENSITY ANALYSIS")
    print("=" * 70)

    # Load 688-paper corpus
    papers, edges = load_corpus()
    communities = sorted(set(p["community"] for p in papers.values()))

    print(f"\nCorpus: {len(papers)} papers, {len(edges)} edges")
    for c in communities:
        n = sum(1 for p in papers.values() if p["community"] == c)
        n_seed = sum(1 for p in papers.values() if p["community"] == c and p["is_seed"])
        print(f"  {c}: {n} papers ({n_seed} seed)")

    # Observed densities
    observed = compute_cross_density(papers, edges)
    print(f"\nObserved cross-community density: {observed['cross_density']*100:.4f}%")
    print(f"Observed within-community density: {observed['within_density']*100:.4f}%")
    print(f"Within/cross ratio: {observed['ratio']:.1f}x")
    print(f"Cross edges: {observed['cross_edges']}/{observed['total_edges']} "
          f"({observed['cross_edges']/observed['total_edges']*100:.1f}%)")

    # Pairwise density matrix
    pairwise_density_table(papers, edges, communities)

    # Permutation test
    print(f"\n{'='*70}")
    print(f"PERMUTATION TEST (n={N_PERMUTATIONS})")
    print(f"{'='*70}")
    print("Null hypothesis: community labels are independent of citation structure")
    print("Shuffling community labels and recomputing cross-density...\n")

    perm = permutation_test(papers, edges)
    print(f"  Observed cross-density: {perm['observed_cross_density']*100:.4f}%")
    print(f"  Null (shuffled) cross-density: {perm['perm_cross_mean']*100:.4f}% "
          f"(SD={perm['perm_cross_std']*100:.4f}%)")
    print(f"  Null 95% range: [{perm['perm_cross_ci_lo']*100:.4f}%, "
          f"{perm['perm_cross_ci_hi']*100:.4f}%]")
    print(f"  p-value (cross-density <= observed): {perm['p_value_cross']:.4f}")
    print(f"  p-value (within/cross ratio >= observed): {perm['p_value_ratio']:.4f}")

    if perm["p_value_cross"] < 0.05:
        print("\n  ** Cross-density is SIGNIFICANTLY LOWER than chance (p < 0.05)")
        print("  ** Community structure reflects real citation insularity")
    else:
        print("\n  Cross-density is NOT significantly different from chance")

    # Bootstrap CIs
    print(f"\n{'='*70}")
    print(f"BOOTSTRAP CONFIDENCE INTERVALS (n={N_PERMUTATIONS})")
    print(f"{'='*70}")

    boot = bootstrap_cross_density(papers, edges)
    if boot["cross_density_ci"]:
        print(f"  Cross-density 95% CI: [{boot['cross_density_ci'][0]*100:.4f}%, "
              f"{boot['cross_density_ci'][1]*100:.4f}%]")
    if boot["within_density_ci"]:
        print(f"  Within-density 95% CI: [{boot['within_density_ci'][0]*100:.4f}%, "
              f"{boot['within_density_ci'][1]*100:.4f}%]")
    if boot["ratio_ci"]:
        print(f"  Within/cross ratio 95% CI: [{boot['ratio_ci'][0]:.1f}x, "
              f"{boot['ratio_ci'][1]:.1f}x]")

    # Sensitivity analysis
    sensitivity_analysis(papers, edges)

    # Also check the canonical 27-paper network for comparison
    print(f"\n{'='*70}")
    print("CANONICAL 27-PAPER NETWORK (for comparison)")
    print(f"{'='*70}")
    canonical = load_canonical_network()
    print(f"  Papers: {len(canonical.get('papers', canonical.get('nodes', [])))}")
    print(f"  Edges: {len(canonical.get('edges', []))}")
    if "communities" in canonical:
        for comm, members in canonical["communities"].items():
            print(f"  {comm}: {len(members)} papers")


if __name__ == "__main__":
    main()

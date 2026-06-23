"""
Experiment 1: Proxy-control regression and matched-pairs analysis
for ICLR 2024/2023 demand-compliance data.

Tests whether compliance predicts acceptance after controlling for
observable quality proxies (mean review score, number of reviewers,
mean review length). Addresses the confound that "compliant papers
are just better" by matching on score and computing within-pair diffs.

Outputs: regression tables, matched-pairs results, bootstrap CIs.
"""

import csv
import json
import statistics
import random
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
MC_IAYN_DATA = Path.home() / "Downloads" / "mc_iayn" / "data"

SEED = 42
N_BOOTSTRAP = 10_000
random.seed(SEED)


def load_demand_compliance(year: int) -> dict:
    path = DATA / f"iclr{year}_demand_compliance.csv"
    if not path.exists():
        path = MC_IAYN_DATA / f"iclr{year}_demand_compliance.csv"
    papers = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            papers[row["paper_id"]] = {
                "title": row["title"],
                "has_demand": row["has_demand"] == "True",
                "demand_count": int(row["demand_count"]),
                "has_compliance": row["has_compliance"] == "True",
                "compliance_count": int(row["compliance_count"]),
                "decision": row["decision"],
                "accepted": row["acceptance"] == "True",
            }
    return papers


def load_scores(year: int) -> dict:
    path = MC_IAYN_DATA / f"iclr{year}_scores.json"
    if not path.exists():
        path = DATA / f"iclr{year}_scores.json"
    with open(path) as f:
        return json.load(f)


def load_reviews(year: int) -> dict:
    path = MC_IAYN_DATA / f"iclr{year}_reviews_raw.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    reviews_by_id = {}
    for entry in data:
        pid = entry["paper_id"]
        reviews_by_id[pid] = {
            "reviews": entry.get("reviews", []),
            "n_reviews": len(entry.get("reviews", [])),
            "mean_review_length": (
                statistics.mean(len(r) for r in entry["reviews"])
                if entry.get("reviews")
                else 0
            ),
        }
    return reviews_by_id


def merge_data(year: int) -> list:
    papers = load_demand_compliance(year)
    scores = load_scores(year)
    reviews = load_reviews(year)

    merged = []
    for pid, p in papers.items():
        if pid not in scores:
            continue
        s = scores[pid]
        r = reviews.get(pid, {})
        mean_score = statistics.mean(s)
        merged.append({
            "paper_id": pid,
            "year": year,
            "accepted": p["accepted"],
            "has_demand": p["has_demand"],
            "has_compliance": p["has_compliance"],
            "demand_count": p["demand_count"],
            "compliance_count": p["compliance_count"],
            "decision": p["decision"],
            "mean_score": mean_score,
            "score_std": statistics.stdev(s) if len(s) > 1 else 0,
            "n_reviewers": len(s),
            "mean_review_length": r.get("mean_review_length", 0),
        })
    return merged


def is_borderline(paper: dict) -> bool:
    return 4.5 <= paper["mean_score"] <= 6.5


def logistic_regression_manual(X, y, max_iter=200, lr=0.01):
    """Minimal logistic regression via gradient descent. No external deps."""
    import math
    n = len(y)
    p = len(X[0])
    beta = [0.0] * p

    for _ in range(max_iter):
        grad = [0.0] * p
        for i in range(n):
            z = sum(beta[j] * X[i][j] for j in range(p))
            z = max(-500, min(500, z))
            pred = 1 / (1 + math.exp(-z))
            err = pred - y[i]
            for j in range(p):
                grad[j] += err * X[i][j] / n
        for j in range(p):
            beta[j] -= lr * grad[j]

    log_lik = 0
    for i in range(n):
        z = sum(beta[j] * X[i][j] for j in range(p))
        z = max(-500, min(500, z))
        pred = 1 / (1 + math.exp(-z))
        pred = max(1e-10, min(1 - 1e-10, pred))
        log_lik += y[i] * math.log(pred) + (1 - y[i]) * math.log(1 - pred)

    return beta, log_lik


def standardize(values):
    mu = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 1
    if sd == 0:
        sd = 1
    return [(v - mu) / sd for v in values]


def run_regression(data, label: str):
    print(f"\n{'='*70}")
    print(f"LOGISTIC REGRESSION: {label}")
    print(f"{'='*70}")
    print(f"N = {len(data)}")
    print(f"  Accepted: {sum(d['accepted'] for d in data)}")
    print(f"  Rejected: {sum(not d['accepted'] for d in data)}")
    print(f"  Has compliance: {sum(d['has_compliance'] for d in data)}")

    y = [int(d["accepted"]) for d in data]

    mean_scores_raw = [d["mean_score"] for d in data]
    n_reviewers_raw = [float(d["n_reviewers"]) for d in data]
    review_len_raw = [d["mean_review_length"] for d in data]

    mean_scores = standardize(mean_scores_raw)
    n_reviewers = standardize(n_reviewers_raw)
    review_len = standardize(review_len_raw)
    compliance = [int(d["has_compliance"]) for d in data]

    # Model 1: acceptance ~ compliance (unadjusted)
    X1 = [[1, compliance[i]] for i in range(len(data))]
    beta1, ll1 = logistic_regression_manual(X1, y)
    print(f"\nModel 1: acceptance ~ compliance (unadjusted)")
    print(f"  Intercept: {beta1[0]:.4f}")
    print(f"  Compliance coef: {beta1[1]:.4f}")
    print(f"  Log-likelihood: {ll1:.2f}")

    # Model 2: acceptance ~ compliance + mean_score
    X2 = [[1, compliance[i], mean_scores[i]] for i in range(len(data))]
    beta2, ll2 = logistic_regression_manual(X2, y)
    print(f"\nModel 2: acceptance ~ compliance + mean_score")
    print(f"  Intercept: {beta2[0]:.4f}")
    print(f"  Compliance coef: {beta2[1]:.4f}")
    print(f"  Mean score coef: {beta2[2]:.4f}")
    print(f"  Log-likelihood: {ll2:.2f}")

    # Model 3: acceptance ~ compliance + mean_score + n_reviewers + review_length
    X3 = [[1, compliance[i], mean_scores[i], n_reviewers[i], review_len[i]]
          for i in range(len(data))]
    beta3, ll3 = logistic_regression_manual(X3, y, max_iter=500)
    print(f"\nModel 3: acceptance ~ compliance + mean_score + n_reviewers + review_length")
    print(f"  Intercept: {beta3[0]:.4f}")
    print(f"  Compliance coef: {beta3[1]:.4f}")
    print(f"  Mean score coef: {beta3[2]:.4f}")
    print(f"  N reviewers coef: {beta3[3]:.4f}")
    print(f"  Review length coef: {beta3[4]:.4f}")
    print(f"  Log-likelihood: {ll3:.2f}")

    return beta3[1]  # compliance coefficient from full model


def run_matched_pairs(data, label: str):
    print(f"\n{'='*70}")
    print(f"MATCHED-PAIRS ANALYSIS: {label}")
    print(f"{'='*70}")

    compliant = [d for d in data if d["has_compliance"]]
    non_compliant = [d for d in data if not d["has_compliance"]]

    print(f"  Compliant: {len(compliant)}")
    print(f"  Non-compliant: {len(non_compliant)}")

    # Match on rounded mean score (to nearest 0.5)
    def score_bin(s): return round(s * 2) / 2

    comp_by_bin = defaultdict(list)
    noncomp_by_bin = defaultdict(list)
    for d in compliant:
        comp_by_bin[score_bin(d["mean_score"])].append(d)
    for d in non_compliant:
        noncomp_by_bin[score_bin(d["mean_score"])].append(d)

    pairs = []
    for sbin in comp_by_bin:
        if sbin not in noncomp_by_bin:
            continue
        c_list = list(comp_by_bin[sbin])
        nc_list = list(noncomp_by_bin[sbin])
        random.shuffle(c_list)
        random.shuffle(nc_list)
        n_pairs = min(len(c_list), len(nc_list))
        for i in range(n_pairs):
            pairs.append((c_list[i], nc_list[i]))

    print(f"  Matched pairs: {len(pairs)}")

    if not pairs:
        print("  No matched pairs found.")
        return

    diffs = [int(c["accepted"]) - int(nc["accepted"]) for c, nc in pairs]
    mean_diff = statistics.mean(diffs)
    print(f"  Mean within-pair acceptance diff (compliant - non-compliant): {mean_diff:.4f}")
    print(f"  = {mean_diff*100:.2f} percentage points")

    comp_rate = statistics.mean(int(c["accepted"]) for c, _ in pairs)
    noncomp_rate = statistics.mean(int(nc["accepted"]) for _, nc in pairs)
    print(f"  Matched compliant acceptance rate: {comp_rate:.4f} ({comp_rate*100:.1f}%)")
    print(f"  Matched non-compliant acceptance rate: {noncomp_rate:.4f} ({noncomp_rate*100:.1f}%)")

    # Bootstrap CI on matched-pairs diff
    boot_diffs = []
    for _ in range(N_BOOTSTRAP):
        sample = random.choices(diffs, k=len(diffs))
        boot_diffs.append(statistics.mean(sample))
    boot_diffs.sort()
    ci_lo = boot_diffs[int(0.025 * N_BOOTSTRAP)]
    ci_hi = boot_diffs[int(0.975 * N_BOOTSTRAP)]
    print(f"  Bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"                  = [{ci_lo*100:.2f}pp, {ci_hi*100:.2f}pp]")

    # Permutation test
    observed = mean_diff
    n_perm = N_BOOTSTRAP
    count_extreme = 0
    for _ in range(n_perm):
        perm_diffs = []
        for c, nc in pairs:
            if random.random() < 0.5:
                perm_diffs.append(int(c["accepted"]) - int(nc["accepted"]))
            else:
                perm_diffs.append(int(nc["accepted"]) - int(c["accepted"]))
        if abs(statistics.mean(perm_diffs)) >= abs(observed):
            count_extreme += 1
    p_value = count_extreme / n_perm
    print(f"  Permutation test p-value (two-sided): {p_value:.4f}")

    return mean_diff


def simpson_paradox_table(data, label: str):
    print(f"\n{'='*70}")
    print(f"SIMPSON'S PARADOX DECOMPOSITION: {label}")
    print(f"{'='*70}")

    strata = {
        "Low (< 4.5)": lambda d: d["mean_score"] < 4.5,
        "Borderline (4.5-6.5)": lambda d: 4.5 <= d["mean_score"] <= 6.5,
        "High (> 6.5)": lambda d: d["mean_score"] > 6.5,
    }

    for stratum_name, pred in strata.items():
        subset = [d for d in data if pred(d)]
        if not subset:
            continue
        comp = [d for d in subset if d["has_compliance"]]
        nocomp = [d for d in subset if not d["has_compliance"]]
        comp_rate = statistics.mean(int(d["accepted"]) for d in comp) if comp else 0
        nocomp_rate = statistics.mean(int(d["accepted"]) for d in nocomp) if nocomp else 0
        diff = comp_rate - nocomp_rate
        print(f"\n  {stratum_name}: N={len(subset)}")
        print(f"    Compliant: {len(comp)}, acceptance = {comp_rate*100:.1f}%")
        print(f"    Non-compliant: {len(nocomp)}, acceptance = {nocomp_rate*100:.1f}%")
        print(f"    Difference: {diff*100:+.1f}pp")


def bootstrap_aggregate_lift(data, label: str):
    print(f"\n{'='*70}")
    print(f"BOOTSTRAP CI FOR AGGREGATE COMPLIANCE LIFT: {label}")
    print(f"{'='*70}")

    comp = [d for d in data if d["has_compliance"]]
    nocomp = [d for d in data if not d["has_compliance"]]
    obs_diff = (statistics.mean(int(d["accepted"]) for d in comp) -
                statistics.mean(int(d["accepted"]) for d in nocomp))
    print(f"  Observed aggregate lift: {obs_diff*100:.1f}pp")

    boot_diffs = []
    for _ in range(N_BOOTSTRAP):
        bc = random.choices(comp, k=len(comp))
        bnc = random.choices(nocomp, k=len(nocomp))
        bd = (statistics.mean(int(d["accepted"]) for d in bc) -
              statistics.mean(int(d["accepted"]) for d in bnc))
        boot_diffs.append(bd)
    boot_diffs.sort()
    ci_lo = boot_diffs[int(0.025 * N_BOOTSTRAP)]
    ci_hi = boot_diffs[int(0.975 * N_BOOTSTRAP)]
    print(f"  Bootstrap 95% CI: [{ci_lo*100:.1f}pp, {ci_hi*100:.1f}pp]")

    # Same for borderline only
    comp_b = [d for d in comp if is_borderline(d)]
    nocomp_b = [d for d in nocomp if is_borderline(d)]
    if comp_b and nocomp_b:
        obs_b = (statistics.mean(int(d["accepted"]) for d in comp_b) -
                 statistics.mean(int(d["accepted"]) for d in nocomp_b))
        print(f"\n  Borderline stratum observed lift: {obs_b*100:.1f}pp")
        boot_b = []
        for _ in range(N_BOOTSTRAP):
            bc = random.choices(comp_b, k=len(comp_b))
            bnc = random.choices(nocomp_b, k=len(nocomp_b))
            bd = (statistics.mean(int(d["accepted"]) for d in bc) -
                  statistics.mean(int(d["accepted"]) for d in bnc))
            boot_b.append(bd)
        boot_b.sort()
        ci_lo_b = boot_b[int(0.025 * N_BOOTSTRAP)]
        ci_hi_b = boot_b[int(0.975 * N_BOOTSTRAP)]
        print(f"  Borderline bootstrap 95% CI: [{ci_lo_b*100:.1f}pp, {ci_hi_b*100:.1f}pp]")


def main():
    print("EXPERIMENT 1: DEMAND-COMPLIANCE ANALYSIS")
    print("=" * 70)

    # Load and merge data for both years
    for year in [2024, 2023]:
        data = merge_data(year)
        print(f"\nICLR {year}: {len(data)} papers with scores + demand/compliance data")

        # Descriptive stats
        print(f"  Mean score: {statistics.mean(d['mean_score'] for d in data):.2f}")
        print(f"  Acceptance rate: {statistics.mean(d['accepted'] for d in data)*100:.1f}%")
        print(f"  Has demand: {sum(d['has_demand'] for d in data)} ({sum(d['has_demand'] for d in data)/len(data)*100:.1f}%)")
        print(f"  Has compliance: {sum(d['has_compliance'] for d in data)} ({sum(d['has_compliance'] for d in data)/len(data)*100:.1f}%)")

        borderline = [d for d in data if is_borderline(d)]
        print(f"  Borderline (4.5-6.5): {len(borderline)} papers")

        # Simpson's paradox decomposition
        simpson_paradox_table(data, f"ICLR {year}")

        # Bootstrap CIs for lifts
        bootstrap_aggregate_lift(data, f"ICLR {year}")

        # Logistic regressions
        run_regression(data, f"ICLR {year} — ALL PAPERS")
        run_regression(borderline, f"ICLR {year} — BORDERLINE ONLY")

        # Matched pairs on borderline
        run_matched_pairs(borderline, f"ICLR {year} — BORDERLINE")

    # Pooled 2023+2024 analysis
    print("\n\n" + "=" * 70)
    print("POOLED 2023 + 2024 ANALYSIS")
    print("=" * 70)
    pooled = merge_data(2024) + merge_data(2023)
    print(f"Total pooled: {len(pooled)} papers")
    borderline_pooled = [d for d in pooled if is_borderline(d)]
    print(f"Borderline pooled: {len(borderline_pooled)} papers")

    simpson_paradox_table(pooled, "POOLED 2023+2024")
    bootstrap_aggregate_lift(pooled, "POOLED 2023+2024")
    run_regression(borderline_pooled, "POOLED BORDERLINE")
    run_matched_pairs(borderline_pooled, "POOLED BORDERLINE")


if __name__ == "__main__":
    main()

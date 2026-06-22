"""Generate lifecycle and correction adoption figures for benchmarks_engines paper."""
import json
import sys

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:
    print("matplotlib not available")
    sys.exit(1)

data = json.load(open("/Users/elliottower/Downloads/mc_iayn/data/openalex_citation_analysis.json"))
bl = data["benchmark_lifecycles"]
ca = data["correction_adoption"]

# --- Figure 1: Benchmark Lifecycle Curves ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [3, 2]})

# Panel A: Absolute citation curves (selected benchmarks)
selected = ["ImageNet", "GLUE", "SuperGLUE", "SQuAD", "CIFAR-10", "HumanEval", "MMLU"]
colors = {
    "ImageNet": "#2c3e50",
    "CIFAR-10": "#7f8c8d",
    "GLUE": "#e74c3c",
    "SuperGLUE": "#c0392b",
    "SQuAD": "#e67e22",
    "MMLU": "#3498db",
    "HumanEval": "#2ecc71",
}
styles = {
    "ImageNet": "-",
    "CIFAR-10": "--",
    "GLUE": "-",
    "SuperGLUE": "--",
    "SQuAD": "-.",
    "MMLU": "-",
    "HumanEval": "--",
}

for name in selected:
    by_year = bl[name]["by_year"]
    years = sorted([int(y) for y in by_year.keys() if int(y) >= 2015])
    cites = [by_year[str(y)] for y in years]
    # Only plot years with nonzero citations or after first nonzero
    first_nonzero = next((i for i, c in enumerate(cites) if c > 0), 0)
    years = years[first_nonzero:]
    cites = cites[first_nonzero:]
    ax1.plot(years, cites, styles.get(name, "-"), color=colors.get(name, "#333"),
             label=name, linewidth=2, marker="o", markersize=3)

ax1.set_xlabel("Year", fontsize=11)
ax1.set_ylabel("Citations per year", fontsize=11)
ax1.set_title("(a) Benchmark adoption and abandonment", fontsize=12, fontweight="bold")
ax1.legend(fontsize=8, loc="upper left", ncol=2)
ax1.set_xlim(2015, 2025)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=9)

# Panel B: Correction adoption rates (% of ML papers)
corrections = {
    "Benjamini-Hochberg": "Benjamini-Hochberg 1995",
    "Bonferroni": "Dunn 1961 (Bonferroni)",
    "Holm": "Holm 1979",
}
corr_colors = {
    "Benjamini-Hochberg": "#e74c3c",
    "Bonferroni": "#3498db",
    "Holm": "#2ecc71",
}

total_ml = ca["total_ml_by_year"]

for label, key in corrections.items():
    pct = ca[key]["pct_of_ml_by_year"]
    years = sorted([int(y) for y in pct.keys()])
    vals = [pct[str(y)] for y in years]
    ax2.plot(years, vals, "-o", color=corr_colors[label], label=label,
             linewidth=2, markersize=3)

ax2.set_xlabel("Year", fontsize=11)
ax2.set_ylabel("% of ML papers citing", fontsize=11)
ax2.set_title("(b) Statistical correction adoption", fontsize=12, fontweight="bold")
ax2.legend(fontsize=8)
ax2.set_xlim(2015, 2025)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.2f}%"))
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig("/Users/elliottower/Downloads/mc_iayn/paper/fig_lifecycle.pdf", bbox_inches="tight", dpi=300)
plt.savefig("/Users/elliottower/Downloads/mc_iayn/paper/fig_lifecycle.png", bbox_inches="tight", dpi=300)
print("Saved fig_lifecycle.pdf and fig_lifecycle.png")

# --- Figure 2: Normalized lifecycle (peak = 1.0) ---
fig2, ax3 = plt.subplots(1, 1, figsize=(7, 4.5))

for name in ["ImageNet", "GLUE", "SuperGLUE", "SQuAD", "MMLU", "HumanEval"]:
    by_year = bl[name]["by_year"]
    years = sorted([int(y) for y in by_year.keys()])
    cites = [by_year[str(y)] for y in years]
    first_nonzero = next((i for i, c in enumerate(cites) if c > 0), 0)
    years = years[first_nonzero:]
    cites = cites[first_nonzero:]
    peak = max(cites)
    if peak > 0:
        normed = [c / peak for c in years_cites] if False else [c / peak for c in cites]
        ax3.plot(years, normed, styles.get(name, "-"), color=colors.get(name, "#333"),
                 label=name, linewidth=2, marker="o", markersize=3)

ax3.set_xlabel("Year", fontsize=11)
ax3.set_ylabel("Citations / peak citations", fontsize=11)
ax3.set_title("Benchmark lifecycles (normalized to peak)", fontsize=12, fontweight="bold")
ax3.legend(fontsize=8, loc="upper left")
ax3.set_xlim(2015, 2025)
ax3.set_ylim(0, 1.15)
ax3.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
ax3.grid(True, alpha=0.3)
ax3.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig("/Users/elliottower/Downloads/mc_iayn/paper/fig_lifecycle_normalized.pdf", bbox_inches="tight", dpi=300)
plt.savefig("/Users/elliottower/Downloads/mc_iayn/paper/fig_lifecycle_normalized.png", bbox_inches="tight", dpi=300)
print("Saved fig_lifecycle_normalized.pdf and fig_lifecycle_normalized.png")

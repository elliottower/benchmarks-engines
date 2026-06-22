"""Generate single-panel lifecycle figure for slides."""
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

fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))

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
    first_nonzero = next((i for i, c in enumerate(cites) if c > 0), 0)
    years = years[first_nonzero:]
    cites = cites[first_nonzero:]
    ax.plot(years, cites, styles.get(name, "-"), color=colors.get(name, "#333"),
            label=name, linewidth=2, marker="o", markersize=3)

ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Citations per year", fontsize=11)
ax.set_title("Benchmark adoption and abandonment", fontsize=12, fontweight="bold")
ax.legend(fontsize=8, loc="upper left", ncol=2)
ax.set_xlim(2015, 2025)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig("/Users/elliottower/Downloads/mc_iayn/paper/figures/updated/fig_lifecycle_single.png", bbox_inches="tight", dpi=300)
plt.savefig("/Users/elliottower/Downloads/mc_iayn/paper/figures/updated/fig_lifecycle_single.pdf", bbox_inches="tight", dpi=300)
print("Saved fig_lifecycle_single.png and .pdf")

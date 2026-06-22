# Benchmarks Are Engines, Not Cameras

This project explores how benchmarks shape ML research, drawing on Donald MacKenzie's framework of performativity from the sociology of finance. The core idea: ML benchmarks don't just measure progress — they actively influence the research they evaluate. When a benchmark becomes the target of optimization, it gradually shifts from measurement tool to optimization target, with consequences for statistical validity, proof standards, and the direction of the field.

The project covers the multiple comparisons problem in ML (and how corrections from medicine and genomics could be adopted), proof culture fragmentation in mechanistic interpretability (where SAE and causal abstraction communities share a foundation but have developed largely independently), and the dynamics of benchmark adoption and abandonment over time.

## Slides

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20792837.svg)](https://doi.org/10.5281/zenodo.20792837)

**[Slides](slides/benchmarks_engines_slides_v1.pdf)** ([LaTeX source](slides/benchmarks_engines_slides_v1.tex)) — 46 slides covering performativity in ML benchmarks, multiple comparisons and FWER/FDR, proof culture fragmentation in mechanistic interpretability, and counterperformativity in peer review.

To compile:
```bash
cd slides
pdflatex benchmarks_engines_slides_v1.tex
```

## Paper

Draft paper in progress (TBD). Current working version in `paper/`.

## Data

| File | Description | Source |
|------|-------------|--------|
| `data/iclr2024_demand_compliance.csv` | Experiment demand and compliance signals for 7,404 ICLR 2024 submissions | OpenReview API |
| `data/iclr2023_demand_compliance.csv` | Same for 3,792 ICLR 2023 submissions | OpenReview API |
| `data/openalex_citation_analysis.json` | Benchmark lifecycle citation counts (ImageNet, GLUE, etc.) | OpenAlex API |
| `data/mi_citation_network_final.json` | Citation network among 27 canonical MI papers | Semantic Scholar API |
| `data/mi_citation_corpus/` | Expanded 688-paper MI citation corpus (209 SAE, 40 Causal, 128 Circuits, 311 general) | Perplexity deep research |
| `data/perplexity_citation_data/` | Independent citation validation dataset (27 papers) | Perplexity deep research |
| `data/fwer_audit_landmark_papers.json` | FWER audit of landmark ML papers | Manual audit |
| `data/mi_audit_papers.json` | MI paper evidence standard audit | Manual audit |
| `data/self_audit_comparisons.json` | Self-audit of implicit comparisons | Manual audit |

## Scripts

| Script | What it does |
|--------|-------------|
| `scripts/citation_network_analysis.py` | Analyze citation links among MI papers via Semantic Scholar |
| `scripts/citation_network_openalex.py` | Fetch benchmark lifecycle data from OpenAlex |
| `scripts/generate_lifecycle_figures.py` | Generate benchmark adoption/abandonment curves |
| `scripts/gen_lifecycle_single.py` | Generate single-panel lifecycle figure for slides |

## Reproducing

```bash
# Benchmark lifecycle figures
pip install matplotlib
python scripts/gen_lifecycle_single.py

# Citation network analysis (requires Semantic Scholar API key)
pip install requests
python scripts/citation_network_analysis.py
```

Some data was sourced via Perplexity deep research and is included directly in `data/` rather than being reproducible via script.

## License

MIT

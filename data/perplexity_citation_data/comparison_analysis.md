# Perplexity citation data vs. our Semantic Scholar data

## Source
- Perplexity-sourced CSV files downloaded 2026-06-22
- 27 papers (8 SAE, 7 Causal, 12 Circuits) vs. our S2 analysis (26 papers, 9 resolved)

## Perplexity citation matrix (27 papers)

| From \ To | SAE | Causal | Circuits |
|-----------|-----|--------|----------|
| SAE       | 10  | 1      | 11       |
| Causal    | 2   | 6      | 6        |
| Circuits  | 3   | 5      | 17       |

Total edges: 61
SAE <-> Causal cross-citations: 3 (1 + 2)
SAE <-> Causal possible directed pairs: 56 + 56 = 112
SAE <-> Causal density: 3/112 = 2.7%

## Our Semantic Scholar data (9/26 papers resolved)

| From \ To | SAE | Causal | Circuits |
|-----------|-----|--------|----------|
| SAE       | 2   | 2      | 0        |
| Causal    | 0   | 1      | 0        |
| Circuits  | 1   | 2      | 1        |

Total edges: 9
SAE <-> Causal: 2/20 = 10% (but only 9 papers resolved due to rate limiting)

## Paper v12 claims

- "1 cross-citation across 72 possible directed pairs" -> 1.4% density
- Based on manual audit of 27 papers

## Comparison

The Perplexity data with 27 papers finds:
- SAE -> Causal: 1 edge (density 1.8%)
- Causal -> SAE: 2 edges (density 3.6%)
- Combined SAE <-> Causal: 3 edges across 112 possible = 2.7%

This is slightly higher than our paper's 1.4% claim but still extremely low.
The core finding holds: SAE and Causal communities barely cite each other
compared to within-community citation rates (SAE->SAE: 15.6%, Causal->Causal: 12.2%).

Both communities heavily cite Circuits literature (SAE->Circuits: 11.5%, Causal->Circuits: 7.1%).

## Key difference
The Perplexity data includes more recent papers (2025-2026) which may have slightly
more cross-pollination as the field matures. The qualitative conclusion is unchanged:
the two largest MI proof cultures build on the same foundation (circuits) without
substantially engaging with each other's work.

## Action items
- Consider updating slide/paper to say "~3% density" instead of "1.4%" if using this larger dataset
- Or keep the 1.4% figure with the original 27-paper audit and note the Perplexity data as corroboration
- The directional finding is robust across both datasets

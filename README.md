# CS2 BTP — Player Role Consistency and Round Outcomes

**Bachelor's Thesis Project · Sachin Kumar (S20240010206) · IIIT Sri City**

An end-to-end research pipeline that (1) derives behavioural player **roles**
from professional Counter-Strike 2 demo files using Self-Organizing Maps
(following Drachen, Canossa & Yannakakis, IEEE CIG 2009), (2) quantifies each
player's **role consistency**, and (3) tests whether prior consistency
predicts **round outcomes** under a leakage-free, prospective design.

**Headline result.** Consistency and winning are strongly associated within a
half (OLS β = 0.54, p = 0.002), but a team's *prior* average consistency does
not predict the next round (logit β = 0.32, 95% CI [−0.88, 1.52]) — a null
robust to eco-exclusion, taxonomy granularity (k ± 1), per-map splits, and
permutation testing. The association appears to run substantially from
winning → consistency, not the reverse. Full treatment: the thesis report.

**Corpus.** 76 professional maps, 10 events (2024–2025), 13 teams,
1,652 rounds, 16,520 player-rounds. Deterministic: fixed seed + fixed config
reproduce every number in the thesis.

## Repository / Drive layout

```
cs2-btp/
├── raw_dems/                 input .dem files (event__stage__match naming)
├── manifest.csv              source of truth: one row per demo + status
├── code/cs2btp/              the Python package (9 modules)
├── notebooks/                7 Colab driver notebooks (run in order)
├── parsed/<demo_id>/         per-demo parquet tables (rounds, ticks, events)
├── features/                 player-round feature tables (+ role labels)
├── models/                   SOM weights, scalers, role profiles & names
├── analysis/                 consistency tables, regression datasets, QC,
│                             stability, permutation & robustness artifacts
├── figures/                  thesis-ready PNGs (200 dpi)
└── docs/                     this documentation set
```

## Quickstart (Google Colab + Drive)

1. Upload `code/cs2btp/` to `My Drive/cs2-btp/code/cs2btp/` and `notebooks/`
   to `My Drive/cs2-btp/notebooks/`; put demos in `raw_dems/`.
2. Open each notebook in Colab and run in order (table below). Every stage is
   resumable; re-running skips finished work.
3. After replacing any `.py` file in Drive: delete
   `code/cs2btp/__pycache__` and **Runtime → Disconnect and delete runtime**
   before re-running (Colab caches imported modules aggressively).

| # | Notebook | Stage | Typical runtime |
|---|----------|-------|-----------------|
| 00 | setup_and_smoke_test | env check; parse ONE demo; validate vs HLTV | 5 min |
| 01 | parse_all_demos | parse everything in the manifest (resumable) | 1–3 min/demo |
| 02 | build_features | 19 behavioural features per player-round (cached) | ~1 min/demo |
| 03 | discover_roles | per-side SOMs; **choose k; name roles**; stability | 45–70 min |
| 04 | consistency_metrics | modal share / entropy / switch; rolling variant | 2 min |
| 05 | outcome_analysis | round-level logit; descriptives; half-level OLS | 2 min |
| 06 | robustness | no-eco; k ± 1; per-map; permutation; weakest-link battery | 60–90 min |

Notebook 03 contains the two human-in-the-loop cells (`K_BY_SIDE`,
`FINAL_NAMES`); both are **locked with the final decisions** for this corpus
and safe to Run-all.

## Documentation index (`docs/`)

| File | Contents |
|------|----------|
| `ARCHITECTURE_AND_DATA.md` | pipeline architecture, data flow, every file schema |
| `MODULE_REFERENCE.md` | API reference for all 9 modules + full config reference |
| `NOTEBOOK_GUIDE.md` | per-notebook inputs/outputs, editable cells, checks |
| `DECISIONS_AND_RESULTS.md` | the complete decision log + results summary |
| `REPRODUCTION_AND_TROUBLESHOOTING.md` | exact reproduction steps, environment, known issues |

## Requirements

Google Colab (CPU is sufficient) with Drive mounted. Installed by the
bootstrap cell: `demoparser2` (tested 0.41.3), `minisom`. Pre-installed on
Colab: pandas, numpy, scipy, scikit-learn, statsmodels, joblib, matplotlib,
pyarrow.

## Methodological lineage

Role discovery replicates **Drachen, Canossa & Yannakakis, "Player Modeling
using Self-Organization in Tomb Raider: Underworld" (IEEE CIG 2009)**:
behavioural telemetry → emergent SOM → cluster the map → manual inspection
and naming → stability analysis. Distinct from **Drachen, Sifa, Bauckhage &
Thurau (IEEE CIG 2012)**, "Guns, Swords and Data" (large-scale behavioural
clustering) — cite both, correctly attributed.

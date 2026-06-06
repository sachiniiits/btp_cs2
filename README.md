# CS2 Player Role Classifier

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![Status](https://img.shields.io/badge/status-active--research-orange)](.)
[![BTP](https://img.shields.io/badge/type-Bachelor's%20Thesis-purple)](.)

> **BTP Mid-Term — Sachin Kumar | UG-2 | S20240010206**
>
> *Measuring Player Role Consistency and Its Impact on Match Outcome in CS2 — using behavioural signals extracted from raw `.dem` replay files, without manual annotation.*

---

## What This Does

CS2 professional players fall into recognisable tactical roles — Entry fraggers who push first, Lurkers who flank late, Supports who flash and trade — but no demo file explicitly labels them. This pipeline extracts 14 per-round behavioural features from raw binary `.dem` recordings and uses an ensemble of three unsupervised clustering algorithms to assign each player a role automatically.

**Two phases:**

1. **G1 — Feature Engineering**: Parse `.dem` → extract 14 metrics per *(player, round, side)* tuple
2. **G2 — Clustering**: Ensemble vote → cluster profiles → role assignment → player-level aggregation

---

## Pipeline Architecture

```
N × .dem files  (binary, ~150 MB each)
       │
       ▼  demoparser2 (Rust-based)
       │
  kills / damage / positions / grenades / round_ends
       │
       ▼  reconstruct_rounds()  +  assign_round_num()
       │
  Events tagged with round_num
       │
       ▼  extract_features_from_dem()  ×  N
       │
  14-feature DataFrame  (~3,600 rows for 12 demos)
  ├── g1_features.parquet          [raw]
       │
       ▼  clip → log1p → StandardScaler
       │
  Scaled DataFrame  (+ 14 "_scaled" columns)
  ├── g1_clean.parquet
       │
       ▼  KMeans  +  GMM  +  HDBSCAN  →  majority vote
       │
  Clustered DataFrame  (+ cluster_id, gmm_max_prob)
  ├── g2_clustered.parquet
       │
       ▼  weighted scoring matrix  →  greedy role assignment
       │
  Labeled DataFrame  (+ role_label)
  ├── g2_labeled.parquet
       │
       ▼  groupby player → majority role → consistency threshold
       │
  Player-level summary  (~30 rows)
  └── g2_player_roles.parquet
```

---

## 14 Behavioural Features

| Dimension | Feature | Range | What It Captures |
|-----------|---------|-------|-----------------|
| **Combat** | `kill_count` | 0–5 | Kills per round (capped at ace) |
| | `damage_dealt` | 0–500 | HP damage dealt |
| | `headshot_pct` | 0–1 | HS fraction — riflers ~60%, AWPers ~15% |
| | `multi_kill_round` | {0,1} | Binary: ≥3 kills in one round |
| | `opening_duel_won` | {−1,0,1} | Won first fight (+1), died first (−1), uninvolved (0) |
| **Utility & Teamplay** | `utility_used` | 0–5 | Grenades thrown |
| | `flash_assists` | 0–5 | Kills where this player's flash blinded the victim |
| | `trade_kills` | 0–5 | Kills within 5 sec (320 ticks) of a teammate's death |
| **Positioning** | `distance_traveled` | 0–5000 | 2D path length per round |
| | `mean_isolation` | 0–2000 | Mean nearest-teammate distance (cKDTree, same-team only) |
| | `zone_primary` | {1,2,3} | Map zone: CT-side / mid / T-side — percentile-adaptive per map |
| **Timing** | `first_engagement_rel` | 0–1 | Relative tick of first kill (0 = round start, 1 = never) |
| | `death_tick_rel` | 0–1 | Relative tick of death (1.0 = survived the round) |
| | `survived` | {0,1} | Binary: alive at round end |

> ✅ **G1 Validation**: All 14 features pass range checks — confirmed in the validation report printed at end of Cell 5.

---

## Clustering Methodology

### K Selection
Both metrics are evaluated over K = 3–7; consensus rule picks the final K.

| Metric | What It Favours | Current Result |
|--------|----------------|----------------|
| Silhouette Score | Tightest, best-separated clusters | Best K = 3 (score ≈ 0.324) |
| GMM BIC | Best fit-to-complexity ratio | Best K = 7 (BIC still falling) |
| **Chosen K** | Midpoint when disagreement > 1 | **K = 5** ( (3+7)/2 = 5 ) |

### Ensemble Vote
Three algorithms label every data point independently. Labels are **Hungarian-aligned** to KMeans before voting. Each point's final cluster is decided by majority.

| Algorithm | Configuration | Strengths |
|-----------|--------------|-----------|
| **KMeans** | K=5, n_init=10 | Fast, stable; reference for alignment |
| **GMM** | Full covariance, n_init=5 | Handles elliptical clusters; soft probabilities |
| **HDBSCAN** | min_cluster_size=20, min_samples=5 | Finds non-convex shapes; noise detection |

HDBSCAN noise points (label = −1) inherit the KMeans label before alignment.

**Ensemble Agreement (all 3 agree): 38.7%**

---

## Results — Mid-Term Evaluation (v3)

### Cluster Profiles

| Cluster | Assigned Role | Rows | % | Key Signals |
|---------|-------------|------|---|-------------|
| C0 | **Support** | 444 | 21.9% | Low kills/dmg, high utility & flash assists |
| C1 | **IGL** | 568 | 28.0% | Low kills, high survival & isolation |
| C2 | **Lurker** | 712 | 35.1% | High distance & isolation, moderate dmg, late entry |
| C3 | **Rifler** | 145 | 7.1% | Highest dmg (~400), ~3.8 kills/rd, high HS%, multi-kills |
| C4 | **AWPer** | 159 | 7.8% | Low kills, very low HS%, high distance, negative opening duel |

### Player-Level Summary (31 players, Flex threshold = 50%)

| Role | Notable Players | Consistency |
|------|----------------|-------------|
| Lurker | hyped, JDC, LNZ | 73%, 63%, 59% |
| IGL | SunPayus, xfl0ud | 50%, 50% |
| Rifler | zweih | 100% (4 rounds only) |
| **Flex** | ropz, chopper, tN1R | 42%, 43%, 47% |

**23 of 31 players (74%) fall below the 50% consistency threshold → classified as Flex.**  
This is the primary open problem — see Issues below.

### Visualisation Outputs

Sample outputs generated by the pipeline (see `outputs/sample/`):

| File | Contents |
|------|----------|
| `g2_correlation_matrix.png` | 14×14 Pearson correlation heatmap |
| `g2_k_selection.png` | Silhouette score + GMM BIC vs K |
| `g2_pca_tsne.png` | PCA and t-SNE cluster views |
| `g2_profile_heatmap.png` | Cluster × feature normalised mean heatmap |
| `g2_radar_charts.png` | One radar chart per cluster |
| `g2_roles_final.png` | Role distribution bar chart + PCA with role colours |
| `g2_player_roles.png` | Player consistency histogram + per-player role distribution |

---

## Bug Fixes: v2 → v3

| Fix | Problem in v2 | Resolution in v3 |
|-----|--------------|-----------------|
| **FIX 1** | `trade_kills` had no cap — values reached 17 due to double-counting | Added `CAP_TRADE_KILLS = 5` and `break` after first matching death |
| **FIX 2** | `zone_primary` always returned 2 — `compute_zone_adaptive([mean_x])` triggered the `len < 10` guard | Pass global X distribution (`all_x`, ~23k values) as reference |
| **FIX 3** | Flash assists checked `attacker_name + flash_duration > 0` — measured kills-while-flashed, not assists given | Use `assister_name == player` AND `assistedflash == True` |
| **FIX 4** | No Rifler role — generic fragger clusters mislabelled as AWPer | Added Rifler to scoring matrix with high `headshot_pct` weight |

---

## Known Issues & Roadmap

| Issue | Severity | Plan |
|-------|----------|------|
| 74% players classified Flex — insufficient per-player data | High | Scale to 100+ demos |
| Cluster imbalance: Lurker = 35%, Rifler = 7% | Medium | More diverse match data |
| Some cluster profile features show exactly 0 (possible column selection bug) | Medium | Investigate raw vs scaled column indexing |
| `distance_traveled` & `mean_isolation` highly correlated (r = 0.93) | Low | Consider dropping one or using PCA pre-clustering |
| Entry Fragger role never appears as assigned cluster | Low | Revisit scoring matrix weights |

---

## Setup

### Requirements
- Python 3.9+
- Google Colab (recommended) — the pipeline mounts Google Drive automatically
- CS2 `.dem` files placed in the configured `raw_dems/` folder

### Install
```bash
pip install -r requirements.txt
```

### Configure paths
Edit the constants in `src/cs2_pipeline_v3.py` (Cell 1):
```python
_BASE      = "/content/drive/MyDrive/cs2-btp"   # your Drive path
DEM_FOLDER = os.path.join(_BASE, "raw_dems")
OUTPUT_DIR = Path(os.path.join(_BASE, "processed"))
```

### Run (Google Colab)
Execute cells in order — each cell is self-contained:
```
Cell 1: Setup & configuration
Cell 2: DEM parsing utilities
Cell 3: Round reconstruction & feature helpers
Cell 4: Feature extraction → g1_features.parquet
Cell 5: Preprocessing & scaling → g1_clean.parquet
Cell 6: G2 clustering & role assignment → g2_*.parquet + PNGs
```

### Getting `.dem` files
Professional match demos are publicly downloadable from [HLTV.org](https://www.hltv.org) under each match's "Demo" tab. Place them in your configured `raw_dems/` folder. See `data/README.md` for details.

---

## Repository Structure

```
cs2-role-classifier/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
│
├── src/
│   └── cs2_pipeline_v3.py        # Full pipeline (Cells 1–6)
│
├── docs/
│   ├── code_explanation.pdf      # Line-by-line explanation of all pipeline stages
│   ├── btp_midterm_slides.pdf    # Mid-term evaluation presentation
│   └── CONTEXT.md                # Design decisions & pipeline context
│
├── outputs/
│   └── sample/                   # Sample visualisation outputs from a test run
│       ├── g2_correlation_matrix.png
│       ├── g2_k_selection.png
│       ├── g2_pca_tsne.png
│       ├── g2_profile_heatmap.png
│       ├── g2_radar_charts.png
│       ├── g2_roles_final.png
│       └── g2_player_roles.png
│
└── data/
    └── README.md                 # How to obtain .dem files (not tracked in git)
```

---

## Notable Correlations (Feature Analysis)

From the correlation matrix generated by the pipeline:

| Correlation | Value | Interpretation |
|-------------|-------|----------------|
| `distance_traveled` ↔ `mean_isolation` | r = 0.93 | High collinearity — both log-scaled but kept |
| `kill_count` ↔ `multi_kill_round` | r = 0.75 | Expected — kill count drives multi-kill flag |
| `kill_count` ↔ `trade_kills` | r = 0.71 | Players with kills also get more trades |
| `damage_dealt` ↔ `first_engagement_rel` | r = −0.47 | Early engagers deal more damage |
| `opening_duel_won` ↔ `first_engagement_rel` | r = −0.48 | Winning the entry = engaging early |
| `zone_primary` ↔ everything | r ≈ 0 | Clean orthogonal feature |

---

## References

- Kuzhii, Y., & Furgala, Y. (2024). *Feature Engineering for Role Assessment in Counter-Strike 2.*
- HLTV Rating 2.0: https://www.hltv.org/news/20695/introducing-rating-20
- demoparser2 (Rust-based CS2 parser): https://github.com/LaihoE/demoparser
- HDBSCAN library: https://hdbscan.readthedocs.io

---

*Sachin Kumar — UG-2 — S20240010206*
# PROJECT_CONTEXT.md — CS2 Role Consistency BTP (Final State)

> **Purpose of this file.** Complete, self-contained context for AI models
> (and humans) working with this project after its completion. Everything
> below is the **canonical, verified final state** as of 2026-07-07. All
> numbers were cross-checked against the archived artifacts and an
> independent model audit. A model reading this file should treat these
> numbers as ground truth, verify any claim against the `analysis/` files
> when possible, and follow the rules in §14 before changing anything.

---

## 1. Project identity

- **Title:** Player Role Consistency and Round Outcomes in Counter-Strike 2 —
  Measuring behavioural role consistency with Self-Organizing Maps and
  estimating its association with round-level match outcomes.
- **Author:** Sachin Kumar (Roll S20240010206), UG-2, Indian Institute of
  Information Technology, Sri City. Bachelor's Thesis Project (BTP).
- **Research question:** Does a player/team maintaining consistent
  behavioural role patterns across rounds correlate with higher round-win
  probability? (Explicitly **correlation**, never causal.)
- **Methodological lineage:** Direct replication-and-extension of
  **Drachen, Canossa & Yannakakis, "Player Modeling using Self-Organization
  in Tomb Raider: Underworld", IEEE CIG 2009** (behavioural telemetry →
  emergent SOM → cluster → manual naming → stability). Distinct paper:
  Drachen, Sifa, Bauckhage & Thurau, "Guns, Swords and Data", IEEE CIG
  **2012** (large-scale behavioural clustering). **Never conflate the two**
  — the author's original slides did; the thesis cites both correctly.

## 2. Status

**COMPLETE.** Pipeline, analysis, robustness, 33-page thesis draft,
documentation set (README + 5 docs), and GitHub packaging guidance all
finished and mutually consistent. Remaining items are human-only:
thesis placeholders (supervisor/department/dates/programme), Word TOC field
update, institute format & AI-policy compliance, fixing the conflated
citation on the **presentation slides**, optional HLTV spot-checks, and one
**optional** archival re-run of notebook 03 (regenerates
`per_cluster_stability_*.json` and `role_buytype_crosstab_*.csv`; changes
no numbers; ~2/100 value — safe to skip).

## 3. Environment & layout

Google Colab (CPU) + Google Drive at `My Drive/cs2-btp/`. Bootstrap installs
`demoparser2` (**0.41.3**) and `minisom`. Layout (mirrored by the repo):

```
cs2-btp/
├── raw_dems/            .dem inputs, named <event>__<stage>__<match>.dem  [not in git]
├── manifest.csv         source of truth; one row per demo; status lifecycle
├── code/cs2btp/         9-module Python package
├── notebooks/           00..06 Colab drivers (run in order; all resumable)
├── parsed/<demo_id>/    per-demo parquet (rounds/ticks/kills/hurts/fires/blinds/bomb/header) [not in git]
├── features/            player_round_features(.roles).parquet
├── models/              som_{T,CT}.joblib, role_profile_{T,CT}.csv, role_names_{T,CT}.json
├── analysis/            all result tables/JSONs (canonical snapshot)
├── figures/             200-dpi PNGs (radars carry FINAL role names)
└── docs/                ARCHITECTURE_AND_DATA, MODULE_REFERENCE, NOTEBOOK_GUIDE,
                         DECISIONS_AND_RESULTS, REPRODUCTION_AND_TROUBLESHOOTING
```

**Colab gotcha (recurred 3×):** after replacing any `.py` on Drive, delete
`code/cs2btp/__pycache__`, check Drive didn't create a duplicate
(`config (1).py`), then **Runtime → Disconnect and delete runtime**; verify
with `print(cfg.__file__, len(cfg.CLUSTER_FEATURES))` → must show 19.

## 4. Dataset (canonical)

76 demos, all `featured`; 10 events (2024–2025); 13 teams; **1,652 rounds**;
**16,520 player-rounds** (8,260 per side). Events (maps): PGL Major
Copenhagen 2024 (12), BLAST.tv Austin Major 2025 (10), Perfect World
Shanghai Major 2024 (10), StarLadder Budapest Major 2025 (9), ESL Pro League
S20 2024 (8), IEM Cologne 2024 (8), IEM Katowice 2025 (7), ESL Pro League
S21 2025 (6), IEM Chengdu 2025 (3), IEM Cologne 2025 (3). Maps: nuke 18,
mirage 13, inferno 13, ancient 11, dust2 10, anubis 5, train 2, overpass 2,
vertigo 2 (thin four excluded from per-map robustness, kept in pooled FE).

**QC:** 75/76 clean. The one flag:
`iem-cologne-2024__final__vitality-vs-natus-vincere-m3-mirage`, 42 rounds =
the official **22–20 triple-overtime** map (verified vs HLTV). The QC
columns `score_T_wins/score_CT_wins` = 19/23 count rounds won by the **T
side vs CT side** — a *different quantity* from the team score. Demo
retained; QC round-count ceiling later raised 40 → 60; QC **flags, never
gates** (features stage processes all parsed demos).

**Parsing:** rounds from `round_freeze_end`/`round_end` with
`is_warmup_period` filtering; winner map {2: T, 3: CT}; round-length sanity
4–400 s; player state sampled at **2 Hz** (stride 32 of 64 ticks); events at
full resolution; `round_num` assigned by tick intervals; one demo in memory
at a time (Drive→local copy→parse→cleanup).

## 5. Configuration (final, locked for this corpus)

Seed **42** everywhere (SOM, codebook k-means, subsampling, stability,
permutation) ⇒ full determinism (verified: independent re-runs reproduced
permutation JSONs byte-identically and the main logit to 4 decimals).
TICK_HZ 2 · SOM 22×22 grid (Vesanto 5√N), **100,000 iterations** (raised
from 20k; measured stability gain ≈ +0.05 ARI, plateau beyond), σ 1.5,
lr 0.5 · STABILITY_RUNS 8 · k range 4–7 · MIN_PRIOR_ROUNDS 3 ·
eco < $2,000 / force < $3,900 (team avg freeze-end equip) ·
OPENING_WINDOW 20 s · EARLY_UTIL 25 s · TRADE_RADIUS 700 · TELEPORT 2000.

## 6. Feature set (19; behaviour-only by design)

Positioning: dist_nearest_teammate, dist_team_centroid, distance_traveled,
forward_disp_20s, zone_entropy, site_time_share. Timing:
time_to_first_contact_s, opening_duel_involved. Utility: flashes/smokes/
mollies/he_thrown, util_early_share. Weapon style: awp_share, rifle_share,
shots_fired, trade_presence, time_alive_share. Economy: buy_value_rel_team.

**Computed but EXCLUDED from clustering** (each exclusion is a documented
finding): `planted_bomb`/`defused_bomb` (round events; defuses occur only in
CT-won rounds → would couple taxonomy to outcomes; dev runs produced a
degenerate "defuser role" ≈2% of CT rounds), `enemies_flashed` (utility
*success*, not choice; heavy-tailed count formed a "flash-outlier" pocket
≈1.5% hijacking a cluster at any k), and performance context `ctx_kills/
deaths/damage/won_round`.

## 7. Role taxonomy (final, locked)

`K_BY_SIDE = {'T': 4, 'CT': 5}`. Final k-tables (silhouette /
min_cluster_share / max): T — k4 **0.106**/0.063/0.426; k5 0.093/0.060;
k6 0.091; k7 0.090. CT — k4 0.108/**0.107**/0.477; k5 0.090/**0.107**/0.258;
k6 0.081; k7 0.078. **CT decision rationale (viva-critical):** k=5 trades a
marginal silhouette edge (0.108→0.090) for splitting the dominant 48%
rifler mass into orthogonal, balanced Site Anchor vs Rotating Rifler
archetypes — better construct validity and consistency resolution where the
data mass lies; k=4 covered by the k±1 robustness run. Silhouettes ≈0.1 are
expected (behaviour is a continuum; soft clusters mirror Drachen 2009).

Final `role_id → name` mapping and shares of player-rounds:

| Side | id | Name | n | Share | Signature (z vs side mean) |
|---|---|---|---|---|---|
| T | 0 | Aggressor | 3,522 | 42.6% | contact −0.36, opening +0.33, alive −0.59, minimal util; absorbs eco rushes (39% eco) |
| T | 1 | AWPer | 519 | 6.3% | awp_share +3.58, buy +1.49, shots −0.68 |
| T | 2 | Support Rifler | 3,309 | 40.1% | travel +0.62, util +0.4…0.5, site +0.43, alive +0.52 |
| T | 3 | Lurker | 910 | 11.0% | dist features +1.89/+1.80, trade −0.55 |
| CT | 0 | Site Anchor | 2,076 | 25.1% | dist +0.63/+0.80, site +0.52, entropy −0.47, late contact |
| CT | 1 | AWPer | 883 | 10.7% | awp +2.74, buy +1.11, shots −0.70 |
| CT | 2 | Rotating Rifler | 2,134 | 25.8% | travel +0.63, entropy +0.50, rifle +0.68, high util |
| CT | 3 | Eco/Save | 1,720 | 20.8% | util uniformly negative (early-util −0.80), low buy |
| CT | 4 | Aggressor | 1,447 | 17.5% | opening +1.23, contact −0.70, alive −1.34 |

**Numbering shuffles between fits.** Any re-fit (new data or config) must
re-match names to profiles by *signature* (above), never by index; the
locked notebook 03 contains a signature cheat-sheet and expected-share
tripwires. Full z-profiles: `models/role_profile_{T,CT}.csv`.

## 8. Validity evidence

- **Economy crosstab (T, row-normalised eco/force/full):** Aggressor
  .39/.14/.47 · AWPer .09/.03/.88 · Support .06/.11/.83 · Lurker
  .14/.09/.77. CT Eco/Save confirmed predominantly non-full-buy.
- **Star-player face validity (top-2 modal roles):** ZywOo CT AWPer .46 /
  Eco .18, T Agg .35 / Sup .31 · m0NESY CT AWPer .53 / Eco .23 ·
  broky CT AWPer .48 / Eco .20 · donk CT Rotating Rifler .39 / Agg .25,
  **T Aggressor .54**, T AWPer rounds 1.5% (NOT zero — a corrected claim).
  Patterns: every AWPer's #2 CT role is Eco/Save (economy forces role
  departures); T distributions more dispersed than CT (execute-dependent vs
  structural).
- **Most consistent player-halves (n_rounds ≥ 8):** sh1ro, mezii et al. at
  0.917 (11/12); IGL apEX appears at both extremes (0.90 T half; 0.25 CT
  halves). Unfiltered top-10 is an artifact of short OT halves — always
  filter n ≥ 8 for display.
- **Stability:** global ARI (8 bootstrap+seed retrainings) T 0.471
  (min 0.344), CT 0.561 (min 0.481); k-means agreement 0.390 / 0.600.
  Per-cluster best-match Jaccard (6 retrainings, seed 123):
  T {Aggressor .68, AWPer .90, Support .73, Lurker .47};
  CT {Anchor .53, AWPer .93, Rotator .64, Eco .74, Aggressor .70}.
  Reading: **specialists reproduce strongly; positional-isolation roles are
  soft boundaries** — role structure is real but graded. Downstream analysis
  uses ONE fixed-seed taxonomy, so regression validity never depended on
  run-to-run agreement; taxonomy dependence tested directly via k±1.

## 9. Consistency metrics

Scoped to **player-halves** (`half_key = demo|steamid|h<k>`, contiguous
side blocks, OT-safe) because T/CT taxonomies are separate. Match-level
descriptives (n = 1,670 halves): modal_share mean **0.548** (SD 0.160,
range 0.25–1.00); entropy_norm 0.626 (normalised by log₂k per side);
switch_rate **0.589** (roles are round-level tactical assignments;
consistency is a modal property of a half). **Caveat:** T vs CT modal
shares are NOT directly comparable (k=4 vs k=5 floors); models use
within-round differentials.

**Rolling (leakage-free) variant:** for round t, computed from the half's
rounds strictly before t; NaN until ≥3 priors ⇒ rounds 1–3 of every half
(including both pistols) never enter the regression. Team-round table:
3,304 rows, **2,306 usable (69.8%)**. Aggregations: team mean (`cons_mean`,
headline) and team minimum (`cons_min`, weakest-link).

## 10. Outcome analysis (canonical numbers)

Round dataset: 1,652 rounds, **1,153 usable**. One row per round, T-side
perspective; SEs clustered by match (effective N ≈ 76, honestly reported
via CIs). `pistol` auto-drops (constant 0 in-sample by construction);
degenerate covariates auto-dropped in single-map subsets.

**Main logit** (pseudo-R² 0.0868): cons_diff **β = 0.319** (SE 0.613,
p = 0.602, 95% CI [−0.881, 1.520]) — informative null; a 0.1 consistency
shift moves win probability by −2…+4 pp. Positive control equip_diff_k
β = 0.328/$1k (z = 9.83, p < .001). score_diff −0.026 (p = .043; MR12
loss-bonus artifact, tracked from current-T perspective — coarse control,
do not over-read). round_num −0.011 (ns). Map FE recover known balance:
nuke −0.576 (p = .0095), train −0.557 (p = .102). Quartile T-win rates:
48.4 / 46.5 / 46.0 / 46.7% (flat).

**Alternatives:** entropy_diff β = 0.0015 (p = .998). cons_min_diff
β = 1.102 (p = .0387) → sent to battery (one of three pre-specified
measures; would not survive multiple-comparisons correction).

**Contemporaneous half-level OLS** (n = 152 team-halves, deliberately NOT
leakage-protected): team_modal_share **β = 0.542** (SE 0.178, p = 0.002);
side(T) −0.073 (p = .041); R² 0.044.

## 11. Robustness (canonical)

**cons_diff — null is robust:** no-eco β = 0.254 (p = .710, n = 680);
k−1 β = −0.298 (p = .603); k+1 β = 0.927 (p = .112); per-map betas scatter
(ancient −0.715 n164 · anubis +3.037 n80 · dust2 +0.574 n150 · inferno
−0.431 n181 · mirage +1.869 n208 · nuke +0.032 n282); permutation
(role_id shuffled among 5 teammates within demo×round×side, 100 replicates):
real 0.319, permuted −0.154 ± 0.387, **p = 0.47**. Permuted mean sits
slightly <0 because all-same-role rounds (ecos) are shuffle-invariant.

**cons_min_diff — fragile, exploratory, likely economy-mediated:**
no-eco β **flips sign** to −0.516 (p = .427); k−1 collapses to 0.295
(p = .522); k+1 0.777 (p = .187); permutation real 1.102, permuted
0.222 ± 0.424, p = 0.010 (= minimum attainable at 100 replicates).
Interpretation logic (viva-critical): sign-flip under subsetting =
specification dependence; the permutation rejects only "no player-linked
structure", which cannot distinguish consistency from economy-trajectory
artifacts — the eco-exclusion check adjudicates, and selects the artifact
(team-minimum is exactly the statistic eco rounds crater).

## 12. Canonical conclusion (exact framing)

Consistency and success travel together within halves (level 3), but prior
average consistency does not predict the next round (level 1, robust null
with large effects ruled out), implying much of the raw association runs
through the **reverse channel — winning enables consistency** (economy
preserved → plans preserved → roles preserved). The weakest-link signal
(level 2) dissolved under exactly the check built for its confound.
Durable contributions independent of the null: the validated per-side
behavioural taxonomy, and the leakage-free measurement + clustered
inference + uniform robustness-battery framework that can *distinguish*
the causal channels. Always frame as correlation; the null is the headline,
not a failure.

## 13. Development history (bugs found & fixed — all in current code)

1. demoparser2 0.41.x returns `[]` (list) for absent event types →
   `_safe_event` coerces to DataFrame.
2. Drive/Colab module-caching ghosts (duplicate `config.py`, stale
   `__pycache__`) — caused one full "rerun" to silently use old config.
3. Interim degenerate clusters (defuser ~2%, flash-outlier ~1.5%) →
   feature exclusions (§6). Feature-set evolution: 22 → 20 → **19**.
4. SOM under-training: 20k → 100k iterations (+~0.05 ARI, plateau).
5. CT k evolution: 4 → 5 → (post-exclusion re-evaluation) → **5**, with the
   anchor/rotator split as the deciding evidence each time.
6. Robustness loops overwrote canonical parquets with permutation
   replicates → `save=False` flags on `rolling`, `team_round_table`,
   `build_round_dataset`; `permutation_test` internals never save.
7. Radar PNGs saved pre-naming carried heuristic labels → notebook 03 now
   re-saves radars with FINAL_NAMES after the naming cell.
8. QC ceiling 40 → 60 after the legitimate 42-round triple-OT final;
   docstring corrected to "flags, not gates".
9. Thesis corrections after independent model audit (all verified):
   quadruple→triple OT; 19–23 side-tally vs 22–20 team score; softened
   scoreboard-validation claim; donk 1.5% not 0%; "Rotating Rifler"
   unified; score_diff perspective caveat; permutation p-resolution note;
   behaviour–performance boundary limitation added (time_alive_share,
   opening_duel_involved, trade_presence are outcome-tinged); interim
   cluster numbers reworded as reproducible development observations.
   **The audit found zero numerical errors in the analysis.**

## 14. Rules for future models

- **Do not** change `config.py`, `K_BY_SIDE`, or `FINAL_NAMES` for this
  corpus; every thesis number depends on them. The taxonomy and results are
  frozen at tag-equivalent state "v1.0-thesis".
- **Verify before asserting:** headline numbers live in `analysis/`;
  prefer refitting from `round_outcome_dataset.parquet` over quoting.
- **If extending with new demos:** rerun 01→02→03 (k-selection re-read;
  re-match names by signature — numbering WILL shuffle) →04→05→06; rerun
  BOTH robustness batteries for any coefficient you intend to report; keep
  the correlation framing and the pre-specified decision rule (sign+size
  stability across no-eco and k±1, plus permutation).
- **Citation discipline:** Drachen 2009 = SOM methodology replicated here;
  Drachen 2012 = large-scale clustering. Slides historically conflated them.
- **Phrasing traps:** 19–23 ≠ team score; T/CT modal shares not comparable;
  pistols excluded by construction (not a modelling choice); effective
  N ≈ 76 matches; cons_min is exploratory; nulls reported via CI, not p.
- Content produced by AI assistance: the author must be able to defend
  every sentence; institute AI-policy compliance is the author's check.

## 15. Deliverables inventory

Thesis: `BTP_Thesis_Sachin_Kumar.docx` (33 pp; placeholders highlighted
yellow; TOC needs Word field update). Code+docs: `cs2-btp-pipeline.zip`
(README, docs/×5, code/cs2btp ×9 modules, notebooks ×7).
Results snapshot: manifest.csv + models/ + analysis/ + figures/ (≈2 MB
total; `features/player_round_features_roles.parquet` ≈1–2 MB enables
demo-free reproduction from the roles stage). Scripts: PowerShell demo
extractor (event__stage__ prefixing + dedupe). Not distributable: raw
.dem files (HLTV content) — `manifest.csv` is the dataset specification
for re-download.

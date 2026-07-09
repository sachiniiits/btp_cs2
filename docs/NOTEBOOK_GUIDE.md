# Notebook Guide

All seven notebooks share an identical bootstrap cell (mount Drive, pip
install `demoparser2` + `minisom`, put `code/` on `sys.path`, import
`cs2btp`, `ensure_dirs()`). Run top-to-bottom in numerical order. After
replacing any `.py` in Drive: delete `code/cs2btp/__pycache__`, then
**Runtime → Disconnect and delete runtime** before re-running.

## 00 · setup_and_smoke_test (once, ~5 min)

Purpose: prove the environment end-to-end on ONE demo before committing to
the corpus. Syncs the manifest, parses the first pending demo, prints the
per-round winner list, runs QC, plots one round's positional traces.
**Manual check:** the printed round count and winner tallies must be
consistent with the official HLTV scoreline for that map.

## 01 · parse_all_demos (resumable; 1–3 min/demo)

Drives parsing off the manifest; demos already `parsed` are skipped, so
re-run freely after adding files or a Colab disconnect. Ends with the QC
report and a list of failures. Expected hygiene: 1–2 corrupt HLTV archives
per ~60 is normal — replace the file and `mf.reset_failed()`. QC *flags*
(round count 13–60, 10 players, side presence, coverage) are review
prompts, not gates; the corpus's one flag was a legitimate 42-round
triple-overtime final, verified and retained.

## 02 · build_features (~1 min/demo, cached)

One row per (demo, round, player); per-demo feature files cached in
`features/`. Sanity cells report per-side counts, buy-type mix, feature
summaries, zero-shares and NaN rates. Expected at this corpus: 8,260 rows
per side, max NaN share ≈ 0.

## 03 · discover_roles (45–70 min; the human-in-the-loop notebook)

Sections: (1) standardize per side → (2) k-selection tables →
(3) fit SOMs, print profiles, U-matrices + radar figures →
(4) manual naming → (5) stability (global ARI + per-cluster Jaccard) →
Appendix (buy-type crosstabs, star-player face validity).

The two decision cells are **locked** for this corpus:

* `K_BY_SIDE = {'T': 4, 'CT': 5}` — T: clear silhouette peak; CT: k = 5
  trades a marginal silhouette edge for splitting the dominant rifler mass
  into orthogonal Site Anchor / Rotating Rifler archetypes (rationale
  comment inline; k = 4 covered by notebook 06's k ± 1 run).
* `FINAL_NAMES` — T: Aggressor / AWPer / Support Rifler / Lurker;
  CT: Site Anchor / AWPer / Rotating Rifler / Eco/Save / Aggressor.

Because the pipeline is deterministic, Run-all reproduces the identical
taxonomy. **If demos are added or config changes**, cluster numbering can
shuffle: re-read the profiles against the signature cheat-sheet in §3 and
re-match `FINAL_NAMES` before trusting downstream stages. A post-naming
cell re-saves the radar figures with the final names so `figures/` always
matches the report. Outputs archived: `models/*`, labeled features parquet,
`stability_*.csv`, `per_cluster_stability_*.json`,
`role_buytype_crosstab_*.csv`.

## 04 · consistency_metrics (~2 min, Run-all)

Match-level descriptives + histogram (a filtered `n_rounds ≥ 8` variant is
the thesis figure; short OT halves trivially score modal share 1.0), the
face-validity most/least-consistent tables, then the prospective rolling
metric and the team-round table. Expected: 3,304 team-rounds, 2,306 usable
(69.8% — rounds 1–3 of each half are NaN by design).

## 05 · outcome_analysis (~2 min, Run-all)

Builds the round dataset (1,652 rounds, 1,153 usable), fits the main logit
(SEs clustered by match), plots coefficients and the quartile gradient,
runs the two alternative consistency measures and the half-level OLS.
Reading order: check the equipment control first (must be strongly
positive — it is the pipeline's positive control), then the map fixed
effects (nuke/train CT-sided = face validity), then the consistency
coefficient with its confidence interval.

## 06 · robustness (60–90 min, Run-all)

The same battery for every candidate effect: (a) eco-exclusion,
(b) k ± 1 re-discovery, (c) per-map splits (≥80 rounds & ≥3 matches),
(d) permutation test (100 replicates; minimum attainable p = 0.01) — first
for `cons_diff`, then section (e) repeats everything for the weakest-link
`cons_min_diff`. All variant computations pass `save=False`; canonical
artifacts from 04–05 are never overwritten. Decision rule (fixed in
advance): an effect is robust only if its point estimate keeps sign and
rough size across (a)–(b) *and* survives (d); a sign flip under subsetting
marks it specification-dependent.

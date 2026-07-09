# Reproduction and Troubleshooting

## 1. Environment

* Google Colab, standard CPU runtime, Google Drive mounted.
* Installed by the bootstrap cell: `demoparser2` (developed against
  **0.41.3**) and `minisom`. Everything else (pandas, numpy, scipy,
  scikit-learn, statsmodels, joblib, matplotlib, pyarrow) ships with Colab.
* Determinism: `RANDOM_SEED = 42` in `config.py` seeds the SOM, codebook
  k-means, k-selection subsample, stability and permutation machinery.
  Fixed seed + fixed config + fixed feature files ⇒ byte-identical taxonomy
  and downstream numbers (verified: independent re-runs reproduced the
  permutation JSONs exactly and the main logit to four decimals).

## 2. Full reproduction from raw demos

1. **Acquire demos.** Download GOTV archives from HLTV per the selection
   rules (2024+, top-tier events, ≥10 teams, ≥6 maps per map name).
   Extract with the provided PowerShell script, which renames every `.dem`
   to `<event-slug>__<stage-slug>__<original-name>.dem` and dedupes
   collisions — this naming is what keeps the manifest sane.
2. **Upload** to `My Drive/cs2-btp/raw_dems/`; place `code/cs2btp/` and
   `notebooks/` per the README. If the same match exists under two names
   (e.g., an old un-prefixed copy), delete one: **duplicate matches
   silently bias every analysis** and the manifest cannot detect them.
3. **Run notebooks 00 → 06** per the Notebook Guide. 00 once; 01–02 whenever
   demos are added; 03 whenever features change (re-verify names!); 04–06
   downstream. End-to-end on this corpus: ≈ 4–6 hours of Colab time, most
   of it unattended in 01, 03 and 06.
4. **Verify** against the archived artifacts: `analysis/` and `figures/`
   should match the committed results snapshot (see
   DECISIONS_AND_RESULTS.md, Part 2).

## 3. Known issues and their fixes (encountered during development)

| symptom | cause | fix (already in the code) |
|---|---|---|
| `AttributeError: 'list' object has no attribute 'empty'` during parsing | demoparser2 0.41.x returns an empty **list**, not a DataFrame, when a demo has zero events of a requested type (e.g., `player_blind`) | `_safe_event` coerces non-DataFrame returns; `_filter_warmup` hardened |
| Edited `.py` on Drive but behaviour unchanged | Colab caches imported modules; Drive can also keep duplicate `config.py`/`config (1).py`; stale `__pycache__` | delete duplicates; `!rm -rf .../cs2btp/__pycache__`; **Disconnect and delete runtime**; verify with `print(cfg.__file__, len(cfg.CLUSTER_FEATURES))` |
| A valid demo flagged by QC for round count | 42-round triple-overtime final exceeded the original 40-round ceiling | ceiling raised to 60; QC documented as *flagging*, not gating — flagged demos are manually verified |
| QC `score_T_wins/score_CT_wins` ≠ official score | those columns tally rounds won by the **T side** vs **CT side** — a different quantity from the team score (teams swap sides) | documentation clarified; use round-count + manual check against HLTV for validation |
| Canonical parquets contained permutation replicates | robustness loops called builders that save on every invocation | `rolling`, `team_round_table`, `build_round_dataset` take `save=`; all variants and `permutation_test` internals pass `save=False` |
| Radar PNGs carried heuristic names ("T-role-0") | figures were saved before the manual-naming step | notebook 03 re-saves radars with `FINAL_NAMES` after naming |
| `FINAL_NAMES` KeyError at high k | a codebook cluster can receive zero samples | notebook uses `.get(i, 'unused-N')`; an `unused` role signals k is too high |
| Singular-matrix crash on single-map subsets | constant covariates (map FE, pistol) in a subset | `fit_main_logit` drops degenerate terms automatically (pistol is *always* dropped: pistol rounds cannot have ≥3 priors) |
| Colab disconnect mid-parse / RAM pressure | long unattended runs | everything is manifest-driven and resumable; one demo in memory at a time — just re-run the notebook |
| Corrupt HLTV archive fails to parse | ~1–2 per 60 downloads is normal | manifest row gets `failed` + error; replace the file, `mf.reset_failed()`, re-run 01 |

## 4. Data-hygiene rules (hard-won)

* One manifest row per demo; fill `event` (auto) and optionally `date` as
  files arrive — never trust bare filenames across events.
* Never mix two copies of the same match under different names.
* Re-running 03 after adding demos: cluster numbering may shuffle even with
  a fixed seed (the data changed) — re-match `FINAL_NAMES` to profiles via
  the signature cheat-sheet before running 04+.
* Treat `analysis/` + `figures/` + `models/` + `manifest.csv` as the
  citable results snapshot; archive a zip of them alongside any report.

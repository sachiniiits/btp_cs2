# Module Reference — `cs2btp`

Nine modules; every tunable lives in `config.py`. Functions that write
canonical files accept `save=True|False` where robustness variants need to
avoid overwriting them.

## config.py — all decisions in one place

| constant | value | meaning |
|---|---|---|
| DRIVE_ROOT | /content/drive/MyDrive/cs2-btp | project root on Drive |
| LOCAL_TMP | /content/tmp_dems | Colab-local scratch for .dem copies |
| DEFAULT_TICKRATE | 64 | CS2 native tick rate |
| TICK_HZ / TICK_STRIDE | 2 / 32 | player-state sampling rate |
| TICK_PROPS | 10 props | X, Y, Z, health, is_alive, active_weapon_name, current_equip_value, last_place_name, team_num, team_clan_name |
| OPENING_WINDOW_S | 20 | window for forward-displacement feature |
| EARLY_UTIL_S | 25 | "early utility" cutoff |
| TRADE_RADIUS | 700 | units for trade-proximity feature |
| TELEPORT_JUMP | 2000 | per-sample displacement treated as artifact |
| CLUSTER_FEATURES | 19 names | the behavioural clustering set (see docs/ARCHITECTURE) |
| SOM_K_RANGE | 4–7 | candidate role counts per side |
| SOM_ITERATIONS | 100,000 | SOM training iterations (converged for 22×22 grid) |
| SOM_SIGMA / SOM_LR | 1.5 / 0.5 | SOM neighbourhood / learning rate |
| STABILITY_RUNS | 8 | retrainings for the global ARI check |
| RANDOM_SEED | 42 | global determinism seed |
| MIN_PRIOR_ROUNDS | 3 | rolling consistency defined only after ≥3 priors |
| ECO_MAX / FORCE_MAX | 2000 / 3900 | buy-type thresholds ($, team average) |

Also: weapon-name normalisation (`norm_weapon`, SNIPERS/RIFLES/GRENADES
sets, defensive against `"AK-47"` / `"ak47"` / `"weapon_ak47"` variants) and
`buy_type()`; `ensure_dirs()` creates the Drive tree.

## manifest.py

* `load() / save(df)` — manifest I/O.
* `sync_with_raw()` — adds a row per new `.dem`; parses the
  `event__stage__match` filename convention to auto-fill `event`; dedupes
  `demo_id` collisions with a `--k` suffix; strips extraction dedupe
  suffixes when guessing `map_name` (so `mirage-2` → mirage but `dust2`
  survives).
* `pending(status)` — rows awaiting a stage.
* `set_status(demo_id, status, notes)` — stage bookkeeping.
* `reset_failed()` — flips all `failed` rows back to `new` for retry.

## parsing.py

* `parse_demo(dem_path, demo_id) -> dict[str, DataFrame]` — full parse:
  reconstructs official rounds from `round_freeze_end`/`round_end`
  (warm-up filtered via `is_warmup_period`; winner map {2: T, 3: CT};
  round-length sanity 4–400 s), builds the explicit 2 Hz tick list per round,
  pulls events, assigns `round_num` by tick intervals, casts steamids to
  string.
* `_safe_event` — wraps `parser.parse_event`, **coercing demoparser2's
  empty-list returns to empty DataFrames** (0.41.x returns `[]` when a demo
  has zero events of a type; this crashed unpatched code).
* `parse_and_save(demo_id, filename)` — Drive→local copy, parse, write
  `parsed/<demo_id>/*.parquet`, update manifest, clean up; exceptions land in
  the manifest notes.
* `parse_all_pending()` — resumable loop over `status == new`.
* `load_parsed(demo_id)` — read a demo's parquet dict back.

## qc.py

* `check_demo(demo_id) -> dict` — flags: round count 13–60 (raised from 40
  after a legitimate 42-round triple-OT final), 10 players/round, both sides
  present, 2 Hz coverage > 0.7, side-win tally. **Flags, does not gate**:
  a flagged demo is inspected manually and replaced or retained.
* `qc_report(demo_ids)` — batch → `analysis/qc_report.csv`.

## features.py

* `build_demo_features(demo_id) -> DataFrame` — one row per
  (round, player): all 19 clustering features, excluded-by-design columns
  (`planted_bomb`, `defused_bomb`, `enemies_flashed`), context (`ctx_*`,
  `buy_type`, `team_avg_equip`). Notable mechanics: nearest-teammate /
  centroid distances over alive samples; path length with teleport
  filtering; opening-window displacement; zone entropy over
  `last_place_name`; time-to-first-contact from `player_hurt`;
  trade-proximity via nearest 2 Hz sample to each teammate-death position;
  utility attribution from `weapon_fire` (throw = behaviour, not detonation).
* `build_all(force=False)` — cached per demo in `features/<demo_id>.parquet`,
  concatenated to `player_round_features.parquet`; flips manifest to
  `featured`.

## roles.py

* `som_grid_size(n)` — Vesanto heuristic ≈ 5√N units (22×22 at N = 8,260).
* `fit_som(X, seed)` — PCA init + `train_random` for SOM_ITERATIONS.
* `som_labels(som, X, k)` — k-means on the codebook (two-stage clustering,
  Vesanto & Alhoniemi 2000); samples inherit their BMU's cluster.
* `select_k(X)` — silhouette (4,000-row subsample) + min/max cluster share
  per candidate k.
* `stability_check(X, k, n_runs, bootstrap)` — pairwise ARI across seeded
  (+bootstrap) retrainings, always labelling the full dataset.
* `kmeans_baseline(X, k, som_labels)` — ARI vs plain k-means on the data.
* `profile_clusters(...)` — z-scored feature means per cluster (the table
  used for manual naming).
* `discover_roles(features, k_by_side, save=True)` — per-side scaler + SOM
  (inf-guarded input), labels, profiles, heuristic name suggestions; saves
  models/profiles/labeled parquet when `save`.
* `apply_manual_names(labeled, names_by_side)` — the human-in-the-loop step;
  persists `role_names_*.json` and the labeled parquet.

## consistency.py

* `_half_id(df)` — contiguous side blocks per (demo, player) → `half_key`.
* `match_level(labeled)` — per player-half modal_share, entropy_norm
  (normalised by log₂ k of that side), switch_rate →
  `consistency_match_level.parquet`.
* `rolling(labeled, save=True)` — strictly prospective per-round variant
  (priors only, ≥ MIN_PRIOR_ROUNDS) → `consistency_rolling.parquet`.
* `team_round_table(rolled, save=True)` — team mean/min per (demo, round,
  side) → `team_round_consistency.parquet`.

## outcome.py

* `build_round_dataset(team_round, save=True)` — T/CT pivot per round +
  winner + controls (equip_diff_k, score_diff from the current-T
  perspective, round_num, pistol flag, map) →
  `round_outcome_dataset.parquet`.
* `fit_main_logit(df, cons_var="cons_diff")` — logit with match-clustered
  SEs; **degenerate covariates auto-dropped** (single-map subsets lose the
  map FE; the pistol flag drops because pistol rounds can never have ≥3
  priors and so never enter the sample).
* `quartile_table(df)` — descriptive gradient → CSV.
* `half_level_ols(match_level, labeled)` — contemporaneous association
  (win share ~ team modal share + side; clustered SEs).
* `permutation_test(labeled, n_perm, cons_var)` — shuffles `role_id` among
  the five teammates within each (demo, round, side); re-estimates the
  target coefficient per replicate; two-sided exceedance p. All internal
  builds use `save=False`. (Minimum attainable p = 1/n_perm.)

## viz.py

`round_positions` (smoke-test traces), `umatrix`, `role_radar` (top-10
discriminative features; call with FINAL names after manual naming — the
locked notebook 03 does this), `quartile_bar`, `coef_plot`,
`consistency_hist`. All save 200 dpi PNGs to `figures/`.

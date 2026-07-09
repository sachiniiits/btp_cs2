# Architecture and Data

## 1. Pipeline architecture

```
 raw_dems/*.dem ──▶ [manifest] ──▶ [parsing] ──▶ parsed/<demo>/*.parquet
                                       │
                                       ▼
                                  [qc checks] ──▶ analysis/qc_report.csv
                                       │
                                       ▼
                               [features] ──▶ features/player_round_features.parquet
                                       │
                                       ▼
                    [roles: per-side SOM + k-means + manual naming]
                          │                       │
                          ▼                       ▼
     features/player_round_features_roles.parquet   models/{som,profiles,names}
                          │
                          ▼
        [consistency: match-level + prospective rolling + team-round]
                          │
                          ▼
        [outcome: round logit · quartiles · half-level OLS · robustness]
                          │
                          ▼
                 analysis/*  +  figures/*
```

Design invariants:

* **Manifest-driven and resumable.** Every stage reads/writes status through
  `manifest.csv`; interrupted runs resume where they stopped.
* **One demo in memory at a time.** Parsing copies each `.dem` from Drive to
  Colab-local disk, parses, writes small parquet outputs back, deletes the
  local copy. Colab RAM is never a constraint.
* **Deterministic.** `RANDOM_SEED = 42` seeds the SOM, k-means, k-selection
  subsampling, stability and permutation machinery. Same config + same
  features ⇒ identical taxonomy and identical downstream numbers.
* **Canonical artifacts are protected.** Robustness variants call every
  builder with `save=False`; only notebooks 01–05 write the canonical files.

## 2. File schemas

### 2.1 `manifest.csv`

| column | meaning |
|--------|---------|
| demo_id | unique id = filename stem (collision-deduped with `--2` suffix) |
| filename | file in `raw_dems/` |
| event | auto-filled from the `event__stage__` filename prefix |
| date | optional, user-filled |
| team1, team2, map_name | best-effort parse of `team1-vs-team2-mN-map` |
| status | `new → parsed → featured` (or `failed` with the error in notes) |
| notes | round count on success; exception text on failure |

Naming convention (produced by the extraction script):
`<event-slug>__<stage-slug>__<team1>-vs-<team2>-mN-<map>.dem` — makes files
self-describing and collision-proof across events.

### 2.2 `parsed/<demo_id>/` (one folder per demo)

| file | grain | key columns |
|------|-------|-------------|
| rounds.parquet | round | round_num, start_tick (freeze end), end_tick, winner_side (T/CT), reason, round_len_s |
| ticks.parquet | player × 2 Hz sample | tick, steamid, name, X, Y, Z, health, is_alive, active_weapon_name, current_equip_value, last_place_name, team_num (2=T, 3=CT), team_clan_name, round_num |
| kills.parquet | kill event | tick, attacker_/user_ steamid+name, weapon, victim position (user_X/Y/Z), round_num |
| hurts.parquet | damage event | tick, attacker_/user_ steamid, dmg_health, weapon, round_num |
| fires.parquet | weapon_fire event | tick, user_steamid, weapon, round_num |
| blinds.parquet | player_blind event | tick, attacker_/user_ steamid, blind_duration, round_num |
| bomb.parquet | bomb event | tick, user_steamid, bomb_event ∈ {planted, defused, exploded}, round_num |
| header.parquet | 1 row | demo header key–values |

All events carry `round_num` assigned by tick-interval lookup; warm-up rows
are filtered via `is_warmup_period` and by falling outside official rounds.

### 2.3 `features/player_round_features.parquet` (+ `_roles` variant)

One row per (demo, round, player). Identity/context columns:
`demo_id, round_num, steamid, side, team_clan, player_name, round_len_s,
team_avg_equip, buy_type (eco/force/full)`.

The **19 clustering features** (definitions in the thesis, Appendix A):
positioning — `dist_nearest_teammate, dist_team_centroid, distance_traveled,
forward_disp_20s, zone_entropy, site_time_share`; timing —
`time_to_first_contact_s, opening_duel_involved`; utility —
`flashes_thrown, smokes_thrown, mollies_thrown, he_thrown, util_early_share`;
weapon style — `awp_share, rifle_share, shots_fired, trade_presence,
time_alive_share`; economy — `buy_value_rel_team`.

Stored but **excluded from clustering** (by design; see decision log):
`planted_bomb, defused_bomb, enemies_flashed`, and the performance context
`ctx_kills, ctx_deaths, ctx_damage, ctx_won_round`.

The `_roles` variant adds `role_id` (int, per side) and `role_name`
(`"T:Aggressor"`, `"CT:Site Anchor"`, …).

### 2.4 `models/`

| file | contents |
|------|----------|
| som_{T,CT}.joblib | dict: fitted StandardScaler, SOM weight tensor (22×22×19), node→cluster map, k |
| role_profile_{T,CT}.csv | 19 × k table of z-scored feature means per role |
| role_names_{T,CT}.json | final human-assigned names, `{role_id: name}` |

### 2.5 `analysis/`

| file | producer | grain / contents |
|------|----------|------------------|
| qc_report.csv | nb 00/01 | per-demo QC flags (round count, 10 players, side presence, tick coverage, side-win tally) |
| consistency_match_level.parquet | nb 04 | per player-half: half_key, n_rounds, modal_share, entropy_norm, switch_rate |
| consistency_rolling.parquet | nb 04 | player-round grain: all feature/role columns + half_key, roll_modal_share, roll_entropy, roll_switch_rate (NaN until ≥3 priors in the half) |
| team_round_consistency.parquet | nb 04 | per (demo, round, side): cons_mean, cons_min, entropy_mean, switch_mean, team_avg_equip, n_players |
| round_outcome_dataset.parquet | nb 05 | per round: T/CT pivots of the above, winner_side, t_win, cons_diff, cons_min_diff, entropy_diff, equip_diff_k, pistol, score_diff, map_name |
| winrate_by_consistency_quartile.csv | nb 05 | 4 rows: quartile, t_win_rate, n |
| stability_{T,CT}.csv | nb 03 | pairwise ARI across 8 retrainings |
| per_cluster_stability_{T,CT}.json | nb 03 | best-match Jaccard per named role (6 retrainings) |
| role_buytype_crosstab_{T,CT}.csv | nb 03 | role × buy-type row-normalised shares |
| permutation_test.json | nb 06 | cons_diff permutation summary (real β, permuted mean/sd, p, n_valid) |
| robustness_cons_min.json | nb 06 | same for the weakest-link measure |

### 2.6 `figures/` (200 dpi PNG)

`smoke_round_*_traces` (parser validation) · `som_umatrix_{T,CT}` ·
`role_radars_{T,CT}` (final names) · `consistency_distribution` ·
`logit_coefficients` · `winrate_by_consistency_quartile`.

## 3. Key derived definitions

* **Side / half.** `team_num` 2 = T, 3 = CT. A *player-half* (`half_key`) is
  a contiguous block of rounds one player spends on one side within one match
  (handles overtime side swaps): `demo_id|steamid|h<k>`.
* **Buy type.** Team average freeze-end equipment: eco < $2,000,
  force < $3,900, full otherwise.
* **Rolling consistency (leakage-free).** For round *t*: modal share /
  entropy / switch rate over the half's rounds strictly before *t*;
  undefined until ≥ 3 priors. Consequently rounds 1–3 of every half (incl.
  both pistols) never enter the regression sample.
* **Outcome model perspective.** One row per round from the T side's view;
  `t_win = 1` iff the T side won; `*_diff = T − CT`; SEs clustered by
  `demo_id`.

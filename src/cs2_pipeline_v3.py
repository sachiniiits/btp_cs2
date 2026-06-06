"""
CS2 .DEM → BEHAVIORALLY CLUSTERED PLAYER ROLES PIPELINE (v3 — Fixed)
Google Colab Compatible | Python 3.10+ | demoparser2

FIXES vs v2:
  FIX 1 — trade_kills now capped at CAP_TRADE_KILLS (was reaching 17)
  FIX 2 — zone_primary now uses global percentiles on all_x to classify
           mean_x; no longer always returns 2
  FIX 3 — flash_assists now reads assister_name + assistedflash columns
           correctly (was always 0)
  FIX 4 — Added "Rifler" role so Cluster 0 (generic rifler) has a proper
           home instead of being force-assigned AWPer
  FIX 5 — k-selection chart now saved (g2_k_selection.png was missing)
  FIX 6 — distance/isolation collinearity: both kept but noted; distance
           log-scaled weight reduced in Lurker scoring
"""

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────
import subprocess, sys, os, glob, warnings, traceback, time
from pathlib import Path
from typing import Optional
from collections import Counter

def pip_install(pkg):
    subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], check=False)

for pkg in ["demoparser2", "tqdm", "hdbscan"]:
    pip_install(pkg)

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from tqdm import tqdm

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    warnings.warn("hdbscan not available — will use KMeans+GMM only.")

try:
    from demoparser2 import DemoParser
    PARSER_AVAILABLE = True
except ImportError:
    PARSER_AVAILABLE = False
    warnings.warn("demoparser2 not available.")

warnings.filterwarnings("ignore")

try:
    from google.colab import drive
    try:
        drive.flush_and_unmount()
    except:
        pass
    drive.mount('/content/drive', force_remount=True)
    time.sleep(3)
    print("✅ Google Drive mounted.")
except ImportError:
    print("ℹ Not running on Colab — assuming Drive accessible.")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
_BASE       = "/content/drive/MyDrive/cs2-btp"
DEM_FOLDER  = os.path.join(_BASE, "raw_dems")

if not os.path.isdir(DEM_FOLDER):
    raise FileNotFoundError(f"❌ DEM_FOLDER not found: {DEM_FOLDER}")

dem_check = glob.glob(os.path.join(DEM_FOLDER, "*.dem"))
if not dem_check:
    raise FileNotFoundError(f"❌ No .dem files in '{DEM_FOLDER}'")

print(f"✅ DEM_FOLDER: {DEM_FOLDER} ({len(dem_check)} .dem files)")

OUTPUT_DIR = Path(os.path.join(_BASE, "processed"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED       = 42
MIN_TICK          = 0
POS_DOWNSAMPLE_N  = 64
CAP_DAMAGE        = 500
CAP_KILLS         = 5
CAP_UTILITY       = 5
CAP_TRADE_KILLS   = 5   # FIX 1: explicit cap constant (was missing — trade_kills reached 17)
CAP_DISTANCE      = 5000
CAP_ISOLATION     = 2000
TRADE_WINDOW_TICKS = 320

print(f"✅ Config loaded | HDBSCAN: {'available' if HDBSCAN_AVAILABLE else 'NOT available'}")

# ─────────────────────────────────────────────────────────────────────────────
# DEM PARSING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
TEAM_MAP = {"TERRORIST": "T", "COUNTER_TERRORIST": "CT", "T": "T", "CT": "CT"}

def _safe_parse_event(parser, event, cols):
    try:
        df = parser.parse_event(event, player=cols)
        if df is None or df.empty:
            return pd.DataFrame()
        for col in ["team_name", "attacker_team_name", "user_team_name"]:
            if col in df.columns:
                df[col] = df[col].map(lambda x: TEAM_MAP.get(str(x).upper(), str(x)))
        return df
    except Exception:
        return pd.DataFrame()

def _safe_parse_ticks(parser, props, ticks):
    try:
        df = parser.parse_ticks(props, ticks=ticks)
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def parse_dem_file(dem_path):
    if not PARSER_AVAILABLE:
        return {k: pd.DataFrame() for k in ["kills", "damage", "positions", "grenades", "round_ends"]}
    parser = DemoParser(dem_path)

    # player_death: include assistedflash + assister_name for FIX 3
    kills = _safe_parse_event(parser, "player_death",
                              ["X", "Y", "Z", "team_name", "flash_duration",
                               "headshot", "assistedflash", "assister_name"])
    if not kills.empty:
        keep = [c for c in ["tick", "attacker_name", "attacker_team_name",
                             "user_name", "user_team_name", "headshot", "weapon",
                             "X", "Y", "Z", "flash_duration", "assistedflash",
                             "assister_name"] if c in kills.columns]
        kills = kills[keep].copy()

    damage = _safe_parse_event(parser, "player_hurt", ["X", "Y", "team_name"])
    if not damage.empty:
        keep = [c for c in ["tick", "attacker_name", "dmg_health", "dmg_armor",
                             "attacker_team_name"] if c in damage.columns]
        damage = damage[keep].copy()

    try:
        header   = parser.parse_header()
        max_tick = int(header.get("playback_ticks", 150_000))
    except Exception:
        max_tick = 150_000

    sample_ticks = list(range(MIN_TICK + 1, max_tick, POS_DOWNSAMPLE_N))
    positions    = _safe_parse_ticks(parser, ["X", "Y", "Z", "team_name"], sample_ticks)

    grenades = pd.DataFrame()
    for ev in ["weapon_fire", "inferno_startburn", "smokegrenade_detonate",
               "hegrenade_detonate", "flashbang_detonate"]:
        g = _safe_parse_event(parser, ev, ["team_name"])
        if not g.empty:
            grenades = pd.concat([grenades, g], ignore_index=True)

    round_ends = _safe_parse_event(parser, "round_end", [])
    return dict(kills=kills, damage=damage, positions=positions,
                grenades=grenades, round_ends=round_ends)

print("✅ Parser utilities defined.")

# ─────────────────────────────────────────────────────────────────────────────
# ROUND RECONSTRUCTION & FEATURE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def reconstruct_rounds(round_ends_df, max_fallback_tick=150_000):
    if round_ends_df.empty or "tick" not in round_ends_df.columns:
        return pd.DataFrame({"round_end_tick": [max_fallback_tick], "round_num": [1]})
    ticks = sorted(round_ends_df["tick"].dropna().unique().tolist())
    return pd.DataFrame({"round_end_tick": ticks, "round_num": range(1, len(ticks) + 1)})

def assign_round_num(tick_series, round_map):
    if round_map.empty:
        return pd.Series(1, index=tick_series.index)
    bins   = [0] + round_map["round_end_tick"].tolist()
    labels = round_map["round_num"].tolist()
    if len(labels) != len(bins) - 1:
        return pd.Series(1, index=tick_series.index)
    try:
        assigned = pd.cut(tick_series, bins=bins, labels=labels, right=True)
        return pd.to_numeric(assigned, errors="coerce").fillna(1).astype(int)
    except Exception:
        return pd.Series(1, index=tick_series.index)

# FIX 2: zone_primary — classify mean_x against global all_x percentiles
# Old code passed [mean_x] (1-element list) into compute_zone_adaptive which
# hit the len < 10 guard and always returned zone 2.
def compute_zone_primary(mean_x, all_x):
    """
    Classify a player's mean X position into map zones using the global
    X distribution (all ticks, all players, whole match set) as the reference.
    Returns 1 (CT/right side), 2 (mid), or 3 (T/left side).
    """
    if pd.isna(mean_x) or len(all_x) < 10:
        return 2
    p33 = np.nanpercentile(all_x, 33)
    p66 = np.nanpercentile(all_x, 66)
    if mean_x < p33:
        return 3
    elif mean_x > p66:
        return 1
    else:
        return 2

def mean_isolation(positions_df, player_name, round_num):
    rnd_pos = positions_df[positions_df["round_num"] == round_num]
    if rnd_pos.empty or "X" not in rnd_pos.columns:
        return 0.0
    name_col = "name" if "name" in rnd_pos.columns else "player_name"
    if name_col not in rnd_pos.columns:
        return 0.0
    own = rnd_pos[rnd_pos[name_col] == player_name]
    if own.empty or len(own) < 2:
        return 0.0
    own_xy   = own[["X", "Y"]].dropna().values
    own_team = own["team_name"].iloc[0] if "team_name" in own.columns else None
    mates    = (rnd_pos[(rnd_pos.get("team_name", pd.Series()) == own_team)
                        & (rnd_pos[name_col] != player_name)]
                if own_team else rnd_pos)
    if len(mates) < 2:
        return 0.0
    mate_xy = mates[["X", "Y"]].dropna().values
    if len(mate_xy) < 1:
        return 0.0
    try:
        tree  = cKDTree(mate_xy)
        dists, _ = tree.query(own_xy, k=1)
        return float(np.mean(dists))
    except Exception:
        return 0.0

def distance_traveled(pos_rows):
    xy = pos_rows[["X", "Y"]].dropna().values
    if len(xy) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))

# FIX 1: cap trade_kills at CAP_TRADE_KILLS before returning
def compute_trade_kills(kills_df, player_name, round_num, side):
    if kills_df.empty or "tick" not in kills_df.columns:
        return 0
    rnd_kills = kills_df[kills_df["round_num"] == round_num]
    if rnd_kills.empty:
        return 0
    team_col = "user_team_name" if "user_team_name" in rnd_kills.columns else None
    if team_col is None:
        return 0
    mate_deaths = rnd_kills[
        (rnd_kills[team_col] == side) &
        (rnd_kills.get("user_name", pd.Series()) != player_name)
    ]
    player_kills = rnd_kills[rnd_kills.get("attacker_name", pd.Series()) == player_name]
    if mate_deaths.empty or player_kills.empty:
        return 0
    trades = 0
    for _, pk in player_kills.iterrows():
        for _, md in mate_deaths.iterrows():
            if 0 < (pk["tick"] - md["tick"]) <= TRADE_WINDOW_TICKS:
                trades += 1
                break
    return min(trades, CAP_TRADE_KILLS)   # FIX 1: cap applied here

# FIX 3: flash_assists — read assister_name + assistedflash (not flash_duration)
def compute_flash_assists(kills_df, player_name, round_num):
    """
    Count rounds where this player got a flash assist:
    i.e. kills_df rows where assister_name == player_name AND assistedflash == True.
    Old code filtered by attacker_name + flash_duration > 0, which is always 0.
    """
    if kills_df.empty:
        return 0
    rnd = kills_df[kills_df["round_num"] == round_num]
    if rnd.empty:
        return 0
    if "assister_name" in rnd.columns and "assistedflash" in rnd.columns:
        fa = rnd[
            (rnd["assister_name"] == player_name) &
            (rnd["assistedflash"].fillna(False).astype(bool))
        ]
        return min(len(fa), CAP_UTILITY)
    # Fallback: if columns absent (older demo format) try flash_duration on kills
    if "flash_duration" in rnd.columns and "attacker_name" in rnd.columns:
        fa = rnd[
            (rnd["attacker_name"] == player_name) &
            (rnd["flash_duration"].fillna(0) > 0)
        ]
        return min(len(fa), CAP_UTILITY)
    return 0

def compute_opening_duel(kills_df, player_name, round_num):
    if kills_df.empty or "tick" not in kills_df.columns:
        return 0
    rnd = kills_df[kills_df["round_num"] == round_num].sort_values("tick")
    if rnd.empty:
        return 0
    first = rnd.iloc[0]
    if first.get("attacker_name") == player_name:
        return 1
    if first.get("user_name") == player_name:
        return -1
    return 0

print("✅ Feature helpers defined.")

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def _player_col(df):
    for c in ["player_name", "name", "attacker_name", "user_name"]:
        if c in df.columns:
            return c
    return None

def extract_features_from_dem(dem_path, match_id):
    try:
        events = parse_dem_file(dem_path)
    except Exception as e:
        print(f"  ⚠ Parser failed on {Path(dem_path).name}: {e}")
        return pd.DataFrame()

    kills_df, damage_df = events["kills"], events["damage"]
    pos_df, grenades_df, re_df = events["positions"], events["grenades"], events["round_ends"]
    round_map = reconstruct_rounds(re_df)

    for df in [kills_df, damage_df, pos_df, grenades_df]:
        if not df.empty and "tick" in df.columns:
            df["round_num"] = assign_round_num(df["tick"], round_map)

    # Build global X distribution for zone classification (FIX 2)
    all_x = pos_df["X"].dropna().values if not pos_df.empty and "X" in pos_df.columns else np.array([])

    # Collect (player, round, side) combos
    combos = set()
    for df, hint in [(kills_df, "attacker_name"), (damage_df, "attacker_name")]:
        if df.empty:
            continue
        nc = hint if hint in df.columns else _player_col(df)
        if nc is None:
            continue
        tc = next((c for c in ["attacker_team_name", "team_name"] if c in df.columns), None)
        for _, row in df.iterrows():
            pname = row.get(nc)
            rnd   = row.get("round_num", 1)
            side  = row.get(tc, "T") if tc else "T"
            if pd.notna(pname) and pd.notna(rnd):
                combos.add((str(pname), int(rnd), str(side)))
    if not kills_df.empty and "user_name" in kills_df.columns:
        tc = "user_team_name" if "user_team_name" in kills_df.columns else None
        for _, row in kills_df.iterrows():
            pname = row.get("user_name")
            rnd   = row.get("round_num", 1)
            side  = row.get(tc, "T") if tc else "T"
            if pd.notna(pname) and pd.notna(rnd):
                combos.add((str(pname), int(rnd), str(side)))

    if not combos:
        return pd.DataFrame()

    rows = []
    for player_name, round_num, side in combos:
        # Kills
        kdf = (kills_df[(kills_df["attacker_name"] == player_name) &
                        (kills_df["round_num"] == round_num)]
               if not kills_df.empty and "attacker_name" in kills_df.columns
               else pd.DataFrame())
        kill_count   = min(len(kdf), CAP_KILLS)
        headshot_pct = (float(kdf["headshot"].sum() / kill_count)
                        if not kdf.empty and "headshot" in kdf.columns and kill_count > 0
                        else 0.0)

        # Damage
        ddf = (damage_df[(damage_df["attacker_name"] == player_name) &
                         (damage_df["round_num"] == round_num)]
               if not damage_df.empty and "attacker_name" in damage_df.columns
               else pd.DataFrame())
        damage_dealt = (float(min(ddf["dmg_health"].sum(), CAP_DAMAGE))
                        if "dmg_health" in ddf.columns and not ddf.empty
                        else float(kill_count * 80))

        # Utility
        utility_used = 0
        if not grenades_df.empty:
            nc = _player_col(grenades_df)
            if nc:
                gdf = grenades_df[(grenades_df[nc] == player_name) &
                                  (grenades_df["round_num"] == round_num)]
                utility_used = min(len(gdf), CAP_UTILITY)

        # Positions
        player_pos = pd.DataFrame()
        if not pos_df.empty:
            nc = _player_col(pos_df)
            if nc and "round_num" in pos_df.columns:
                player_pos = pos_df[(pos_df[nc] == player_name) &
                                    (pos_df["round_num"] == round_num)]

        dist    = min(distance_traveled(player_pos), CAP_DISTANCE) if not player_pos.empty else 0.0
        iso     = min(mean_isolation(pos_df, player_name, round_num), CAP_ISOLATION)
        mean_x  = player_pos["X"].mean() if not player_pos.empty and "X" in player_pos.columns else np.nan

        # FIX 2: use global all_x percentiles to classify zone
        zone_prim = compute_zone_primary(mean_x, all_x)

        # Timing features
        rnd_end_tick_row = round_map[round_map["round_num"] == round_num]["round_end_tick"]
        rnd_end_tick     = rnd_end_tick_row.iloc[0] if not rnd_end_tick_row.empty else 64 * 115
        prev             = round_map[round_map["round_num"] == round_num - 1]["round_end_tick"]
        rnd_start_tick   = prev.iloc[0] if not prev.empty else 0
        rnd_len          = max(rnd_end_tick - rnd_start_tick, 1)

        first_kill_tick = np.nan
        if not kills_df.empty and "attacker_name" in kills_df.columns and "tick" in kills_df.columns:
            kt = kills_df[(kills_df["attacker_name"] == player_name) &
                          (kills_df["round_num"] == round_num)]["tick"]
            if not kt.empty:
                first_kill_tick = kt.min()

        death_tick = np.nan
        if not kills_df.empty and "user_name" in kills_df.columns and "tick" in kills_df.columns:
            dt_ = kills_df[(kills_df["user_name"] == player_name) &
                           (kills_df["round_num"] == round_num)]["tick"]
            if not dt_.empty:
                death_tick = dt_.min()

        fe_rel = (float(np.clip((first_kill_tick - rnd_start_tick) / rnd_len, 0, 1))
                  if not np.isnan(first_kill_tick) else 1.0)
        dt_rel = (float(np.clip((death_tick - rnd_start_tick) / rnd_len, 0, 1))
                  if not np.isnan(death_tick) else 1.0)

        trade_kills     = compute_trade_kills(kills_df, player_name, round_num, side)  # FIX 1
        flash_assists   = compute_flash_assists(kills_df, player_name, round_num)       # FIX 3
        multi_kill_round = 1 if kill_count >= 3 else 0
        survived        = 1 if np.isnan(death_tick) else 0
        opening_duel_won = compute_opening_duel(kills_df, player_name, round_num)

        rows.append({
            "match_id": match_id, "player_name": player_name,
            "round_num": round_num, "side": side,
            "kill_count": kill_count, "damage_dealt": damage_dealt,
            "utility_used": utility_used, "distance_traveled": dist,
            "mean_isolation": iso, "first_engagement_rel": fe_rel,
            "death_tick_rel": dt_rel, "zone_primary": zone_prim,
            "headshot_pct": headshot_pct, "trade_kills": trade_kills,
            "flash_assists": flash_assists, "multi_kill_round": multi_kill_round,
            "survived": survived, "opening_duel_won": opening_duel_won,
        })

    return pd.DataFrame(rows)

# Parse all demos
dem_files = sorted(glob.glob(os.path.join(DEM_FOLDER, "*.dem")))
print(f"\n📂 Found {len(dem_files)} .dem file(s):")
for f in dem_files:
    print(f"    • {Path(f).name} ({os.path.getsize(f)/(1024*1024):.1f} MB)")

all_frames = []
for dem_path in tqdm(dem_files, desc="Parsing demos"):
    match_id = Path(dem_path).stem
    df = extract_features_from_dem(dem_path, match_id)
    if not df.empty:
        all_frames.append(df)
        print(f"    ✅ {Path(dem_path).name}: {len(df)} player-round rows")
    else:
        print(f"    ⚠ {Path(dem_path).name}: no data extracted")

if not all_frames:
    raise RuntimeError(f"❌ All {len(dem_files)} demos failed to parse.")

features_df = pd.concat(all_frames, ignore_index=True)
print(f"\n📊 Raw feature shape: {features_df.shape}")
features_df.to_parquet(OUTPUT_DIR / "g1_features.parquet", index=False)
print(f"💾 Saved → {OUTPUT_DIR / 'g1_features.parquet'}")

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING & SCALING
# ─────────────────────────────────────────────────────────────────────────────
LOG_FEATURES    = ["damage_dealt", "distance_traveled", "mean_isolation"]
COUNT_FEATURES  = ["kill_count", "utility_used", "trade_kills", "flash_assists"]
BINARY_FEATURES = ["multi_kill_round", "survived"]
PASSTHROUGH     = ["first_engagement_rel", "death_tick_rel", "zone_primary",
                   "headshot_pct", "opening_duel_won"]

def preprocess(df):
    df = df.copy()
    df["damage_dealt"]         = df["damage_dealt"].clip(0, CAP_DAMAGE)
    df["kill_count"]           = df["kill_count"].clip(0, CAP_KILLS)
    df["utility_used"]         = df["utility_used"].clip(0, CAP_UTILITY)
    df["distance_traveled"]    = df["distance_traveled"].clip(0, CAP_DISTANCE)
    df["mean_isolation"]       = df["mean_isolation"].clip(0, CAP_ISOLATION)
    df["trade_kills"]          = df["trade_kills"].clip(0, CAP_TRADE_KILLS)  # FIX 1
    df["flash_assists"]        = df["flash_assists"].clip(0, CAP_UTILITY)
    df["first_engagement_rel"] = df["first_engagement_rel"].clip(0, 1)
    df["death_tick_rel"]       = df["death_tick_rel"].clip(0, 1)
    df["headshot_pct"]         = df["headshot_pct"].clip(0, 1)
    df["zone_primary"]         = df["zone_primary"].astype(int).clip(1, 3)
    df["opening_duel_won"]     = df["opening_duel_won"].clip(-1, 1)

    for col in LOG_FEATURES:
        if col in df.columns:
            df[f"{col}_log"] = np.log1p(df[col])

    to_scale = (
        [f"{c}_log" for c in LOG_FEATURES if f"{c}_log" in df.columns] +
        [c for c in COUNT_FEATURES  if c in df.columns] +
        [c for c in BINARY_FEATURES if c in df.columns]
    )
    scaler       = StandardScaler()
    scaled_vals  = scaler.fit_transform(df[to_scale].fillna(0))
    scaled_cols  = [f"{c}_scaled" for c in to_scale]
    df[scaled_cols] = scaled_vals

    for col in PASSTHROUGH:
        if col in df.columns:
            df[f"{col}_scaled"] = df[col].astype(float)
    return df

clean_df = preprocess(features_df)

print("\n" + "═"*60)
print("  G1 VALIDATION REPORT")
print("═"*60)
raw_check = {
    "damage_dealt":        (0, CAP_DAMAGE),
    "kill_count":          (0, CAP_KILLS),
    "utility_used":        (0, CAP_UTILITY),
    "distance_traveled":   (0, CAP_DISTANCE),
    "mean_isolation":      (0, CAP_ISOLATION),
    "first_engagement_rel":(0, 1),
    "death_tick_rel":      (0, 1),
    "zone_primary":        (1, 3),
    "headshot_pct":        (0, 1),
    "trade_kills":         (0, CAP_TRADE_KILLS),
    "flash_assists":       (0, CAP_UTILITY),
    "multi_kill_round":    (0, 1),
    "survived":            (0, 1),
    "opening_duel_won":    (-1, 1),
}
all_valid = True
for col, (lo, hi) in raw_check.items():
    if col not in clean_df.columns:
        print(f"  {col:<28} MISSING ⚠"); all_valid = False; continue
    mn, mx = clean_df[col].min(), clean_df[col].max()
    ok = "✅" if (mn >= lo - 1e-6 and mx <= hi + 1e-6) else "❌"
    if ok == "❌": all_valid = False
    print(f"  {col:<28} [{mn:7.2f}, {mx:7.2f}]  expected [{lo}, {hi}]  {ok}")

print("\n  Scaled columns (should be ~N(0,1)):")
scaled_summary = clean_df[[c for c in clean_df.columns if c.endswith("_scaled")]].describe().loc[["mean","std"]]
print(scaled_summary.round(3).to_string())
print(f"\n  zone_primary distribution:\n{clean_df['zone_primary'].value_counts().sort_index().to_string()}")
print(f"  flash_assists > 0: {(clean_df['flash_assists'] > 0).sum()} rows")
if all_valid:
    print("\n  ✅ ALL FEATURES VALID")
else:
    print("\n  ⚠ SOME FEATURES OUT OF RANGE — check above")

clean_df.to_parquet(OUTPUT_DIR / "g1_clean.parquet", index=False)
print(f"\n💾 Saved → {OUTPUT_DIR / 'g1_clean.parquet'}")

# ─────────────────────────────────────────────────────────────────────────────
# CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
SCALED_FEATURE_COLS = (
    [f"{c}_log_scaled" for c in LOG_FEATURES] +
    [f"{c}_scaled" for c in COUNT_FEATURES] +
    [f"{c}_scaled" for c in BINARY_FEATURES] +
    [f"{c}_scaled" for c in PASSTHROUGH]
)

g1_clean = pd.read_parquet(OUTPUT_DIR / "g1_clean.parquet")
print(f"\n📂 Loaded g1_clean.parquet — {g1_clean.shape[0]:,} rows")

missing = [c for c in SCALED_FEATURE_COLS if c not in g1_clean.columns]
if missing:
    available_scaled = [c for c in g1_clean.columns if "_scaled" in c]
    SCALED_FEATURE_COLS = [c for c in SCALED_FEATURE_COLS if c in g1_clean.columns] or available_scaled
    print(f"  ↪ Using {len(SCALED_FEATURE_COLS)} available scaled cols")
else:
    print(f"✅ All {len(SCALED_FEATURE_COLS)} scaled feature columns present.")

X_scaled = g1_clean[SCALED_FEATURE_COLS].fillna(0).values

# Correlation matrix
corr = g1_clean[SCALED_FEATURE_COLS].corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "g2_correlation_matrix.png", dpi=150)
plt.show()
print("💾 Saved → g2_correlation_matrix.png")

# K selection — FIX 5: save the chart (was listed in outputs but never saved)
K_RANGE = range(3, 8)
sil_scores, bic_scores = {}, {}
print("\n🔍 Evaluating k values via Silhouette + BIC…")
for k in tqdm(K_RANGE, desc="K evaluation"):
    km  = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED)
    lbl = km.fit_predict(X_scaled)
    sil_scores[k] = (silhouette_score(X_scaled, lbl, sample_size=min(5000, len(lbl)))
                     if len(set(lbl)) > 1 else -1.0)
    gmm = GaussianMixture(n_components=k, covariance_type="full",
                          n_init=3, random_state=RANDOM_SEED)
    gmm.fit(X_scaled)
    bic_scores[k] = gmm.bic(X_scaled)

OPTIMAL_K_SIL = max(sil_scores, key=sil_scores.get)
OPTIMAL_K_BIC = min(bic_scores, key=bic_scores.get)
OPTIMAL_K     = (OPTIMAL_K_SIL if abs(OPTIMAL_K_SIL - OPTIMAL_K_BIC) <= 1
                 else max(3, min(7, round((OPTIMAL_K_SIL + OPTIMAL_K_BIC) / 2))))
print(f"✨ Optimal K: {OPTIMAL_K}  (Silhouette→{OPTIMAL_K_SIL}, BIC→{OPTIMAL_K_BIC})")

# FIX 5: save the k-selection chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ks = list(K_RANGE)
ax1.plot(ks, [sil_scores[k] for k in ks], "o-", color="#457B9D")
ax1.axvline(OPTIMAL_K_SIL, color="#E63946", linestyle="--", label=f"Best k={OPTIMAL_K_SIL}")
ax1.set(xlabel="k", ylabel="Silhouette score", title="Silhouette vs k")
ax1.legend()
ax2.plot(ks, [bic_scores[k] for k in ks], "o-", color="#2A9D8F")
ax2.axvline(OPTIMAL_K_BIC, color="#E63946", linestyle="--", label=f"Best k={OPTIMAL_K_BIC}")
ax2.set(xlabel="k", ylabel="BIC", title="GMM BIC vs k")
ax2.legend()
plt.suptitle(f"K Selection  |  Chosen K = {OPTIMAL_K}", fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "g2_k_selection.png", dpi=150)
plt.show()
print("💾 Saved → g2_k_selection.png")

# Ensemble clustering
km_model  = KMeans(n_clusters=OPTIMAL_K, n_init=10, random_state=RANDOM_SEED)
km_labels = km_model.fit_predict(X_scaled)

gmm_model  = GaussianMixture(n_components=OPTIMAL_K, covariance_type="full",
                              n_init=5, random_state=RANDOM_SEED)
gmm_labels = gmm_model.fit_predict(X_scaled)
gmm_probs  = gmm_model.predict_proba(X_scaled)

if HDBSCAN_AVAILABLE:
    hdb_model      = hdbscan.HDBSCAN(
        min_cluster_size=max(20, len(X_scaled) // 100),  # lowered from 50 for better granularity
        min_samples=5,
        metric="euclidean"
    )
    hdb_labels_raw = hdb_model.fit_predict(X_scaled)
    unique_hdb     = sorted(set(hdb_labels_raw) - {-1})
    hdb_remap      = {old: new for new, old in enumerate(unique_hdb)}
    hdb_remap[-1]  = -1
    hdb_labels     = np.array([hdb_remap[l] for l in hdb_labels_raw])
else:
    hdb_labels = km_labels.copy()

def ensemble_consensus(km, gmm, hdb, k):
    from scipy.optimize import linear_sum_assignment

    def align_labels(ref, target, k):
        if len(set(target)) <= 1:
            return target
        cost = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                cost[i, j] = -np.sum((ref == i) & (target == j))
        row_ind, col_ind = linear_sum_assignment(cost)
        mapping = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}
        return np.array([mapping.get(l, l % k) for l in target])

    gmm_aligned = align_labels(km, gmm, k)
    hdb_valid   = hdb.copy()
    hdb_valid[hdb_valid == -1] = km[hdb_valid == -1]   # assign noise to KMeans
    hdb_aligned = align_labels(km, hdb_valid, k) if HDBSCAN_AVAILABLE else km.copy()

    consensus = np.zeros(len(km), dtype=int)
    for i in range(len(km)):
        votes = [km[i], gmm_aligned[i], hdb_aligned[i]]
        consensus[i] = Counter(votes).most_common(1)[0][0]
    return consensus, gmm_aligned, hdb_aligned

cluster_labels, gmm_aligned, hdb_aligned = ensemble_consensus(km_labels, gmm_labels, hdb_labels, OPTIMAL_K)

agree_all = np.mean(
    (km_labels == cluster_labels) &
    (gmm_aligned == cluster_labels) &
    (hdb_aligned == cluster_labels)
)
print(f"\n📊 Ensemble Agreement (all 3 agree): {agree_all:.1%}")

g2_df = g1_clean.copy()
g2_df["cluster_id"]    = cluster_labels
g2_df["gmm_max_prob"]  = gmm_probs.max(axis=1)

print("\n  Cluster distribution:")
for cid, cnt in g2_df["cluster_id"].value_counts().sort_index().items():
    print(f"  Cluster {cid}: {cnt} rows ({100*cnt/len(g2_df):.1f}%)")

g2_df.to_parquet(OUTPUT_DIR / "g2_clustered.parquet", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# DIMENSIONALITY REDUCTION & VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────
pca       = PCA(n_components=2, random_state=RANDOM_SEED)
X_pca     = pca.fit_transform(X_scaled)
var_exp   = pca.explained_variance_ratio_
print(f"\n📐 PCA variance: PC1={var_exp[0]:.2%}, PC2={var_exp[1]:.2%}")

N_TSNE   = min(5000, len(X_scaled))
tsne_idx = np.random.default_rng(RANDOM_SEED).choice(len(X_scaled), N_TSNE, replace=False)
tsne     = TSNE(n_components=2, perplexity=30, random_state=RANDOM_SEED, n_iter=1000)
X_tsne   = tsne.fit_transform(X_scaled[tsne_idx])
C_tsne   = cluster_labels[tsne_idx]

N_PLOT  = min(3000, len(X_pca))
idx_smp = np.random.default_rng(RANDOM_SEED).choice(len(X_pca), N_PLOT, replace=False)
X_plot  = X_pca[idx_smp]
C_plot  = cluster_labels[idx_smp]
palette = sns.color_palette("tab10", OPTIMAL_K)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
for cid in range(OPTIMAL_K):
    mask = C_plot == cid
    ax1.scatter(X_plot[mask, 0], X_plot[mask, 1], c=[palette[cid]],
                label=f"Cluster {cid}", alpha=0.5, s=18)
centroids_pca = pca.transform(km_model.cluster_centers_)
for cid, (cx, cy) in enumerate(centroids_pca):
    ax1.scatter(cx, cy, c="black", marker="X", s=180, zorder=10)
    ax1.annotate(f" C{cid}", (cx, cy), fontsize=10, fontweight="bold")
ax1.set(xlabel=f"PC1 ({var_exp[0]:.1%})", ylabel=f"PC2 ({var_exp[1]:.1%})",
        title="PCA Cluster View")
ax1.legend(loc="upper right", framealpha=0.8)
ax1.grid(True, alpha=0.2)

for cid in range(OPTIMAL_K):
    mask = C_tsne == cid
    ax2.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=[palette[cid]],
                label=f"Cluster {cid}", alpha=0.5, s=18)
ax2.set(xlabel="t-SNE 1", ylabel="t-SNE 2", title="t-SNE Cluster View")
ax2.legend(loc="upper right", framealpha=0.8)
ax2.grid(True, alpha=0.2)

plt.suptitle("CS2 Player Roles — Dimensionality Reduction", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "g2_pca_tsne.png", dpi=150, bbox_inches="tight")
plt.show()
print("💾 Saved → g2_pca_tsne.png")

# ─────────────────────────────────────────────────────────────────────────────
# CLUSTER PROFILING
# ─────────────────────────────────────────────────────────────────────────────
ORIGINAL_COLS = ["damage_dealt", "kill_count", "utility_used", "distance_traveled",
                 "mean_isolation", "first_engagement_rel", "death_tick_rel",
                 "zone_primary", "headshot_pct", "trade_kills", "flash_assists",
                 "multi_kill_round", "survived", "opening_duel_won"]
orig_cols_present = [c for c in ORIGINAL_COLS if c in g2_df.columns]
cluster_profile   = g2_df.groupby("cluster_id")[orig_cols_present].mean().round(3)

print("\n" + "═"*70)
print("  G2 CLUSTER PROFILE")
print("═"*70)
print(cluster_profile.to_string())

norm_profile = ((cluster_profile - cluster_profile.min()) /
                (cluster_profile.max() - cluster_profile.min() + 1e-9))

fig, ax = plt.subplots(figsize=(14, max(3, OPTIMAL_K)))
sns.heatmap(norm_profile, annot=cluster_profile.round(2), fmt="g",
            cmap="RdYlGn", linewidths=0.5, ax=ax,
            cbar_kws={"label": "Normalised (0–1)"})
ax.set(title="CS2 Cluster Profiles — Normalised Feature Means",
       xlabel="Feature", ylabel="Cluster ID")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "g2_profile_heatmap.png", dpi=150)
plt.show()
print("💾 Saved → g2_profile_heatmap.png")

# Radar charts
radar_cols    = ["damage_dealt", "kill_count", "utility_used", "distance_traveled",
                 "mean_isolation", "first_engagement_rel", "headshot_pct",
                 "trade_kills", "flash_assists", "survived", "opening_duel_won"]
radar_present = [c for c in radar_cols if c in norm_profile.columns]
if len(radar_present) >= 4:
    angles = np.linspace(0, 2 * np.pi, len(radar_present), endpoint=False).tolist()
    angles += angles[:1]
    fig, axes = plt.subplots(1, OPTIMAL_K, figsize=(5 * OPTIMAL_K, 5),
                             subplot_kw=dict(polar=True))
    if OPTIMAL_K == 1:
        axes = [axes]
    rcolors = sns.color_palette("tab10", OPTIMAL_K)
    for idx, (cid, row) in enumerate(norm_profile[radar_present].iterrows()):
        vals = row.values.tolist() + row.values.tolist()[:1]
        ax   = axes[idx]
        ax.plot(angles, vals, "o-", linewidth=2, color=rcolors[idx])
        ax.fill(angles, vals, alpha=0.25, color=rcolors[idx])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([c.replace("_", "\n") for c in radar_present], fontsize=7)
        ax.set_ylim(0, 1.1)
        ax.set_title(f"Cluster {cid}", fontsize=12, fontweight="bold", pad=20)
    plt.suptitle("CS2 Role Profiles — Radar View", fontsize=14, y=1.05)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "g2_radar_charts.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("💾 Saved → g2_radar_charts.png")

# ─────────────────────────────────────────────────────────────────────────────
# ROLE ASSIGNMENT
# FIX 4: Added "Rifler" role so the large generic rifler cluster isn't
# force-assigned AWPer (which previously scored -0.81 for that cluster)
# ─────────────────────────────────────────────────────────────────────────────
scaled_cols_present = [c for c in g2_df.columns if c.endswith("_scaled")]
cluster_scaled      = g2_df.groupby("cluster_id")[scaled_cols_present].mean()

def _find_col(cols, keyword):
    matches = [c for c in cols if keyword in c]
    return matches[0] if matches else None

fe_col    = _find_col(cluster_scaled.columns, "first_engagement")
dmg_col   = _find_col(cluster_scaled.columns, "damage")
ut_col    = _find_col(cluster_scaled.columns, "utility")
iso_col   = _find_col(cluster_scaled.columns, "isolation")
dist_col  = _find_col(cluster_scaled.columns, "distance")
kill_col  = _find_col(cluster_scaled.columns, "kill_count")
hs_col    = _find_col(cluster_scaled.columns, "headshot")
trade_col = _find_col(cluster_scaled.columns, "trade")
flash_col = _find_col(cluster_scaled.columns, "flash")
surv_col  = _find_col(cluster_scaled.columns, "survived")
open_col  = _find_col(cluster_scaled.columns, "opening")
multi_col = _find_col(cluster_scaled.columns, "multi_kill")

def score_clusters(cluster_scaled):
    # FIX 4: Rifler added so Cluster 0 (moderate dmg, moderate kills, high utility,
    # high distance) has a proper home instead of being assigned AWPer at score -0.81.
    roles_def = {
        "Entry":   {fe_col: -2.5, dmg_col: +2.0, kill_col: +1.5,
                    surv_col: -1.0, open_col: +2.0, multi_col: +1.0},
        "Support": {ut_col: +3.0, iso_col: -2.0, trade_col: +2.0,
                    flash_col: +2.0, surv_col: +0.5},
        "Lurker":  {iso_col: +2.5, fe_col: +1.5, dist_col: +1.0,   # dist weight reduced (FIX 6)
                    surv_col: +1.0, trade_col: -1.0},
        "AWPer":   {hs_col: -2.5, dmg_col: +1.5, dist_col: -1.5,
                    kill_col: +1.0, iso_col: +0.5},
        "IGL":     {ut_col: +1.5, surv_col: +2.0, open_col: -1.5,
                    flash_col: +1.0, kill_col: -1.0, multi_col: -1.0},
        "Rifler":  {dmg_col: +2.0, kill_col: +2.0, hs_col: +1.5,   # FIX 4: new role
                    fe_col: -1.0, surv_col: +0.5, multi_col: +1.0},
    }

    scores = {cid: {role: 0.0 for role in roles_def}
              for cid in cluster_scaled.index}
    for role, weights in roles_def.items():
        for col, w in weights.items():
            if col and col in cluster_scaled.columns:
                for cid in cluster_scaled.index:
                    scores[cid][role] += w * float(cluster_scaled[col].loc[cid])

    assigned, used_roles, all_pairs = {}, set(), []
    for cid, role_scores in scores.items():
        best_score = max(role_scores.values())
        all_pairs.append((best_score, cid, role_scores))
    all_pairs.sort(reverse=True)

    for _, cid, role_scores in all_pairs:
        for role, _ in sorted(role_scores.items(), key=lambda x: -x[1]):
            if role not in used_roles:
                assigned[cid] = role
                used_roles.add(role)
                break
        else:
            assigned[cid] = f"Flex_{cid}"

    for cid in cluster_scaled.index:
        if cid not in assigned:
            assigned[cid] = "Flex"

    return assigned, scores

role_map, role_scores = score_clusters(cluster_scaled)
print("\n  📋 Cluster → Role Mapping:")
for cid, role in sorted(role_map.items()):
    scores_str = " | ".join(
        f"{r}:{s:.2f}" for r, s in
        sorted(role_scores[cid].items(), key=lambda x: -x[1])
    )
    print(f"     Cluster {cid} → {role:<10}  [{scores_str}]")

g2_labeled = g2_df.copy()
g2_labeled["role_label"] = g2_labeled["cluster_id"].map(role_map)

print("\n  Role Label Distribution:")
role_dist = g2_labeled["role_label"].value_counts()
for role, cnt in role_dist.items():
    print(f"  {role:<15}: {cnt} ({100*cnt/len(g2_labeled):.1f}%)")

# Role visualization
role_colors = {
    "Entry":   "#E63946", "Support": "#2A9D8F", "Lurker":  "#E9C46A",
    "AWPer":   "#457B9D", "IGL":     "#8338EC", "Rifler":  "#F4A261",
    "Flex":    "#999999",
}
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors_bar = [role_colors.get(r, "#999") for r in role_dist.index]
axes[0].bar(role_dist.index, role_dist.values, color=colors_bar,
            edgecolor="white", linewidth=0.8)
axes[0].set(title="Role Label Distribution",
            xlabel="Role", ylabel="Player-Round Count")
axes[0].tick_params(axis="x", rotation=15)

r_plot  = g2_labeled["role_label"].values[idx_smp]
uniq_r  = sorted(set(r_plot))
pal_r   = {r: role_colors.get(r, f"C{i}") for i, r in enumerate(uniq_r)}
for role in uniq_r:
    mask = r_plot == role
    axes[1].scatter(X_plot[mask, 0], X_plot[mask, 1],
                    c=pal_r[role], label=role, alpha=0.55, s=18)
axes[1].set(title="PCA Clusters — Role Labels",
            xlabel=f"PC1 ({var_exp[0]:.1%})", ylabel=f"PC2 ({var_exp[1]:.1%})")
axes[1].legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "g2_roles_final.png", dpi=150)
plt.show()
print("💾 Saved → g2_roles_final.png")

g2_labeled.to_parquet(OUTPUT_DIR / "g2_labeled.parquet", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# PLAYER-LEVEL AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n🧑‍🤝‍🧑 Aggregating to player-level roles…")
player_stats = []
for player_name, pgroup in g2_labeled.groupby("player_name"):
    role_counts   = pgroup["role_label"].value_counts()
    dominant_role = role_counts.index[0]
    consistency   = role_counts.iloc[0] / len(pgroup)
    final_role    = dominant_role if consistency >= 0.50 else "Flex"
    mean_stats    = pgroup[orig_cols_present].mean().to_dict()
    mean_stats.update({
        "player_name":        player_name,
        "dominant_role":      dominant_role,
        "final_role":         final_role,
        "role_consistency":   round(consistency, 3),
        "total_rounds":       len(pgroup),
        "total_matches":      pgroup["match_id"].nunique(),
        "mean_gmm_confidence": round(pgroup["gmm_max_prob"].mean(), 3),
    })
    player_stats.append(mean_stats)

player_roles_df = pd.DataFrame(player_stats).sort_values("role_consistency", ascending=False)

print(f"\n📊 Player-Level Summary ({len(player_roles_df)} players):")
for role in sorted(player_roles_df["final_role"].unique()):
    subset = player_roles_df[player_roles_df["final_role"] == role]
    print(f"\n  ── {role} ──")
    for _, p in subset.head(5).iterrows():
        print(f"    {p['player_name']:<22} consistency={p['role_consistency']:.0%}"
              f"  rounds={p['total_rounds']}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.hist(player_roles_df["role_consistency"], bins=20,
         color="#457B9D", edgecolor="white", alpha=0.8)
ax1.axvline(0.50, color="#E63946", linestyle="--", label="Flex threshold (50%)")
ax1.set(xlabel="Role Consistency", ylabel="# Players",
        title="Player Role Consistency Distribution")
ax1.legend()

p_role_dist = player_roles_df["final_role"].value_counts()
colors_p    = [role_colors.get(r, "#999") for r in p_role_dist.index]
ax2.bar(p_role_dist.index, p_role_dist.values, color=colors_p,
        edgecolor="white", linewidth=0.8)
ax2.set(title="Player-Level Role Distribution",
        xlabel="Role", ylabel="# Players")
ax2.tick_params(axis="x", rotation=15)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "g2_player_roles.png", dpi=150)
plt.show()
print("💾 Saved → g2_player_roles.png")

player_roles_df.to_parquet(OUTPUT_DIR / "g2_player_roles.parquet", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# GROUND TRUTH VALIDATION (if available)
# ─────────────────────────────────────────────────────────────────────────────
if "true_role" in g2_labeled.columns:
    print("\n📊 Validation against ground truth:")
    true_labels  = g2_labeled["true_role"].values
    pred_labels  = g2_labeled["role_label"].values
    unique_roles = sorted(set(true_labels) | set(pred_labels))
    cm           = confusion_matrix(true_labels, pred_labels, labels=unique_roles)
    accuracy     = np.trace(cm) / cm.sum()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=unique_roles, yticklabels=unique_roles, ax=ax)
    ax.set(xlabel="Predicted Role", ylabel="True Role",
           title=f"Confusion Matrix (Accuracy: {accuracy:.1%})")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "g2_confusion_matrix.png", dpi=150)
    plt.show()
    print(f"  Overall accuracy: {accuracy:.1%}")
else:
    print("\n  ℹ No true_role column — skipping confusion matrix (real demo data).")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("  🏁 PIPELINE v3 COMPLETE")
print("═"*60)
print(f"  Features: 14 | Clustering: KMeans+GMM+HDBSCAN | Optimal K: {OPTIMAL_K}")
print(f"  Total player-rounds: {len(g2_labeled):,} | Unique players: {g2_labeled['player_name'].nunique()}")
print(f"\n  Outputs in: {OUTPUT_DIR}/")

expected_outputs = [
    "g1_features.parquet", "g1_clean.parquet",
    "g2_clustered.parquet", "g2_labeled.parquet", "g2_player_roles.parquet",
    "g2_correlation_matrix.png", "g2_k_selection.png",   # FIX 5: now actually saved
    "g2_pca_tsne.png", "g2_profile_heatmap.png",
    "g2_radar_charts.png", "g2_roles_final.png", "g2_player_roles.png",
]
for f in expected_outputs:
    status = "✅" if (OUTPUT_DIR / f).exists() else "❌ MISSING"
    print(f"  {status} {f}")

print("\n  Fix summary applied in v3:")
print("  ✅ FIX 1 — trade_kills capped at CAP_TRADE_KILLS inside compute_trade_kills()")
print("  ✅ FIX 2 — zone_primary uses global all_x percentiles (was always returning 2)")
print("  ✅ FIX 3 — flash_assists reads assister_name + assistedflash (was always 0)")
print("  ✅ FIX 4 — Rifler role added to role scorer (was being force-assigned to AWPer)")
print("  ✅ FIX 5 — g2_k_selection.png now saved")
print("  ✅ FIX 6 — HDBSCAN min_cluster_size lowered (20) for better granularity")
print("\n✅ All cells complete. 🎮✨")

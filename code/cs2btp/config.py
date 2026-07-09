"""Central configuration for the CS2 BTP pipeline.

Every tunable decision lives here so that robustness checks (Block 6)
can be run by changing config values, not code.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths ----
DRIVE_ROOT = Path("/content/drive/MyDrive/cs2-btp")
RAW_DEMS   = DRIVE_ROOT / "raw_dems"
PARSED     = DRIVE_ROOT / "parsed"
FEATURES   = DRIVE_ROOT / "features"
MODELS     = DRIVE_ROOT / "models"
ANALYSIS   = DRIVE_ROOT / "analysis"
FIGURES    = DRIVE_ROOT / "figures"
MANIFEST   = DRIVE_ROOT / "manifest.csv"

# .dem files are copied here (Colab local SSD) before parsing --
# parsing straight from Drive is 10-50x slower.
LOCAL_TMP  = Path("/content/tmp_dems")

def ensure_dirs():
    for p in [RAW_DEMS, PARSED, FEATURES, MODELS, ANALYSIS, FIGURES, LOCAL_TMP]:
        p.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------- parsing ----
DEFAULT_TICKRATE = 64          # CS2 servers/demos run at 64 ticks/s
TICK_HZ          = 2           # downsample player ticks to 2 Hz
TICK_STRIDE      = DEFAULT_TICKRATE // TICK_HZ

# player properties sampled at every downsampled tick
TICK_PROPS = [
    "X", "Y", "Z", "health", "is_alive",
    "active_weapon_name", "current_equip_value",
    "last_place_name", "team_num", "team_clan_name",
]

# game events we extract (with per-event extra props handled in parsing.py)
EVENT_NAMES = [
    "round_start", "round_freeze_end", "round_end", "round_officially_ended",
    "player_death", "player_hurt", "weapon_fire", "player_blind",
    "bomb_planted", "bomb_defused", "bomb_exploded",
]

# ------------------------------------------------------- feature params ----
OPENING_WINDOW_S = 20.0        # window for "forward advance" displacement
EARLY_UTIL_S     = 25.0        # utility thrown before this = "early util"
TRADE_WINDOW_S   = 5.0         # (kept for reference; proximity is used)
TRADE_RADIUS     = 700.0       # units: "close enough to trade" a teammate death
TELEPORT_JUMP    = 2000.0      # per-sample displacement above this = artifact, ignored

# weapon-name normalisation (defensive: demoparser2 may emit
# "AK-47", "ak47" or "weapon_ak47" depending on source column)
SNIPERS  = {"awp", "ssg08", "ssg 08", "scar20", "scar-20", "g3sg1"}
RIFLES   = {"ak47", "ak-47", "m4a4", "m4a1", "m4a1-s", "m4a1_silencer",
            "famas", "galilar", "galil ar", "aug", "sg556", "sg 553", "sg553"}
GRENADES = {"flashbang", "smokegrenade", "hegrenade", "molotov",
            "incgrenade", "inc grenade", "decoy", "smoke grenade",
            "he grenade"}
KNIVES_PREFIX = ("knife", "bayonet")

def norm_weapon(w):
    if w is None:
        return ""
    w = str(w).lower().strip()
    if w.startswith("weapon_"):
        w = w[len("weapon_"):]
    return w

# ------------------------------------------------- clustering feature set ----
# Deliberately behavioural, not performance: no kills / damage / ADR here.
CLUSTER_FEATURES = [
    # positioning
    "dist_nearest_teammate", "dist_team_centroid", "distance_traveled",
    "forward_disp_20s", "zone_entropy", "site_time_share",
    # timing / aggression
    "time_to_first_contact_s", "opening_duel_involved",
    # utility
    "flashes_thrown", "smokes_thrown", "mollies_thrown", "he_thrown",
    "util_early_share",
    # combat style
    "awp_share", "rifle_share", "shots_fired", "trade_presence",
    "time_alive_share",
    # economy behaviour (relative, not absolute)
    "buy_value_rel_team",
]
# NOTE: planted_bomb / defused_bomb / enemies_flashed are still computed by
# features.py and stored in the parquet, but are EXCLUDED from clustering:
# - planted/defused are round events, not playstyles, and defuses only occur
#   in CT-won rounds, coupling the taxonomy to outcomes non-behaviourally;
# - enemies_flashed is the SUCCESS of utility usage (performance, not
#   behaviour -- the choice to throw flashes is already captured by
#   flashes_thrown) and its heavy-tailed count distribution forms an
#   outlier pocket that hijacks a SOM cluster at any k.

# ------------------------------------------------------------ SOM / roles ----
SOM_K_RANGE       = range(4, 8)   # candidate numbers of roles per side
SOM_ITERATIONS    = 100_000
SOM_SIGMA         = 1.5
SOM_LR            = 0.5
STABILITY_RUNS    = 8             # reruns for the Drachen-style stability check
RANDOM_SEED       = 42

# ------------------------------------------------------------ consistency ----
MIN_PRIOR_ROUNDS  = 3   # rolling consistency defined only after >=3 prior rounds (same half)

# ------------------------------------------------------- economy heuristic ----
# per-player average team equipment value at freeze end (USD)
ECO_MAX   = 2000.0
FORCE_MAX = 3900.0

def buy_type(team_avg_equip):
    if team_avg_equip < ECO_MAX:
        return "eco"
    if team_avg_equip < FORCE_MAX:
        return "force"
    return "full"

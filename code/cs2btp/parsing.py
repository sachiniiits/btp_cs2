"""Block 1: .dem -> tidy parquet tables (rounds, ticks, events).

Design decisions
----------------
* One demo at a time, copied from Drive to local disk first (I/O speed),
  outputs written back to Drive as small parquet files.
* Rounds are reconstructed from round_freeze_end / round_end events,
  warmup filtered via the is_warmup_period prop.
* Player state is downsampled to TICK_HZ (default 2 Hz) which is plenty
  for positioning features and keeps each demo's tick table tiny
  (~50k rows instead of ~2M).
"""
import shutil
import traceback
import numpy as np
import pandas as pd
from demoparser2 import DemoParser

from . import config as cfg
from . import manifest as mf

WINNER_MAP = {2: "T", 3: "CT", "2": "T", "3": "CT", "T": "T", "CT": "CT",
              "t": "T", "ct": "CT"}


def _safe_event(parser, name, player=None, other=None) -> pd.DataFrame:
    """demoparser2 returns an empty *list* (not a DataFrame) when a demo has
    zero events of the requested type -- coerce anything odd to an empty df."""
    try:
        df = parser.parse_event(name, player=player, other=other)
    except Exception:
        return pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    return df


def _filter_warmup(df) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    if df.empty or "is_warmup_period" not in df.columns:
        return df
    col = df["is_warmup_period"]
    keep = ~col.astype(str).str.lower().isin(["true", "1"])
    return df[keep]


def build_rounds(parser) -> pd.DataFrame:
    """One row per official round: round_num, freeze/end ticks, winner."""
    re_df = _safe_event(parser, "round_end",
                        other=["is_warmup_period", "total_rounds_played"])
    fe_df = _safe_event(parser, "round_freeze_end",
                        other=["is_warmup_period"])
    if re_df.empty or fe_df.empty:
        raise ValueError("no round_end / round_freeze_end events found")

    re_df = _filter_warmup(re_df).sort_values("tick")
    fe_df = _filter_warmup(fe_df).sort_values("tick")

    if "winner" in re_df.columns:
        re_df["winner_side"] = re_df["winner"].map(WINNER_MAP)
    else:
        re_df["winner_side"] = None
    re_df = re_df[re_df["winner_side"].isin(["T", "CT"])]

    fe_ticks = fe_df["tick"].to_numpy()
    rows, prev_end = [], -1
    for _, r in re_df.iterrows():
        end_tick = int(r["tick"])
        cand = fe_ticks[(fe_ticks < end_tick) & (fe_ticks > prev_end)]
        if len(cand) == 0:
            continue  # spurious round_end with no matching freeze end
        start_tick = int(cand.max())
        rows.append(dict(start_tick=start_tick, end_tick=end_tick,
                         winner_side=r["winner_side"],
                         reason=r.get("reason", None)))
        prev_end = end_tick

    rounds = pd.DataFrame(rows)
    if rounds.empty:
        raise ValueError("no valid rounds reconstructed")
    rounds["round_num"] = np.arange(1, len(rounds) + 1)
    rounds["round_len_s"] = (rounds["end_tick"] - rounds["start_tick"]) / cfg.DEFAULT_TICKRATE
    # sanity: a competitive round lasts roughly 5s (fast eco kill) .. 155s+plant
    rounds = rounds[(rounds["round_len_s"] > 4) & (rounds["round_len_s"] < 400)]
    rounds["round_num"] = np.arange(1, len(rounds) + 1)
    return rounds.reset_index(drop=True)


def _tick_list(rounds: pd.DataFrame) -> list:
    wanted = []
    for _, r in rounds.iterrows():
        ticks = list(range(int(r.start_tick), int(r.end_tick) + 1, cfg.TICK_STRIDE))
        if ticks[-1] != int(r.end_tick):
            ticks.append(int(r.end_tick))
        wanted.extend(ticks)
    return sorted(set(wanted))


def _assign_round(df: pd.DataFrame, rounds: pd.DataFrame) -> pd.DataFrame:
    """Attach round_num to any event/tick table via tick intervals; drop
    everything outside official rounds (this also drops warmup events)."""
    if df.empty or "tick" not in df.columns:
        return df.assign(round_num=pd.Series(dtype="Int64")) if df.empty else df
    starts = rounds["start_tick"].to_numpy()
    ends = rounds["end_tick"].to_numpy()
    nums = rounds["round_num"].to_numpy()
    idx = np.searchsorted(starts, df["tick"].to_numpy(), side="right") - 1
    ok = (idx >= 0) & (df["tick"].to_numpy() <= ends[np.clip(idx, 0, len(ends) - 1)])
    out = df.loc[ok].copy()
    out["round_num"] = nums[idx[ok]]
    return out


def parse_demo(dem_path, demo_id) -> dict:
    """Parse a single .dem into a dict of DataFrames."""
    parser = DemoParser(str(dem_path))
    try:
        header = parser.parse_header()
    except Exception:
        header = {}

    rounds = build_rounds(parser)

    # ---- events -----------------------------------------------------------
    kills = _safe_event(parser, "player_death",
                        player=["X", "Y", "Z", "last_place_name", "team_num"],
                        other=["is_warmup_period"])
    hurts = _safe_event(parser, "player_hurt", other=["is_warmup_period"])
    fires = _safe_event(parser, "weapon_fire", other=["is_warmup_period"])
    blinds = _safe_event(parser, "player_blind", other=["is_warmup_period"])
    bomb = pd.concat([
        _safe_event(parser, "bomb_planted").assign(bomb_event="planted"),
        _safe_event(parser, "bomb_defused").assign(bomb_event="defused"),
        _safe_event(parser, "bomb_exploded").assign(bomb_event="exploded"),
    ], ignore_index=True)

    kills, hurts, fires, blinds = (
        _filter_warmup(kills), _filter_warmup(hurts),
        _filter_warmup(fires), _filter_warmup(blinds))

    # ---- downsampled ticks -------------------------------------------------
    wanted = _tick_list(rounds)
    ticks = parser.parse_ticks(cfg.TICK_PROPS, ticks=wanted)

    # attach round numbers, drop out-of-round rows
    out = {"rounds": rounds, "header": pd.DataFrame([header]) if header else pd.DataFrame()}
    for name, df in [("ticks", ticks), ("kills", kills), ("hurts", hurts),
                     ("fires", fires), ("blinds", blinds), ("bomb", bomb)]:
        out[name] = _assign_round(df, rounds) if not df.empty else df

    # normalise steamid dtypes to string for safe merging later
    for name, df in out.items():
        if df.empty:
            continue
        for col in df.columns:
            if "steamid" in col:
                df[col] = df[col].astype("string")
    return out


def save_parsed(tables: dict, demo_id: str):
    out_dir = cfg.PARSED / demo_id
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)


def load_parsed(demo_id: str) -> dict:
    out_dir = cfg.PARSED / demo_id
    return {p.stem: pd.read_parquet(p) for p in out_dir.glob("*.parquet")}


def parse_and_save(demo_id: str, filename: str, verbose=True) -> bool:
    """Copy Drive->local, parse, write parquet back to Drive, clean up."""
    src = cfg.RAW_DEMS / filename
    local = cfg.LOCAL_TMP / filename
    try:
        if verbose:
            print(f"[{demo_id}] copying to local disk ...")
        shutil.copy2(src, local)
        if verbose:
            print(f"[{demo_id}] parsing ...")
        tables = parse_demo(local, demo_id)
        save_parsed(tables, demo_id)
        mf.set_status(demo_id, "parsed",
                      notes=f"{len(tables['rounds'])} rounds")
        if verbose:
            print(f"[{demo_id}] OK - {len(tables['rounds'])} rounds")
        return True
    except Exception as e:
        mf.set_status(demo_id, "failed", notes=f"{type(e).__name__}: {e}")
        print(f"[{demo_id}] FAILED: {e}")
        traceback.print_exc(limit=2)
        return False
    finally:
        local.unlink(missing_ok=True)


def parse_all_pending():
    todo = mf.pending("new")
    print(f"{len(todo)} demo(s) to parse")
    for _, row in todo.iterrows():
        parse_and_save(row["demo_id"], row["filename"])

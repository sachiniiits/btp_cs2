"""Block 4: role-consistency metrics -- the novel contribution.

Definitions (all computed WITHIN a side/half, because T and CT roles come
from separate taxonomies and mixing halves would be meaningless):

  modal_share  : fraction of rounds spent in the player's most common role
  entropy_norm : normalised Shannon entropy of the role distribution (0..1)
  switch_rate  : fraction of consecutive-round pairs where role changed

Two flavours:
  * match-level  : descriptive, computed over the whole half
  * rolling      : for round t, computed from rounds < t of the SAME half
                   only (>= MIN_PRIOR_ROUNDS priors) -- this is what enters
                   the outcome regression, preventing outcome->consistency
                   leakage.
"""
import numpy as np
import pandas as pd
from scipy.stats import entropy

from . import config as cfg


def _half_id(df: pd.DataFrame) -> pd.Series:
    """A player's 'half' = contiguous block on one side within a match.
    Encoded as demo_id + side + block index (handles OT side swaps)."""
    df = df.sort_values(["demo_id", "steamid", "round_num"])
    changed = (df.groupby(["demo_id", "steamid"])["side"]
                 .transform(lambda s: (s != s.shift()).cumsum()))
    return df["demo_id"] + "|" + df["steamid"] + "|h" + changed.astype(str)


def _modal_share(roles: np.ndarray) -> float:
    if len(roles) == 0:
        return np.nan
    _, counts = np.unique(roles, return_counts=True)
    return counts.max() / counts.sum()


def _entropy_norm(roles: np.ndarray, n_roles: int) -> float:
    if len(roles) == 0 or n_roles <= 1:
        return np.nan
    _, counts = np.unique(roles, return_counts=True)
    return float(entropy(counts / counts.sum(), base=2) / np.log2(n_roles))


def match_level(labeled: pd.DataFrame) -> pd.DataFrame:
    """Descriptive consistency per (demo, player, half)."""
    df = labeled.copy()
    df["half_key"] = _half_id(df)
    n_roles = {s: df.loc[df["side"] == s, "role_id"].nunique() for s in ("T", "CT")}
    rows = []
    for key, g in df.sort_values("round_num").groupby("half_key"):
        roles = g["role_id"].to_numpy()
        rows.append(dict(
            half_key=key, demo_id=g["demo_id"].iloc[0],
            steamid=g["steamid"].iloc[0], side=g["side"].iloc[0],
            player_name=g.get("player_name", pd.Series([""])).iloc[0],
            team_clan=g["team_clan"].iloc[0], n_rounds=len(g),
            modal_share=_modal_share(roles),
            entropy_norm=_entropy_norm(roles, n_roles[g["side"].iloc[0]]),
            switch_rate=float(np.mean(roles[1:] != roles[:-1])) if len(roles) > 1 else np.nan,
        ))
    out = pd.DataFrame(rows)
    out.to_parquet(cfg.ANALYSIS / "consistency_match_level.parquet", index=False)
    return out


def rolling(labeled: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Leakage-free: for each (player, round), consistency over PRIOR rounds
    of the same half. NaN until MIN_PRIOR_ROUNDS priors exist."""
    df = labeled.copy()
    df["half_key"] = _half_id(df)
    df = df.sort_values(["half_key", "round_num"])
    n_roles = {s: df.loc[df["side"] == s, "role_id"].nunique() for s in ("T", "CT")}

    ms, en, sw = [], [], []
    for _, g in df.groupby("half_key", sort=False):
        roles = g["role_id"].to_numpy()
        side = g["side"].iloc[0]
        for i in range(len(g)):
            prior = roles[:i]
            if len(prior) < cfg.MIN_PRIOR_ROUNDS:
                ms.append(np.nan); en.append(np.nan); sw.append(np.nan)
            else:
                ms.append(_modal_share(prior))
                en.append(_entropy_norm(prior, n_roles[side]))
                sw.append(float(np.mean(prior[1:] != prior[:-1])))
    df["roll_modal_share"] = ms
    df["roll_entropy"] = en
    df["roll_switch_rate"] = sw
    if save:
        df.to_parquet(cfg.ANALYSIS / "consistency_rolling.parquet", index=False)
    return df


def team_round_table(rolled: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Aggregate rolling player consistency to one row per (demo, round, side):
    team mean and team minimum of prior modal share."""
    grp = (rolled.groupby(["demo_id", "round_num", "side"])
           .agg(cons_mean=("roll_modal_share", "mean"),
                cons_min=("roll_modal_share", "min"),
                entropy_mean=("roll_entropy", "mean"),
                switch_mean=("roll_switch_rate", "mean"),
                team_avg_equip=("team_avg_equip", "first"),
                n_players=("steamid", "nunique"))
           .reset_index())
    if save:
        grp.to_parquet(cfg.ANALYSIS / "team_round_consistency.parquet", index=False)
    return grp

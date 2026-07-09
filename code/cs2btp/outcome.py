"""Block 5: does prior role consistency predict round wins?

Main model -- round-level logit, one row per round, T-side perspective:

  P(T wins round) ~ cons_diff (T - CT, from PRIOR rounds only)
                    + equip_diff + score_diff + round_num + pistol
                    + C(map_name),   SEs clustered by match.

Framed as CORRELATION, exactly as the research question states.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from . import config as cfg
from . import parsing
from . import manifest as mf


def build_round_dataset(team_round: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Merge T and CT consistency per round with round outcome + controls."""
    man = mf.load().set_index("demo_id")
    piv = team_round.pivot_table(
        index=["demo_id", "round_num"], columns="side",
        values=["cons_mean", "cons_min", "entropy_mean", "switch_mean",
                "team_avg_equip"]).reset_index()
    piv.columns = ["_".join([c for c in col if c]) for col in piv.columns]

    rows = []
    for demo_id, g in piv.groupby("demo_id"):
        try:
            rounds = parsing.load_parsed(demo_id)["rounds"][
                ["round_num", "winner_side"]]
        except Exception:
            continue
        g = g.merge(rounds, on="round_num", how="inner")
        g["demo_id"] = demo_id
        g["map_name"] = man.loc[demo_id, "map_name"] if demo_id in man.index else ""
        rows.append(g)
    df = pd.concat(rows, ignore_index=True)

    df["t_win"] = (df["winner_side"] == "T").astype(int)
    df["cons_diff"] = df["cons_mean_T"] - df["cons_mean_CT"]
    df["cons_min_diff"] = df["cons_min_T"] - df["cons_min_CT"]
    df["entropy_diff"] = df["entropy_mean_T"] - df["entropy_mean_CT"]
    df["equip_diff_k"] = (df["team_avg_equip_T"] - df["team_avg_equip_CT"]) / 1000.0
    df["pistol"] = df["round_num"].isin([1, 13]).astype(int)

    # score state BEFORE the round, from the T team's perspective
    df = df.sort_values(["demo_id", "round_num"])
    df["t_wins_so_far"] = (df.groupby("demo_id")["t_win"]
                             .transform(lambda s: s.shift().fillna(0).cumsum()))
    df["rounds_so_far"] = df.groupby("demo_id").cumcount()
    df["score_diff"] = df["t_wins_so_far"] - (df["rounds_so_far"] - df["t_wins_so_far"])
    # NOTE: sides swap at halftime, so score_diff-from-T-perspective is a
    # noisy control; the round_num + pistol terms absorb most structure.

    if save:
        df.to_parquet(cfg.ANALYSIS / "round_outcome_dataset.parquet", index=False)
    return df


def fit_main_logit(df: pd.DataFrame, cons_var="cons_diff"):
    d = df.dropna(subset=[cons_var, "equip_diff_k"]).copy()
    terms = [cons_var, "equip_diff_k", "score_diff", "round_num"]
    # drop degenerate terms (e.g. single-map subsets in robustness runs)
    if d["pistol"].nunique() > 1:
        terms.append("pistol")
    if d["map_name"].nunique() > 1:
        terms.append("C(map_name)")
    formula = "t_win ~ " + " + ".join(terms)
    model = smf.logit(formula, data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["demo_id"]}, disp=False)
    return model, d


def quartile_table(df: pd.DataFrame, cons_var="cons_diff") -> pd.DataFrame:
    d = df.dropna(subset=[cons_var]).copy()
    d["q"] = pd.qcut(d[cons_var], 4, labels=["Q1 (least)", "Q2", "Q3", "Q4 (most)"])
    out = d.groupby("q", observed=True).agg(
        t_win_rate=("t_win", "mean"), n=("t_win", "size")).reset_index()
    out.to_csv(cfg.ANALYSIS / "winrate_by_consistency_quartile.csv", index=False)
    return out


def half_level_ols(rolled_match_level: pd.DataFrame,
                   labeled: pd.DataFrame):
    """Coarser sanity check: per (demo, side-half), do teams whose players
    were more role-consistent win more rounds in that half?"""
    won = (labeled.groupby(["demo_id", "side"])
           .agg(rounds_won=("ctx_won_round", lambda s: s.sum() / 5.0),
                n_rounds=("round_num", lambda s: s.nunique()))
           .reset_index())
    team_cons = (rolled_match_level.groupby(["demo_id", "side"])
                 ["modal_share"].mean().reset_index()
                 .rename(columns={"modal_share": "team_modal_share"}))
    d = won.merge(team_cons, on=["demo_id", "side"])
    d["win_share"] = d["rounds_won"] / d["n_rounds"]
    model = smf.ols("win_share ~ team_modal_share + C(side)", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["demo_id"]})
    return model, d


def permutation_test(labeled: pd.DataFrame, team_round_fn=None, n_perm=200,
                     seed=cfg.RANDOM_SEED, cons_var="cons_diff"):
    """Shuffle role labels WITHIN each (demo, round, side) across players --
    destroys player-level consistency while preserving round-level role mix.
    All internal builds use save=False so canonical artifacts on Drive are
    never overwritten by permutation replicates. team_round_fn is accepted
    for backwards compatibility and ignored."""
    from . import consistency as cons
    rng = np.random.default_rng(seed)
    real_ds = build_round_dataset(
        cons.team_round_table(cons.rolling(labeled, save=False), save=False),
        save=False)
    real_model, _ = fit_main_logit(real_ds, cons_var=cons_var)
    real_beta = real_model.params.get(cons_var, np.nan)

    betas = []
    for i in range(n_perm):
        shuf = labeled.copy()
        shuf["role_id"] = (shuf.groupby(["demo_id", "round_num", "side"])["role_id"]
                           .transform(lambda s: rng.permutation(s.to_numpy())))
        ds = build_round_dataset(
            cons.team_round_table(cons.rolling(shuf, save=False), save=False),
            save=False)
        try:
            m, _ = fit_main_logit(ds, cons_var=cons_var)
            betas.append(m.params.get(cons_var, np.nan))
        except Exception:
            betas.append(np.nan)
    betas = np.array([b for b in betas if b == b])
    p = float(np.mean(np.abs(betas) >= abs(real_beta))) if len(betas) else np.nan
    return dict(real_beta=float(real_beta), perm_mean=float(betas.mean()),
                perm_sd=float(betas.std()), p_value=p, n_valid=len(betas))

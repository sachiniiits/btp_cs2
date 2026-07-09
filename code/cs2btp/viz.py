"""Figures for the thesis. All saved to FIGURES as 200-dpi PNGs."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import config as cfg


def _save(fig, name):
    cfg.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.FIGURES / f"{name}.png", dpi=200, bbox_inches="tight")
    return fig


def round_positions(ticks: pd.DataFrame, round_num: int, title=""):
    """Smoke-test plot: player traces for one round, colored by side."""
    rt = ticks[ticks["round_num"] == round_num]
    fig, ax = plt.subplots(figsize=(7, 7))
    for team, color in [(2, "tab:orange"), (3, "tab:blue")]:
        g = rt[rt["team_num"] == team]
        for pid, pg in g.groupby("steamid"):
            ax.plot(pg["X"], pg["Y"], color=color, alpha=0.5, lw=1)
            ax.scatter(pg["X"].iloc[:1], pg["Y"].iloc[:1], color=color, s=30)
    ax.set_title(title or f"Round {round_num} traces (orange=T, blue=CT)")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    return _save(fig, f"smoke_round_{round_num}_traces")


def umatrix(som, side: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(som.distance_map().T, cmap="bone_r", origin="lower")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"SOM U-matrix — {side} side")
    return _save(fig, f"som_umatrix_{side}")


def role_radar(profile: pd.DataFrame, names: dict, side: str, top_n=10):
    """One radar chart per role over the most discriminative features."""
    spread = profile.max(axis=1) - profile.min(axis=1)
    feats = spread.sort_values(ascending=False).head(top_n).index.tolist()
    n_roles = profile.shape[1]
    angles = np.linspace(0, 2 * np.pi, len(feats), endpoint=False).tolist()
    angles += angles[:1]
    ncol = min(3, n_roles)
    nrow = int(np.ceil(n_roles / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.5 * nrow),
                             subplot_kw=dict(polar=True))
    axes = np.atleast_1d(axes).ravel()
    for i, col in enumerate(profile.columns):
        vals = profile.loc[feats, col].tolist(); vals += vals[:1]
        ax = axes[i]
        ax.plot(angles, vals, lw=2); ax.fill(angles, vals, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feats, fontsize=7)
        rid = int(col.split("_")[1])
        ax.set_title(names.get(rid, col), fontsize=11)
    for j in range(n_roles, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"Emergent {side}-side roles (z-scored feature means)", y=1.02)
    fig.tight_layout()
    return _save(fig, f"role_radars_{side}")


def quartile_bar(qtab: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(qtab["q"].astype(str), qtab["t_win_rate"], color="tab:blue")
    ax.axhline(qtab["t_win_rate"].mean(), ls="--", c="gray", lw=1)
    ax.set_ylabel("T-side round win rate")
    ax.set_xlabel("Consistency advantage quartile (T − CT prior modal share)")
    ax.set_title("Round win rate by relative role consistency")
    return _save(fig, "winrate_by_consistency_quartile")


def coef_plot(model, drop_prefix="C(map_name)"):
    params = model.params[~model.params.index.str.startswith(drop_prefix)]
    ci = model.conf_int().loc[params.index]
    params = params.drop("Intercept", errors="ignore")
    ci = ci.drop("Intercept", errors="ignore")
    fig, ax = plt.subplots(figsize=(6, 0.6 * len(params) + 1.5))
    y = np.arange(len(params))
    ax.errorbar(params.values, y,
                xerr=[params.values - ci[0].values, ci[1].values - params.values],
                fmt="o", capsize=4)
    ax.axvline(0, ls="--", c="gray", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(params.index)
    ax.set_xlabel("Logit coefficient (95% CI)")
    ax.set_title("Round-win model")
    return _save(fig, "logit_coefficients")


def consistency_hist(match_level: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    for side, color in [("T", "tab:orange"), ("CT", "tab:blue")]:
        d = match_level.loc[match_level["side"] == side, "modal_share"].dropna()
        ax.hist(d, bins=20, alpha=0.5, label=side, color=color, density=True)
    ax.set_xlabel("Modal role share (per player-half)")
    ax.set_ylabel("Density"); ax.legend()
    ax.set_title("Distribution of role consistency")
    return _save(fig, "consistency_distribution")

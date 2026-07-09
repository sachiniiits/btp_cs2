"""Block 3: role discovery via Self-Organizing Maps.

This module is the deliberate methodological replication of
Drachen, Canossa & Yannakakis (IEEE CIG 2009), "Player Modeling using
Self-Organization in Tomb Raider: Underworld", transported to CS2:

  behavioural features -> emergent SOM -> cluster the map ->
  manually inspect + name the emergent roles -> show stability.

Trained SEPARATELY per side (T / CT): the two sides have structurally
different jobs, and mixing them muddies the taxonomy.
"""
import json
import numpy as np
import pandas as pd
import joblib
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

from . import config as cfg


# --------------------------------------------------------------- helpers ---
def som_grid_size(n_samples: int) -> int:
    """Vesanto heuristic: ~5*sqrt(N) neurons, square grid."""
    n_neurons = 5 * np.sqrt(n_samples)
    return max(6, int(np.ceil(np.sqrt(n_neurons))))


def fit_som(X: np.ndarray, seed: int = cfg.RANDOM_SEED) -> MiniSom:
    g = som_grid_size(len(X))
    som = MiniSom(g, g, X.shape[1], sigma=cfg.SOM_SIGMA,
                  learning_rate=cfg.SOM_LR, random_seed=seed)
    som.pca_weights_init(X)
    som.train_random(X, cfg.SOM_ITERATIONS)
    return som


def som_labels(som: MiniSom, X: np.ndarray, k: int, seed: int = cfg.RANDOM_SEED):
    """Cluster the SOM codebook with k-means, then label samples by their BMU's cluster."""
    g = som.get_weights().shape[0]
    codebook = som.get_weights().reshape(-1, X.shape[1])
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(codebook)
    node_cluster = km.labels_.reshape(g, g)
    labels = np.array([node_cluster[som.winner(x)] for x in X])
    return labels, node_cluster, km


# --------------------------------------------------------- model selection ---
def select_k(X: np.ndarray, k_range=cfg.SOM_K_RANGE, sample=4000,
             seed=cfg.RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    som = fit_som(X, seed)
    rows = []
    for k in k_range:
        labels, _, _ = som_labels(som, X, k, seed)
        sil = silhouette_score(X[idx], labels[idx]) if len(set(labels[idx])) > 1 else np.nan
        sizes = np.bincount(labels, minlength=k) / len(labels)
        rows.append(dict(k=k, silhouette=sil,
                         min_cluster_share=float(sizes.min()),
                         max_cluster_share=float(sizes.max())))
    return pd.DataFrame(rows)


def stability_check(X: np.ndarray, k: int, n_runs=cfg.STABILITY_RUNS,
                    bootstrap=True, seed=cfg.RANDOM_SEED) -> pd.DataFrame:
    """Drachen-style claim: the same structure re-emerges across reruns.

    Retrains the SOM with different seeds (and bootstrap resamples for the
    training step), always labelling the FULL dataset, then measures
    pairwise Adjusted Rand Index between runs.
    """
    rng = np.random.default_rng(seed)
    all_labels = []
    for i in range(n_runs):
        s = int(rng.integers(0, 1_000_000))
        Xtr = X[rng.integers(0, len(X), len(X))] if bootstrap else X
        som = fit_som(Xtr, s)
        labels, _, _ = som_labels(som, X, k, s)
        all_labels.append(labels)
    rows = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            rows.append(dict(run_i=i, run_j=j,
                             ari=adjusted_rand_score(all_labels[i], all_labels[j])))
    return pd.DataFrame(rows)


def kmeans_baseline(X: np.ndarray, k: int, som_lbls: np.ndarray,
                    seed=cfg.RANDOM_SEED) -> float:
    """Agreement between the SOM roles and a plain k-means baseline."""
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    return adjusted_rand_score(som_lbls, km.labels_)


# ------------------------------------------------------------- profiling ---
def profile_clusters(feat_df: pd.DataFrame, labels: np.ndarray,
                     feature_cols=None) -> pd.DataFrame:
    """z-scored mean of every feature per cluster: the table you stare at
    to hand-assign role names (exactly as done in the source paper)."""
    feature_cols = feature_cols or cfg.CLUSTER_FEATURES
    df = feat_df[feature_cols].copy()
    z = (df - df.mean()) / df.std(ddof=0).replace(0, 1)
    z["role_id"] = labels
    prof = z.groupby("role_id").mean().T
    prof.columns = [f"role_{c}" for c in prof.columns]
    return prof


def suggest_names(profile: pd.DataFrame, side: str) -> dict:
    """Heuristic first-guess names from the profile -- these are SUGGESTIONS
    to be reviewed and overwritten manually in the notebook."""
    names = {}
    for col in profile.columns:
        p = profile[col]
        rid = int(col.split("_")[1])
        if p.get("awp_share", 0) > 0.8:
            names[rid] = "AWPer"
        elif p.get("opening_duel_involved", 0) > 0.5 and p.get("time_to_first_contact_s", 0) < -0.3:
            names[rid] = "Entry/Opener" if side == "T" else "Aggressive info CT"
        elif p.get("dist_nearest_teammate", 0) > 0.6 and side == "T":
            names[rid] = "Lurker"
        elif (p.get("flashes_thrown", 0) + p.get("smokes_thrown", 0)) > 0.8:
            names[rid] = "Support/Util"
        elif side == "CT" and p.get("zone_entropy", 0) < -0.3 and p.get("site_time_share", 0) > 0.3:
            names[rid] = "Site Anchor"
        elif side == "CT" and p.get("distance_traveled", 0) > 0.4:
            names[rid] = "Rotator"
        else:
            names[rid] = f"{side}-role-{rid}"
    return names


# -------------------------------------------------------------- pipeline ---
def discover_roles(features: pd.DataFrame, k_by_side: dict,
                   seed=cfg.RANDOM_SEED, save=True):
    """Full Block-3 run. Returns features with role_id/role_name columns
    plus a dict of fitted artefacts per side."""
    out = features.copy()
    out["role_id"] = -1
    out["role_name"] = ""
    artefacts = {}
    for side in ("T", "CT"):
        mask = out["side"] == side
        sub = (out.loc[mask, cfg.CLUSTER_FEATURES]
               .replace([np.inf, -np.inf], np.nan).fillna(0.0))
        scaler = StandardScaler().fit(sub)
        X = scaler.transform(sub)
        som = fit_som(X, seed)
        k = k_by_side[side]
        labels, node_cluster, km = som_labels(som, X, k, seed)
        out.loc[mask, "role_id"] = labels
        prof = profile_clusters(out.loc[mask], labels)
        names = suggest_names(prof, side)
        out.loc[mask, "role_name"] = [f"{side}:{names[l]}" for l in labels]
        artefacts[side] = dict(scaler=scaler, som=som, km=km, k=k,
                               node_cluster=node_cluster, profile=prof,
                               names=names)
        if save:
            cfg.MODELS.mkdir(parents=True, exist_ok=True)
            joblib.dump(dict(scaler=scaler, som_weights=som.get_weights(),
                             node_cluster=node_cluster, k=k),
                        cfg.MODELS / f"som_{side}.joblib")
            prof.to_csv(cfg.MODELS / f"role_profile_{side}.csv")
            with open(cfg.MODELS / f"role_names_{side}.json", "w") as fh:
                json.dump({str(kk): v for kk, v in names.items()}, fh, indent=2)
    if save:
        out.to_parquet(cfg.FEATURES / "player_round_features_roles.parquet",
                       index=False)
    return out, artefacts


def apply_manual_names(labeled: pd.DataFrame, names_by_side: dict) -> pd.DataFrame:
    """After you inspect the profiles and decide on final names, call this
    with e.g. {'T': {0: 'Entry', 1: 'Lurker', ...}, 'CT': {...}}."""
    labeled = labeled.copy()
    for side, names in names_by_side.items():
        m = labeled["side"] == side
        labeled.loc[m, "role_name"] = labeled.loc[m, "role_id"].map(
            lambda r: f"{side}:{names.get(int(r), f'role-{r}')}")
        with open(cfg.MODELS / f"role_names_{side}.json", "w") as fh:
            json.dump({str(k): v for k, v in names.items()}, fh, indent=2)
    labeled.to_parquet(cfg.FEATURES / "player_round_features_roles.parquet",
                       index=False)
    return labeled


def load_labeled() -> pd.DataFrame:
    return pd.read_parquet(cfg.FEATURES / "player_round_features_roles.parquet")

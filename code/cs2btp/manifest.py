"""manifest.csv management.

The manifest is the single source of truth for the dataset. Every .dem in
raw_dems/ gets one row. The parsing loop is driven by, and reports back to,
this file -- which is what makes the whole pipeline resumable and what
protects against filename collisions (demo_id stays unique even if two
files describe the same team pairing).
"""
import re
import pandas as pd
from . import config as cfg

COLUMNS = ["demo_id", "filename", "event", "date", "team1", "team2",
           "map_name", "status", "notes"]
# status lifecycle: new -> parsed -> featured   (or 'failed')


def load() -> pd.DataFrame:
    if cfg.MANIFEST.exists():
        df = pd.read_csv(cfg.MANIFEST, dtype=str).fillna("")
        for c in COLUMNS:
            if c not in df.columns:
                df[c] = ""
        return df[COLUMNS]
    return pd.DataFrame(columns=COLUMNS)


def save(df: pd.DataFrame):
    df[COLUMNS].to_csv(cfg.MANIFEST, index=False)


def _guess_from_filename(stem: str):
    """Handles both plain 'team1-vs-team2-mN-map' names and prefixed
    '<event>__<stage>__team1-vs-team2-mN-map' names produced by the
    extraction script. Returns (event, team1, team2, map_name)."""
    event, core = "", stem
    parts = stem.split("__")
    if len(parts) >= 3:
        event = parts[0].replace("-", " ") + " | " + parts[1].replace("-", " ")
        core = "__".join(parts[2:])
    team1 = team2 = map_name = ""
    m = re.match(r"(.+?)-vs-(.+?)-m\d+-(.+)$", core)
    if m:
        team1, team2, map_name = m.group(1), m.group(2), m.group(3)
    else:
        p = core.split("-")
        if p:
            map_name = p[-1]
    # strip '-2'/'-3' dedupe suffixes without touching names like dust2
    map_name = re.sub(r"-\d+$", "", map_name)
    return event, team1.replace("-", " "), team2.replace("-", " "), map_name


def sync_with_raw() -> pd.DataFrame:
    """Add a manifest row for every new .dem found in raw_dems/."""
    df = load()
    known = set(df["filename"])
    rows = []
    for p in sorted(cfg.RAW_DEMS.glob("*.dem")):
        if p.name in known:
            continue
        stem = p.stem
        demo_id = stem
        # guarantee uniqueness even on filename collisions across events
        existing_ids = set(df["demo_id"]) | {r["demo_id"] for r in rows}
        k = 2
        while demo_id in existing_ids:
            demo_id = f"{stem}--{k}"
            k += 1
        ev, t1, t2, mp = _guess_from_filename(stem)
        rows.append(dict(demo_id=demo_id, filename=p.name, event=ev, date="",
                         team1=t1, team2=t2, map_name=mp, status="new",
                         notes=""))
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        save(df)
        print(f"manifest: added {len(rows)} new demo(s)")
    else:
        print("manifest: no new demos found")
    return df


def set_status(demo_id: str, status: str, notes: str = ""):
    df = load()
    mask = df["demo_id"] == demo_id
    df.loc[mask, "status"] = status
    if notes:
        df.loc[mask, "notes"] = notes[:300]
    save(df)


def pending(status_from="new") -> pd.DataFrame:
    df = sync_with_raw()
    return df[df["status"] == status_from]


def reset_failed():
    """Set every 'failed' demo back to 'new' so the next parse run retries it
    (use after fixing a parser bug or replacing a corrupt file)."""
    df = load()
    n = int((df["status"] == "failed").sum())
    df.loc[df["status"] == "failed", "notes"] = ""
    df.loc[df["status"] == "failed", "status"] = "new"
    save(df)
    print(f"manifest: reset {n} failed demo(s) to 'new'")
    return df

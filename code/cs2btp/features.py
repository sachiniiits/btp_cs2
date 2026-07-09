"""Block 2: player-round behavioural features.

One row per (demo, round, player). Features describe HOW someone plays,
not how well -- the Drachen et al. principle. Kills/damage are stored
only as ctx_* context columns and are never fed to the clustering.
"""
import numpy as np
import pandas as pd
from scipy.stats import entropy

from . import config as cfg
from . import parsing
from . import manifest as mf

SIDE_MAP = {2: "T", 3: "CT"}


def _norm_w(series):
    return series.map(cfg.norm_weapon) if series is not None else series


def _is_gun(w):
    return (w not in cfg.GRENADES) and (not w.startswith(cfg.KNIVES_PREFIX)) \
        and w not in ("", "world", "c4", "taser")


def _grenade_kind(w):
    if "flash" in w:
        return "flash"
    if "smoke" in w:
        return "smoke"
    if "molotov" in w or "inc" in w:
        return "molly"
    if w == "hegrenade" or w == "he grenade":
        return "he"
    return None


def build_demo_features(demo_id: str) -> pd.DataFrame:
    t = parsing.load_parsed(demo_id)
    rounds, ticks = t["rounds"], t["ticks"]
    kills = t.get("kills", pd.DataFrame())
    hurts = t.get("hurts", pd.DataFrame())
    fires = t.get("fires", pd.DataFrame())
    blinds = t.get("blinds", pd.DataFrame())
    bomb = t.get("bomb", pd.DataFrame())

    if not fires.empty and "weapon" in fires.columns:
        fires = fires.assign(wnorm=_norm_w(fires["weapon"]))

    rows = []
    for _, rnd in rounds.iterrows():
        rn, t0, t1 = int(rnd.round_num), int(rnd.start_tick), int(rnd.end_tick)
        rlen = float(rnd.round_len_s)
        rt = ticks[ticks["round_num"] == rn]
        if rt.empty:
            continue

        # ---- round-level context ------------------------------------------
        first_tick = rt[rt["tick"] == rt["tick"].min()]
        side_of = first_tick.set_index("steamid")["team_num"].map(SIDE_MAP).to_dict()
        clan_of = (first_tick.set_index("steamid")["team_clan_name"].astype(str).to_dict()
                   if "team_clan_name" in first_tick.columns else {})
        equip_of = first_tick.set_index("steamid")["current_equip_value"].astype(float).to_dict()

        team_equips = {}
        for s in ("T", "CT"):
            vals = [v for k, v in equip_of.items() if side_of.get(k) == s]
            team_equips[s] = float(np.mean(vals)) if vals else np.nan

        rk = kills[kills["round_num"] == rn] if not kills.empty else pd.DataFrame()
        rh = hurts[hurts["round_num"] == rn] if not hurts.empty else pd.DataFrame()
        rf = fires[fires["round_num"] == rn] if not fires.empty else pd.DataFrame()
        rb = blinds[blinds["round_num"] == rn] if not blinds.empty else pd.DataFrame()
        rbomb = bomb[bomb["round_num"] == rn] if not bomb.empty else pd.DataFrame()

        first_kill_tick = int(rk["tick"].min()) if len(rk) else None
        opening_ids = set()
        if first_kill_tick is not None:
            fk = rk[rk["tick"] == first_kill_tick]
            for c in ("attacker_steamid", "user_steamid"):
                if c in fk.columns:
                    opening_ids |= set(fk[c].dropna().astype(str))

        # positions of every player at each sampled tick (for trade proximity)
        pos_by_tick = rt.set_index(["tick", "steamid"])[["X", "Y", "Z"]]

        players = sorted(set(side_of.keys()))
        for pid in players:
            side = side_of.get(pid)
            if side not in ("T", "CT"):
                continue
            pt = rt[rt["steamid"] == pid].sort_values("tick")
            alive = pt[pt["is_alive"].astype(str).str.lower().isin(["true", "1"])] \
                if "is_alive" in pt.columns else pt
            n_alive = len(alive)

            f = dict(demo_id=demo_id, round_num=rn, steamid=str(pid), side=side,
                     team_clan=clan_of.get(pid, ""), round_len_s=rlen,
                     team_avg_equip=team_equips.get(side, np.nan),
                     buy_type=cfg.buy_type(team_equips.get(side, 4000.0)))
            if "name" in pt.columns and len(pt):
                f["player_name"] = str(pt["name"].iloc[0])

            # ---------------- positioning ----------------------------------
            mates = [q for q in players
                     if side_of.get(q) == side and q != pid]
            dmate, dcent = [], []
            for tick_v, grp in rt.groupby("tick"):
                me = grp[grp["steamid"] == pid]
                if me.empty:
                    continue
                mrow = me.iloc[0]
                if str(mrow.get("is_alive", "True")).lower() not in ("true", "1"):
                    continue
                mateg = grp[grp["steamid"].isin(mates)]
                mateg = mateg[mateg["is_alive"].astype(str).str.lower().isin(["true", "1"])] \
                    if "is_alive" in mateg.columns else mateg
                if len(mateg):
                    dx = mateg["X"].to_numpy(float) - float(mrow["X"])
                    dy = mateg["Y"].to_numpy(float) - float(mrow["Y"])
                    d = np.sqrt(dx ** 2 + dy ** 2)
                    dmate.append(d.min())
                    cx, cy = mateg["X"].astype(float).mean(), mateg["Y"].astype(float).mean()
                    dcent.append(np.hypot(float(mrow["X"]) - cx, float(mrow["Y"]) - cy))
            f["dist_nearest_teammate"] = float(np.mean(dmate)) if dmate else 0.0
            f["dist_team_centroid"] = float(np.mean(dcent)) if dcent else 0.0

            if n_alive >= 2:
                xy = alive[["X", "Y"]].astype(float).to_numpy()
                steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
                steps = steps[steps < cfg.TELEPORT_JUMP]
                f["distance_traveled"] = float(steps.sum())
            else:
                f["distance_traveled"] = 0.0

            # displacement achieved in the opening window
            if n_alive:
                start_pos = alive[["X", "Y"]].astype(float).iloc[0].to_numpy()
                cutoff = t0 + cfg.OPENING_WINDOW_S * cfg.DEFAULT_TICKRATE
                win = alive[alive["tick"] <= cutoff]
                end_pos = (win if len(win) else alive)[["X", "Y"]].astype(float).iloc[-1].to_numpy()
                f["forward_disp_20s"] = float(np.linalg.norm(end_pos - start_pos))
            else:
                f["forward_disp_20s"] = 0.0

            if n_alive and "last_place_name" in alive.columns:
                places = alive["last_place_name"].astype(str)
                counts = places.value_counts(normalize=True)
                f["zone_entropy"] = float(entropy(counts.to_numpy(), base=2)) if len(counts) > 1 else 0.0
                f["site_time_share"] = float(places.str.lower().str.contains("bombsite").mean())
            else:
                f["zone_entropy"], f["site_time_share"] = 0.0, 0.0

            f["time_alive_share"] = float(n_alive / max(len(pt), 1))

            # ---------------- weapon-in-hands style -------------------------
            if n_alive and "active_weapon_name" in alive.columns:
                w = alive["active_weapon_name"].map(cfg.norm_weapon)
                f["awp_share"] = float(w.isin(cfg.SNIPERS).mean())
                f["rifle_share"] = float(w.isin(cfg.RIFLES).mean())
            else:
                f["awp_share"], f["rifle_share"] = 0.0, 0.0

            # ---------------- timing / first contact ------------------------
            ttc = rlen
            opening = 0
            if len(rh):
                mine = rh[(rh.get("attacker_steamid", pd.Series(dtype=str)).astype(str) == str(pid))
                          | (rh.get("user_steamid", pd.Series(dtype=str)).astype(str) == str(pid))]
                if len(mine):
                    ttc = (int(mine["tick"].min()) - t0) / cfg.DEFAULT_TICKRATE
            f["time_to_first_contact_s"] = float(np.clip(ttc, 0, rlen))
            f["opening_duel_involved"] = float(str(pid) in opening_ids)

            # ---------------- utility ---------------------------------------
            fl = sm = mo = he = 0
            early = total_util = 0
            if len(rf):
                mine = rf[rf.get("user_steamid", pd.Series(dtype=str)).astype(str) == str(pid)]
                for _, ev in mine.iterrows():
                    kind = _grenade_kind(ev.get("wnorm", ""))
                    if kind:
                        total_util += 1
                        if (int(ev["tick"]) - t0) / cfg.DEFAULT_TICKRATE <= cfg.EARLY_UTIL_S:
                            early += 1
                        fl += kind == "flash"; sm += kind == "smoke"
                        mo += kind == "molly"; he += kind == "he"
                guns = mine[mine["wnorm"].map(_is_gun)] if "wnorm" in mine.columns else mine.iloc[0:0]
                f["shots_fired"] = float(len(guns))
            else:
                f["shots_fired"] = 0.0
            f["flashes_thrown"], f["smokes_thrown"] = float(fl), float(sm)
            f["mollies_thrown"], f["he_thrown"] = float(mo), float(he)
            f["util_early_share"] = float(early / total_util) if total_util else 0.0

            ef = 0
            if len(rb) and "attacker_steamid" in rb.columns:
                mine = rb[rb["attacker_steamid"].astype(str) == str(pid)]
                if "user_steamid" in mine.columns:
                    ef = int(sum(side_of.get(u) not in (side, None)
                                 for u in mine["user_steamid"].astype(str)))
                else:
                    ef = len(mine)
            f["enemies_flashed"] = float(ef)

            # ---------------- trade presence ---------------------------------
            trade = 0
            if len(rk) and "user_steamid" in rk.columns:
                mate_deaths = rk[rk["user_steamid"].astype(str).isin([str(m) for m in mates])]
                for _, d in mate_deaths.iterrows():
                    # nearest sampled tick at/before the death
                    cand = pt[pt["tick"] <= int(d["tick"])]
                    if cand.empty:
                        continue
                    me = cand.iloc[-1]
                    if str(me.get("is_alive", "True")).lower() not in ("true", "1"):
                        continue
                    ux, uy = d.get("user_X"), d.get("user_Y")
                    if pd.isna(ux) or pd.isna(uy):
                        continue
                    if np.hypot(float(me["X"]) - float(ux),
                                float(me["Y"]) - float(uy)) <= cfg.TRADE_RADIUS:
                        trade += 1
            f["trade_presence"] = float(trade)

            # ---------------- economy / objective ----------------------------
            te = team_equips.get(side, np.nan)
            f["buy_value_rel_team"] = float(equip_of.get(pid, np.nan) / te) if te and te == te and te > 0 else 1.0
            planted = defused = 0.0
            if len(rbomb) and "user_steamid" in rbomb.columns:
                mine = rbomb[rbomb["user_steamid"].astype(str) == str(pid)]
                planted = float((mine["bomb_event"] == "planted").any())
                defused = float((mine["bomb_event"] == "defused").any())
            f["planted_bomb"], f["defused_bomb"] = planted, defused

            # ---------------- context (NOT clustering features) --------------
            k = d = dmg = 0
            if len(rk):
                if "attacker_steamid" in rk.columns:
                    k = int((rk["attacker_steamid"].astype(str) == str(pid)).sum())
                if "user_steamid" in rk.columns:
                    d = int((rk["user_steamid"].astype(str) == str(pid)).sum())
            if len(rh) and "attacker_steamid" in rh.columns and "dmg_health" in rh.columns:
                dmg = int(rh.loc[rh["attacker_steamid"].astype(str) == str(pid),
                                 "dmg_health"].astype(float).clip(0, 100).sum())
            f["ctx_kills"], f["ctx_deaths"], f["ctx_damage"] = k, d, dmg
            f["ctx_won_round"] = float(rnd.winner_side == side)
            rows.append(f)

    return pd.DataFrame(rows)


def build_all(force=False) -> pd.DataFrame:
    """Build features for every parsed demo; concat + save."""
    df_m = mf.load()
    done = []
    for _, row in df_m[df_m["status"].isin(["parsed", "featured"])].iterrows():
        demo_id = row["demo_id"]
        out = cfg.FEATURES / f"{demo_id}.parquet"
        if out.exists() and not force:
            done.append(pd.read_parquet(out))
            continue
        print(f"features: {demo_id}")
        try:
            feats = build_demo_features(demo_id)
            cfg.FEATURES.mkdir(parents=True, exist_ok=True)
            feats.to_parquet(out, index=False)
            mf.set_status(demo_id, "featured")
            done.append(feats)
        except Exception as e:
            print(f"  FAILED: {e}")
    all_df = pd.concat(done, ignore_index=True) if done else pd.DataFrame()
    if len(all_df):
        all_df.to_parquet(cfg.FEATURES / "player_round_features.parquet", index=False)
        print(f"total player-rounds: {len(all_df)}")
    return all_df


def load_all() -> pd.DataFrame:
    return pd.read_parquet(cfg.FEATURES / "player_round_features.parquet")

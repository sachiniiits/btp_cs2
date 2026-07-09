"""Quality-control checks for parsed demos.

These checks FLAG demos for manual review; the feature stage does not
gate on them, so a flagged demo must be inspected by the analyst and
either replaced or explicitly retained. The QC report is also a thesis
artifact ("data validation" subsection).
"""
import pandas as pd
from . import config as cfg
from . import parsing


def check_demo(demo_id: str) -> dict:
    t = parsing.load_parsed(demo_id)
    rounds, ticks = t["rounds"], t["ticks"]
    res = {"demo_id": demo_id, "n_rounds": len(rounds)}

    # 1. plausible round count (MR12 regulation is 13..24; each overtime
    #    adds up to 6 rounds -- ceiling 60 allows very deep OT)
    res["rounds_plausible"] = 13 <= len(rounds) <= 60

    # 2. exactly 10 active players per round (5v5)
    per_round = ticks.groupby("round_num")["steamid"].nunique()
    res["players_per_round_mode"] = int(per_round.mode().iloc[0]) if len(per_round) else 0
    res["all_rounds_10_players"] = bool((per_round == 10).mean() > 0.9)

    # 3. both sides present each round
    sides = ticks.groupby("round_num")["team_num"].nunique()
    res["both_sides_every_round"] = bool((sides >= 2).all()) if len(sides) else False

    # 4. tick coverage: expected ~ round_len * TICK_HZ samples per player
    cov = (ticks.groupby("round_num")["tick"].nunique()
           / (rounds.set_index("round_num")["round_len_s"] * cfg.TICK_HZ))
    res["tick_coverage_median"] = float(cov.median()) if len(cov) else 0.0

    # 5. final score
    score = rounds["winner_side"].value_counts().to_dict()
    res["score_T_wins"] = int(score.get("T", 0))
    res["score_CT_wins"] = int(score.get("CT", 0))

    res["pass"] = bool(res["rounds_plausible"] and res["all_rounds_10_players"]
                       and res["both_sides_every_round"]
                       and res["tick_coverage_median"] > 0.7)
    return res


def qc_report(demo_ids) -> pd.DataFrame:
    rows = []
    for d in demo_ids:
        try:
            rows.append(check_demo(d))
        except Exception as e:
            rows.append({"demo_id": d, "pass": False, "error": str(e)})
    rep = pd.DataFrame(rows)
    cfg.ANALYSIS.mkdir(parents=True, exist_ok=True)
    rep.to_csv(cfg.ANALYSIS / "qc_report.csv", index=False)
    return rep

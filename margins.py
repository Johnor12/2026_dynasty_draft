#!/usr/bin/env python3
"""What KeepTradeCut's calculator says about every trade the league has made this season.

The sweep prices only packages inside a band of calculator margins (KTC_SLACK in trades.py,
KTC_LOSS in sweep.py). This shows where the league's own trades fell: every completed player
trade from Sleeper, each side's package on today's Crowd + Trades values, and the margin each
side accepted, so the band can be set from what managers here have actually agreed to.
Startup-draft slot swaps are skipped, since no board prices them, and values are today's
rather than the trade date's, so older trades drift. Picks are priced at the tier the
original team's current slot (HIS_PICKS, MY_2027_SLOT) gives them, in every season.

Usage:
    uv run margins.py
"""

from __future__ import annotations

import datetime
import json
from types import SimpleNamespace

from odds import SLEEPER, fetch
from trades import HIS_PICKS, LEAGUE, MY_2027_SLOT, ktc_blend, ktc_calculator, pick_tier

ROUND = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}


def main() -> int:
    league = json.loads(LEAGUE.read_text())
    league_id = league["league_id"]
    users = {u["user_id"]: u["display_name"] for u in fetch(f"{SLEEPER}/{league_id}/users")}
    team = {r["roster_id"]: users[r["owner_id"]] for r in fetch(f"{SLEEPER}/{league_id}/rosters")}
    players, picks = ktc_blend()
    slot = {name.split()[0]: s for his in HIS_PICKS.values() for name, s in his.items() if name.endswith("1st")}
    slot[league["me"]["username"]] = MY_2027_SLOT
    names = {sid: row["name"] for sid, row in league["players"].items()}

    trades = {}
    for week in range(0, league["weeks"][0] + 1):
        for t in fetch(f"{SLEEPER}/{league_id}/transactions/{week}"):
            if t["type"] == "trade" and t["status"] == "complete" and t["adds"]:
                trades[t["transaction_id"]] = t

    print("margin is each side's view of the calculator: positive means it favored him\n")
    accepted = []
    for t in sorted(trades.values(), key=lambda t: t["status_updated"]):
        sides: dict[int, list] = {r: [] for r in t["roster_ids"]}
        for sid, r in t["adds"].items():
            sides[r].append((names.get(sid, sid), players.get(sid, 0), sid))
        for p in t.get("draft_picks", []):
            key = f"{p['season']} {pick_tier(slot[team[p['roster_id']]])} {ROUND[p['round']]}"
            sides[p["owner_id"]].append((f"{p['season']} {ROUND[p['round']]} ({team[p['roster_id']]})", picks[key], key))
        a, b = t["roster_ids"]
        board = {key: value for side in sides.values() for _, value, key in side}
        got_a = [SimpleNamespace(sleeper_id=key) for _, _, key in sides[a]]
        got_b = [SimpleNamespace(sleeper_id=key) for _, _, key in sides[b]]
        adj_a, adj_b = ktc_calculator(board, got_a, got_b)
        when = datetime.datetime.fromtimestamp(t["status_updated"] / 1000).strftime("%Y-%m-%d")
        print(f"{when}")
        for r, adj in ((a, adj_a), (b, adj_b)):
            other = adj_b if r == a else adj_a
            pieces = ", ".join(f"{name} {value:.0f}" for name, value, _ in sides[r])
            print(f"  {team[r]:<14} gets {pieces}  | raw {sum(v for _, v, _ in sides[r]):5.0f}  calc {adj:5.0f}  margin {adj - other:+6.0f}")
            accepted.append((adj - other, team[r], when))
    accepted.sort()
    print("\nmargins accepted, worst first: " + ", ".join(f"{m:+.0f} ({who} {when})" for m, who, when in accepted[:6]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

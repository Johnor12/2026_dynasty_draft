#!/usr/bin/env python3
"""Fetch our Sleeper league's rosters and every player's rest-of-season projection.

Publishes ``league.json`` at the repo root: who holds whom (active, taxi, IR) and each
QB/RB/WR/TE's projected points from Sleeper's current week through the last scoring
week, in the league's own ``scoring_settings``. Projections are Sleeper's weekly ones
(currently Rotowire), summed week by week so byes and announced absences are already
in the total. Their endpoint is undocumented but is the one Sleeper's own app reads.

Position is the first of Sleeper's ``fantasy_positions`` this league can start, not
``position``: Travis Hunter is a ``DB`` there but WR-eligible.

Usage:
    uv run league_pipeline/fetch_league.py
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import urllib.request
from pathlib import Path

LEAGUE_ID = "1388293617071714304"
MY_USERNAME = "johnor"
API = "https://api.sleeper.app/v1"
PROJECTIONS_API = "https://api.sleeper.app/projections/nfl"
POSITIONS = ("QB", "RB", "WR", "TE")
# Playoffs start in week 15 with three one-week rounds, so week 17 is the last that scores.
LAST_WEEK = 17
TIMEOUT_SECONDS = 30

OUTPUT = Path(__file__).resolve().parent.parent / "league.json"


def get_json(url: str):
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return json.loads(response.read())


def main() -> int:
    state = get_json(f"{API}/state/nfl")
    # Sleeper's `week` is the week whose games are next, which is when a claim now plays.
    if state["season_type"] != "regular":
        print(f"error: NFL season_type is {state['season_type']!r}, not regular", file=sys.stderr)
        return 1
    league = get_json(f"{API}/league/{LEAGUE_ID}")
    if league["season"] != state["season"]:
        print(f"error: league season {league['season']} is not {state['season']}", file=sys.stderr)
        return 1
    rosters = get_json(f"{API}/league/{LEAGUE_ID}/rosters")
    usernames = {
        user["user_id"]: user["display_name"]
        for user in get_json(f"{API}/league/{LEAGUE_ID}/users")
    }

    me = next(
        (r for r in rosters if (usernames.get(r["owner_id"]) or "").lower() == MY_USERNAME),
        None,
    )
    if me is None:
        print(f"error: no roster owned by {MY_USERNAME!r}", file=sys.stderr)
        return 1

    scoring = league["scoring_settings"]
    weeks = list(range(state["week"], LAST_WEEK + 1))
    query = "&".join(f"position[]={p}" for p in POSITIONS)
    players: dict[str, dict] = {}
    companies: set[str] = set()
    for week in weeks:
        print(f"GET projections {state['season']} week {week}", file=sys.stderr)
        rows = get_json(f"{PROJECTIONS_API}/{state['season']}/{week}?season_type=regular&{query}")
        for row in rows:
            info = row["player"]
            position = next((p for p in info["fantasy_positions"] if p in POSITIONS), None)
            if position is None:
                continue
            companies.add(row["company"])
            entry = players.setdefault(
                row["player_id"],
                {
                    "name": f"{info['first_name']} {info['last_name']}",
                    "position": position,
                    "team": row["team"],
                    "injury_status": info["injury_status"],
                    "ros_points": 0.0,
                },
            )
            entry["ros_points"] += sum(
                value * scoring[stat] for stat, value in row["stats"].items() if stat in scoring
            )
    for entry in players.values():
        entry["ros_points"] = round(entry["ros_points"], 2)

    document = {
        "source": API,
        "fetched_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "league_id": LEAGUE_ID,
        "league_name": league["name"],
        "season": state["season"],
        "weeks": weeks,
        "projection_source": sorted(companies),
        "roster_positions": league["roster_positions"],
        "me": {
            "username": usernames[me["owner_id"]],
            "user_id": me["owner_id"],
            "roster_id": me["roster_id"],
        },
        "rosters": [
            {
                "roster_id": r["roster_id"],
                "owner_id": r["owner_id"],
                "username": usernames.get(r["owner_id"]),
                # `players` includes the taxi and IR players.
                "players": r["players"] or [],
                "taxi": r["taxi"] or [],
                "reserve": r["reserve"] or [],
            }
            for r in sorted(rosters, key=lambda r: r["roster_id"])
        ],
        "players": dict(sorted(players.items(), key=lambda kv: int(kv[0]))),
    }
    OUTPUT.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    print(
        f"league: {len(rosters)} rosters, {len(players)} projected players, "
        f"weeks {weeks[0]}-{weeks[-1]} -> {OUTPUT.name}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

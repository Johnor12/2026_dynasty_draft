#!/usr/bin/env python3
"""Fetch what rookie-draft slots have actually returned, in this league's scoring.

Publishes ``pick_history.json`` at the repo root: for each draft class and rookie slot,
the player taken there, his position, and his first- and second-season fantasy points
under the league's ``scoring_settings``. trades.py values a future pick as the expectation
over the players history put near that slot, so a pick is a lottery over real outcomes
rather than one averaged projection.

Rookie-draft order is proxied by NFL draft order among QB/RB/WR/TE (nflverse's
draft_picks release). Superflex rookie drafts broadly follow draft capital, with RBs a
little earlier and late QBs a little later than this proxy. Scoring is Sleeper's season
stats dotted with the league's scoring settings, the same arithmetic fetch_league.py
applies to projections. A drafted player Sleeper has never carried scored nothing.

Classes 2019-2024 are the ones with two completed seasons. Slots 1-40 cover this league's
four rookie rounds. History moves once a year, so this is not part of the waivers refresh.

Usage:
    uv run league_pipeline/fetch_pick_history.py
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import re
import sys
import urllib.request
from pathlib import Path

LEAGUE_ID = "1388293617071714304"
API = "https://api.sleeper.app/v1"
DRAFT_PICKS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/draft_picks/draft_picks.csv"
)
POSITIONS = ("QB", "RB", "WR", "TE")
CLASSES = range(2019, 2025)
SLOTS = 40
TIMEOUT_SECONDS = 60

OUTPUT = Path(__file__).resolve().parent.parent / "pick_history.json"


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"Accept": "*/*", "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return response.read()


def squash(name: str) -> str:
    """Sleeper's search_full_name form: lowercase letters only, no Jr./III suffix."""
    words = [w for w in name.lower().split() if w.strip(".") not in ("jr", "sr", "ii", "iii", "iv", "v")]
    return re.sub(r"[^a-z]", "", "".join(words))


def main() -> int:
    scoring = json.loads(fetch(f"{API}/league/{LEAGUE_ID}"))["scoring_settings"]
    players = json.loads(fetch(f"{API}/players/nfl"))
    by_name: dict[tuple[str, str], list[dict]] = {}
    for p in players.values():
        if p.get("position") in POSITIONS and p.get("search_full_name"):
            by_name.setdefault((p["search_full_name"], p["position"]), []).append(p)

    drafted = [
        row
        for row in csv.DictReader(io.StringIO(fetch(DRAFT_PICKS_URL).decode()))
        if int(row["season"]) in CLASSES and row["position"] in POSITIONS
    ]
    stats = {
        season: json.loads(fetch(f"{API}/stats/nfl/regular/{season}"))
        for season in range(CLASSES[0], CLASSES[-1] + 2)
    }

    def points(sleeper_id: str | None, season: int) -> float:
        row = stats[season].get(sleeper_id or "", {})
        return round(sum(v * scoring[k] for k, v in row.items() if k in scoring), 1)

    rows = []
    unmatched = []
    for year in CLASSES:
        picks = sorted((r for r in drafted if int(r["season"]) == year), key=lambda r: int(r["pick"]))
        for slot, r in enumerate(picks[:SLOTS], start=1):
            candidates = by_name.get((squash(r["pfr_player_name"]), r["position"]), [])
            # Same name and position twice: the one whose rookie year is this class.
            if len(candidates) > 1:
                candidates = [
                    c for c in candidates if (c.get("metadata") or {}).get("rookie_year") == str(year)
                ] or candidates
            sleeper_id = candidates[0]["player_id"] if len(candidates) == 1 else None
            if sleeper_id is None:
                unmatched.append(f"{year} {slot} {r['pfr_player_name']} {r['position']}")
            rows.append(
                {
                    "class": year,
                    "slot": slot,
                    "nfl_pick": int(r["pick"]),
                    "name": r["pfr_player_name"],
                    "position": r["position"],
                    "sleeper_id": sleeper_id,
                    "yr1_points": points(sleeper_id, year),
                    "yr2_points": points(sleeper_id, year + 1),
                }
            )
    if unmatched:
        print(f"warning: {len(unmatched)} drafted players not on Sleeper, scored 0:", file=sys.stderr)
        for line in unmatched:
            print(f"  {line}", file=sys.stderr)

    OUTPUT.write_text(
        json.dumps(
            {
                "source": {"draft_order": DRAFT_PICKS_URL, "stats": f"{API}/stats/nfl/regular"},
                "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "classes": list(CLASSES),
                "slots": SLOTS,
                "rows": rows,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"{len(rows)} rookie slots over {len(CLASSES)} classes -> {OUTPUT.name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Playoff and title odds in each target year, with and without a trade.

trades.py prices a trade in expected lineup points. This turns those points into chances of
winning, which is what a win-now-versus-later trade actually changes. Every roster is valued
as trades.py values it (same artifacts, same wire, same pick lotteries), then seasons are
simulated with each team's weekly score drawn around its projected lineup points with the
spread of this season's actual weekly scores.

2026 is simulated from here: Sleeper's standings and remaining schedule, each team's
rest-of-season points spread over the weeks left, and the league's playoff: six teams from
week 15, top two seeds on a bye, seeded by wins then points. 2027 and 2028 are each one
season with a random schedule and half of each team's DraftSharks 2027-28 points, so they
are "if the season were played with today's rosters" odds, and the trade's effect on them
is the point, not the level. Only the two rosters in the trade change.

Usage:
    uv run odds.py nedjh2 --give "2027 1st" --get "Kyren Williams"
    uv run odds.py nedjh2
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import urllib.request

from trades import (
    HIS_PICKS, LEAGUE, MY_PICKS, PICK_HISTORY, POOL, Pricer, Team, find, load_players,
    load_team, pick_lotteries, pick_players,
)

SLEEPER = "https://api.sleeper.app/v1/league"
SEASONS = 10000
# The league's playoff, from Sleeper's settings: playoff_teams 6, playoff_week_start 15,
# one week per round, so byes for the top two seeds and a final in week 17.
PLAYOFF_TEAMS = 6
LAST_REGULAR_WEEK = 14
FUTURE_WEEKS = 17  # a full season's points spread over 17 weeks, 14 of them regular season


def fetch(url: str):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def season_odds(
    mean: dict[int, float], sd: float, wins: dict[int, float], points: dict[int, float],
    schedule: list[list[tuple[int, int]]], rng: random.Random,
) -> dict[int, tuple[float, float, float]]:
    """Per roster: playoff share, title share, and mean final wins over SEASONS seasons."""
    ids = list(mean)
    made = {i: 0 for i in ids}
    titles = {i: 0 for i in ids}
    total_wins = {i: 0.0 for i in ids}

    def game(a: int, b: int) -> int:
        return a if rng.gauss(mean[a], sd) > rng.gauss(mean[b], sd) else b

    for _ in range(SEASONS):
        w = dict(wins)
        p = dict(points)
        for week in schedule:
            score = {i: rng.gauss(mean[i], sd) for i in ids}
            for i in ids:
                p[i] += score[i]
            for a, b in week:
                w[a if score[a] > score[b] else b] += 1
        seeds = sorted(ids, key=lambda i: (-w[i], -p[i]))[:PLAYOFF_TEAMS]
        for i in seeds:
            made[i] += 1
        for i in ids:
            total_wins[i] += w[i]
        champ = game(game(seeds[0], game(seeds[3], seeds[4])), game(seeds[1], game(seeds[2], seeds[5])))
        titles[champ] += 1
    return {i: (made[i] / SEASONS, titles[i] / SEASONS, total_wins[i] / SEASONS) for i in ids}


def random_schedule(ids: list[int], rng: random.Random) -> list[list[tuple[int, int]]]:
    weeks = []
    for _ in range(LAST_REGULAR_WEEK):
        order = ids[:]
        rng.shuffle(order)
        weeks.append(list(zip(order[0::2], order[1::2])))
    return weeks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("opponent", help="Sleeper username")
    ap.add_argument("--give", nargs="+", metavar="NAME", help="my players or picks to send")
    ap.add_argument("--get", nargs="+", metavar="NAME", help="his players to receive")
    args = ap.parse_args()
    if bool(args.give) != bool(args.get):
        ap.error("--give and --get go together")

    league = json.loads(LEAGUE.read_text())
    players = load_players(league, json.loads(POOL.read_text()))
    his_picks = HIS_PICKS.get(args.opponent, {})
    slots = MY_PICKS | his_picks
    picks = pick_players(slots)
    lotteries = pick_lotteries(json.loads(PICK_HISTORY.read_text()), slots)
    me = load_team(league, players, league["me"]["username"])
    me.picks = [picks[n] for n in MY_PICKS]
    him = load_team(league, players, args.opponent)
    him.picks = [picks[n] for n in his_picks]
    rostered = {sid for r in league["rosters"] for sid in r["players"]}
    pricer = Pricer(league, [p for sid, p in players.items() if sid not in rostered], lotteries)
    roster_id = {r["username"]: r["roster_id"] for r in league["rosters"]}
    name_of = {v: k for k, v in roster_id.items()}
    teams: dict[int, Team] = {roster_id[n]: load_team(league, players, n) for n in roster_id}
    values = {i: t.value(pricer.wire) for i, t in teams.items()}
    mine = roster_id[me.name]

    # Standings, the schedule, and the spread of weekly scores so far, live from Sleeper.
    league_id = league["league_id"]
    wins, points = {}, {}
    for r in fetch(f"{SLEEPER}/{league_id}/rosters"):
        s = r["settings"]
        wins[r["roster_id"]] = s["wins"]
        points[r["roster_id"]] = s["fpts"] + s.get("fpts_decimal", 0) / 100
    weeks_left = league["weeks"]
    schedule, scored = [], {i: [] for i in teams}
    for week in range(1, LAST_REGULAR_WEEK + 1):
        pairs: dict[int, list[int]] = {}
        for row in fetch(f"{SLEEPER}/{league_id}/matchups/{week}"):
            pairs.setdefault(row["matchup_id"], []).append(row["roster_id"])
            if week < weeks_left[0]:
                scored[row["roster_id"]].append(row["points"])
        if week >= weeks_left[0]:
            schedule.append([tuple(p) for p in pairs.values()])
    residuals = [p - statistics.mean(v) for v in scored.values() for p in v]
    played = len(next(iter(scored.values())))
    sd = statistics.pstdev(residuals) * (played / (played - 1)) ** 0.5
    print(f"weekly score spread {sd:.1f} over {played} played weeks; {len(schedule)} regular-season weeks left")

    rng = random.Random(1)
    future_schedule = random_schedule(list(teams), rng)

    def report(label: str, vals: dict[int, dict[str, float]]) -> None:
        now = season_odds(
            {i: v["yr1"] / len(weeks_left) for i, v in vals.items()}, sd, wins, points, schedule, rng
        )
        later = season_odds(
            {i: v["yr23"] / 2 / FUTURE_WEEKS for i, v in vals.items()}, sd,
            {i: 0 for i in vals}, {i: 0.0 for i in vals}, future_schedule, rng,
        )
        print(
            f"\n{label}: ros {vals[mine]['yr1']:.0f}, 2027-28 {vals[mine]['yr23']:.0f}\n"
            f"  2026:      playoffs {now[mine][0]:5.1%}  title {now[mine][1]:5.1%}  "
            f"{now[mine][2]:.1f} wins\n"
            f"  2027/2028: playoffs {later[mine][0]:5.1%}  title {later[mine][1]:5.1%}  each year"
        )
        for year, odds in (("2026", now), ("2027/28", later)):
            top = sorted(odds, key=lambda i: -odds[i][1])[:5]
            print(f"  {year} title odds: " + ", ".join(f"{name_of[i]} {odds[i][1]:.0%}" for i in top))

    report("no trade", values)
    if args.give:
        give = [find(me, n) for n in args.give]
        get = [find(him, n) for n in args.get]
        my_after, his_after, my_moves, his_moves = pricer.price(me, him, give, get)
        after = dict(values)
        after[mine], after[roster_id[him.name]] = my_after, his_after
        report(f"give {', '.join(args.give)} for {', '.join(args.get)}", after)
        for who, moves in ((me.name, my_moves), (him.name, his_moves)):
            if moves:
                print(f"  {who} then: {'; '.join(moves)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

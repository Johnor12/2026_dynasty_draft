#!/usr/bin/env python3
"""Refresh provider rankings and our league, then list +EV waiver moves.

Three refresh steps, each a standalone script run with ``uv run`` from the repo root,
stopping on the first failure:

    1. data_source_investigator/fetch_rankings.py   provider boards -> data/raw/
    2. data_source_investigator/build_rankings.py   -> data/rankings.json
    3. league_pipeline/fetch_league.py              rosters + projections -> league.json

Moves are then priced the way rank.py priced draft picks: my roster's expected optimal
lineup points (ranker/value.py) summed over two horizons, which in-season are
    yr1  = rest-of-season points, from league.json
    yr23 = DraftSharks' 2027-28 points, from pool.json (0 for anyone it does not project)
The wire fallback is the best unrostered body at each position in each horizon. So an add
also spends that free agent's value as a fallback, and a drop becomes one:
    EV(add A, drop B) = V(roster - B + A, wire - A + B) - V(roster, wire)
Only active players are drop candidates; releasing a taxi or IR player frees no active spot.

The refreshed provider rankings never enter EV — my valuation stays projection-based, as
it was in the draft. They are printed as each player's median market rank (`mkt`).

Each free agent is listed once, with his best drop. Moves are not additive: make the top
one and re-run.

Usage:
    uv run waivers.py
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from ranker.league import BENCH_SLOTS, POSITIONS, STARTING_SLOTS
from ranker.pool import Player
from ranker.value import HORIZONS, expected_lineup_value, horizon_points

REPO_ROOT = Path(__file__).resolve().parent
LEAGUE = REPO_ROOT / "league.json"
POOL = REPO_ROOT / "pool.json"
SOURCE_RANKINGS = REPO_ROOT / "data_source_investigator/data/rankings.json"

STEPS = (
    ("fetch rankings", "data_source_investigator/fetch_rankings.py"),
    ("build rankings", "data_source_investigator/build_rankings.py"),
    ("league", "league_pipeline/fetch_league.py"),
)

# The smallest gain that prints as +0.1. Deep-bench swaps are worth hundredths of a point.
MIN_EV = 0.05

Wire = dict[str, dict[str, float]]


def load_players(league: dict, pool: dict) -> dict[str, Player]:
    yr23 = {p["sleeper_id"]: p["points_3yr"] - p["points_1yr"] for p in pool["players"]}
    return {
        sid: Player(
            player_id=int(sid),
            name=row["name"],
            position=row["position"],
            team=row["team"] or "FA",
            age=None,
            bye_week=None,
            is_rookie=False,
            points=row["ros_points"] + yr23.get(sid, 0),
            points_1yr=row["ros_points"],
            provider_adp=None,
            sleeper_id=sid,
        )
        for sid, row in league["players"].items()
    }


def market_ranks(rankings: dict) -> dict[str, float]:
    ranks: dict[str, list[int]] = defaultdict(list)
    for source in rankings["sources"]:
        # Derived from pool.json, so it is frozen at the draft rather than refreshed.
        if source["id"] == "draftsharks_adp":
            continue
        for row in source["players"]:
            if row["sleeper_id"]:
                ranks[row["sleeper_id"]].append(row["rank"])
    return {sid: statistics.median(r) for sid, r in ranks.items()}


def top_free(free: list[Player]) -> dict[str, dict[str, list[Player]]]:
    """The two best free agents per horizon and position: enough to replace one add."""
    return {
        h: {
            pos: sorted(
                (p for p in free if p.position == pos),
                key=lambda p: (-horizon_points(p, h), p.player_id),
            )[:2]
            for pos in POSITIONS
        }
        for h in HORIZONS
    }


def wire_levels(top: dict, add: Player | None = None, drop: Player | None = None) -> Wire:
    out: Wire = {}
    for h, by_pos in top.items():
        out[h] = {}
        for pos, best in by_pos.items():
            left = [p for p in best if p is not add]
            level = horizon_points(left[0], h) if left else 0.0
            if drop is not None and drop.position == pos:
                level = max(level, horizon_points(drop, h))
            out[h][pos] = level
    return out


def roster_value(roster: list[Player], wire: Wire) -> dict[str, float]:
    return {h: expected_lineup_value(roster, wire[h], h) for h in HORIZONS}


def describe(p: Player | None, market: dict[str, float]) -> str:
    if p is None:
        return "(open spot)"
    rank = market.get(p.sleeper_id)
    mkt = f"#{rank:.0f}" if rank is not None else "-"
    return (
        f"{p.position} {p.name[:22]:<22} {p.team:<3} "
        f"{p.points_yr1:4.0f} {p.points_yr23:4.0f} {mkt:>5}"
    )


def main() -> int:
    for number, (name, script) in enumerate(STEPS, start=1):
        print(f"\n=== [{number}/{len(STEPS)}] {name} ===", file=sys.stderr)
        code = subprocess.run(["uv", "run", script], cwd=REPO_ROOT).returncode
        if code != 0:
            print(f"waivers failed at step '{name}' (exit {code})", file=sys.stderr)
            return code

    league = json.loads(LEAGUE.read_text())
    expected_slots = Counter(
        {"SUPER_FLEX" if slot == "SF" else slot: n for slot, n in STARTING_SLOTS.items()}
        | {"BN": BENCH_SLOTS}
    )
    if Counter(league["roster_positions"]) != expected_slots:
        print(
            f"error: Sleeper roster_positions {league['roster_positions']} no longer match "
            "ranker/league.py",
            file=sys.stderr,
        )
        return 1

    players = load_players(league, json.loads(POOL.read_text()))
    market = market_ranks(json.loads(SOURCE_RANKINGS.read_text()))
    me = next(r for r in league["rosters"] if r["roster_id"] == league["me"]["roster_id"])
    unprojected = [sid for sid in me["players"] if sid not in players]
    if unprojected:
        print(f"error: my players {unprojected} have no Sleeper projection row", file=sys.stderr)
        return 1

    roster = [players[sid] for sid in me["players"]]
    held = set(me["taxi"]) | set(me["reserve"])
    droppable = [players[sid] for sid in me["players"] if sid not in held]
    open_spots = len(league["roster_positions"]) - len(droppable)
    drops: list[Player | None] = droppable + ([None] if open_spots > 0 else [])

    rostered = {sid for r in league["rosters"] for sid in r["players"]}
    free = [p for sid, p in players.items() if sid not in rostered]
    top = top_free(free)
    base = roster_value(roster, wire_levels(top))

    moves = []
    for add in free:
        if add.points_yr1 <= 0 and add.points_yr23 <= 0:
            continue
        best = None
        for drop in drops:
            after = roster_value(
                [p for p in roster if p is not drop] + [add], wire_levels(top, add, drop)
            )
            gain = {h: after[h] - base[h] for h in HORIZONS}
            total = sum(gain.values())
            if best is None or total > best[0]:
                best = (total, gain, drop)
        if best[0] >= MIN_EV:
            moves.append((best[0], best[1], add, best[2]))
    moves.sort(key=lambda m: (-m[0], m[2].player_id))

    weeks = league["weeks"]
    print(
        f"\nweeks {weeks[0]}-{weeks[-1]} left; {len(droppable)}/{len(league['roster_positions'])} "
        f"active + {len(me['taxi'])} taxi + {len(me['reserve'])} IR; "
        f"{sum(p.points_yr1 > 0 for p in free)} free agents with a projection"
    )
    if not moves:
        print("no +EV waiver moves")
        return 0
    print(
        "EV in expected lineup points: this season (ros) + 2027-28 (y2-3); mkt = median "
        "provider rank\n"
    )
    columns = f"{'':<3}{'name':<23}{'tm':<4}{'ros':>4} {'y2-3':>4} {'mkt':>5}"
    print(f"{'EV':>6} {'ros':>6} {'y2-3':>6}   add {columns}   drop {columns}")
    for total, gain, add, drop in moves:
        print(
            f"{total:+6.1f} {gain['yr1']:+6.1f} {gain['yr23']:+6.1f}   "
            f"add {describe(add, market)}   drop {describe(drop, market)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

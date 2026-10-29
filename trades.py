#!/usr/bin/env python3
"""Price trades with one opponent: a proposed package, or every small package.

Reads the artifacts waivers.py refreshes (league.json, pool.json, the provider board and
its raw snapshots) plus pick_history.json, and does not refetch; run `uv run waivers.py`
first when the in-season artifacts are stale.

Both rosters are valued the way waivers.py values mine: expected optimal lineup points
(ranker/value.py) in the in-season horizons, yr1 = rest of season and yr23 = 2027-28,
against the same free-agent wire, except at quarterback, where the wire is taken as empty:
in superflex every startable QB is rostered, and the one playable free agent this season
had nobody behind him, so no trade is priced on the assumption that a QB can be found. A
side's EV is its value after the trade minus before.
Rosters refill to the league shape after the swap: an incoming player keeps IR or taxi
when the receiver has that spot open, surplus active bodies are cut cheapest-first, and an
open active spot takes the best free agent.

My 2027 picks (MY_PICKS) are tradeable, and so are the opponent's at the slots HIS_PICKS
gives them. A pick is a lottery, not a projection: valuing a mid first as one
averaged rookie made it bench depth worth +2 points, because the average rookie never
starts while the hit does. So a pick is priced as the expectation, over the players
history put within PICK_WINDOW slots of its own (pick_history.json, six classes), of the
lineup value each would add to the roster holding the pick, carrying his own position and
his real first two seasons of scoring as the pick's 2027-28 points. A pick that changes
hands loses that value on the sender's current roster and gains it on the receiver's
post-trade roster; a pick that stays put is taken as worth the same before and after, and
two picks in one package are valued independently.

Every package also carries my market delta on KeepTradeCut superflex TE+, on FantasyCalc
10-team superflex, and `mkt`, their blend, with FantasyCalc rescaled onto KTC's scale by
the ratio of the boards over shared players. KeepTradeCut values are its "Crowd + Trades"
blend (`blendValue` in the snapshot), the trade calculator's default source, which prices
picks a few hundred below the crowd-only `value` the rankings page lists. `KTCcalc` is my
margin on KeepTradeCut's trade
calculator, whose value adjustment (ported below from the site's script) shrinks every
piece toward a share of the trade's best piece and credits the side whose pieces keep more
of their value with the single-player equivalent of the gap, so quantity-for-quality reads
as a loss for the seller and even a 1-for-1 gap is amplified. The opponent is scored on
both horizons by default; --tanking scores him on yr23 only, for a seller who has written
off this season. His EV is printed but a sweep filters on what he will check: my margin
on the calculator may be at most KTC_SLACK, about what his own Burrow-for-Love offer
conceded there.

Usage:
    uv run trades.py nedjh2 --tanking
    uv run trades.py nedjh2 --tanking --give "Josh Jacobs" --get "Chase Brown"
    uv run trades.py nedjh2 --tanking --give "2027 2nd" --get "Chase Brown"
    uv run trades.py nedjh2 --tanking --give "Josh Jacobs" --get "nedjh2 2027 1st"
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import multiprocessing
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ranker.pool import Player
from ranker.value import HORIZONS, expected_lineup_value
from waivers import MIN_EV, Wire, load_players, roster_value, top_free, wire_levels

REPO_ROOT = Path(__file__).resolve().parent
LEAGUE = REPO_ROOT / "league.json"
POOL = REPO_ROOT / "pool.json"
PICK_HISTORY = REPO_ROOT / "pick_history.json"
RANKINGS = REPO_ROOT / "data_source_investigator/data/rankings.json"
KTC_RAW = REPO_ROOT / "data_source_investigator/data/raw/keeptradecut.html"
FANTASYCALC_RAW = REPO_ROOT / "data_source_investigator/data/raw/fantasycalc.json"

# Where 2027 picks fall in the rookie draft: reverse order of mean 2026 wins in sweep.py's
# simulation after week 7 (nedjh2 2.6 ... johnor and justinmfftt 9.3). Both boards price
# picks by tier: early is 1-3, mid 4-7, late 8-10.
MY_2027_SLOT = 9
# My 2027 1st went to nedjh2 on 2026-10-22.
MY_PICKS = {"2027 2nd": 10 + MY_2027_SLOT}
# Each opponent's 2027 firsts and seconds as Sleeper's traded picks had them on
# 2026-10-28, named by the team whose finish sets the slot.
HIS_PICKS = {
    "nedjh2": {
        "nedjh2 2027 1st": 1, "nedjh2 2027 2nd": 11, "AndNic31 2027 1st": 5,
        "johnor 2027 1st": MY_2027_SLOT, "justinmfftt 2027 2nd": 20,
    },
    "UGrizz14": {"UGrizz14 2027 1st": 2, "UGrizz14 2027 2nd": 12},
    "dccb31": {"dccb31 2027 1st": 4, "dccb31 2027 2nd": 14},
    "dirtydole": {"dirtydole 2027 1st": 3, "dirtydole 2027 2nd": 13},
    "AndNic31": {"AndNic31 2027 2nd": 15},
    "bwood637": {"bwood637 2027 1st": 6, "bwood637 2027 2nd": 16},
    "jedmxk": {"jedmxk 2027 2nd": 18},
    "samlemonskoff": {"samlemonskoff 2027 1st": 7, "samlemonskoff 2027 2nd": 17},
    "justinmfftt": {"justinmfftt 2027 1st": 10, "jedmxk 2027 1st": 8},
}
PICK_WINDOW = 2  # slots either side of mine whose history forms the lottery
KTC_VARIANCE = 5  # the calculator's default fairness band, in percent of the two totals
KTC_MAX_PLAYER = 10000
# My margin on the calculator he will still accept. His own Burrow-for-Love offer favored me
# by about 1,500 there on the Crowd + Trades values, so he is not treating its verdict as a
# hard line.
KTC_SLACK = 1500
WORKERS = 8
TOP_PER_SHAPE = 12
TOP_PER_TARGET = 12

Market = dict[str, dict[str, float]]  # board -> asset key -> value


@dataclass
class Team:
    """One roster in the league shape: who is active, on taxi, and on IR."""

    name: str
    active: list[Player]
    taxi: list[Player]
    reserve: list[Player]
    picks: list[Player] = field(default_factory=list)

    def bodies(self) -> list[Player]:
        return self.active + self.taxi + self.reserve

    def players(self) -> list[Player]:
        return self.bodies() + self.picks

    def value(self, wire: Wire) -> dict[str, float]:
        return roster_value(self.bodies(), wire)


def pick_players(picks: dict[str, int]) -> dict[str, Player]:
    """Picks as zero-point markers, so they can sit in a package like a player."""
    return {
        name: Player(
            player_id=-1 - i,
            name=name,
            position="WR",
            team="-",
            age=None,
            bye_week=None,
            is_rookie=True,
            points=0,
            points_1yr=0,
            provider_adp=None,
            sleeper_id=name,
        )
        for i, name in enumerate(picks)
    }


def pick_lotteries(history: dict, picks: dict[str, int]) -> dict[str, list[Player]]:
    """Each pick's outcomes: the rookies history put near its slot, as 2027-28 bodies."""
    out: dict[str, list[Player]] = {}
    for i, (name, slot) in enumerate(picks.items()):
        near = [r for r in history["rows"] if abs(r["slot"] - slot) <= PICK_WINDOW]
        out[name] = [
            Player(
                player_id=-100 * (i + 1) - j,
                name=f"{name} as {row['name']}",
                position=row["position"],
                team="-",
                age=None,
                bye_week=None,
                is_rookie=True,
                points=round(row["yr1_points"] + row["yr2_points"]),
                points_1yr=0,
                provider_adp=None,
                sleeper_id=None,
            )
            for j, row in enumerate(near)
        ]
    return out


def lottery_value(bodies: list[Player], outcomes: list[Player], wire: Wire) -> float:
    """Expected 2027-28 lineup points a pick adds to this roster."""
    base = expected_lineup_value(bodies, wire["yr23"], "yr23")
    return sum(
        expected_lineup_value(bodies + [o], wire["yr23"], "yr23") - base for o in outcomes
    ) / len(outcomes)


def pick_tier(slot: int) -> str:
    """KeepTradeCut's and FantasyCalc's pick tiers: early is 1-3 in the round, mid 4-7, late 8-10."""
    in_round = (slot - 1) % 10 + 1
    return "Early" if in_round <= 3 else "Mid" if in_round <= 7 else "Late"


def ktc_blend() -> tuple[dict[str, float], dict[str, float]]:
    """KeepTradeCut's Crowd + Trades values off the raw snapshot: players by Sleeper id, picks by KTC name.

    Players come from the matched board, which supplies the Sleeper ids; their values and the
    picks' are the Crowd + Trades blend read off the raw snapshot (build_rankings.py keeps the
    crowd-only value, and picks are not players so it drops them, which is why the KTC fetch
    includes RDP). The blend matched the live calculator on 2026-10-22.
    """
    rankings = json.loads(RANKINGS.read_text())
    text = KTC_RAW.read_text()
    start = text.index('<script type="application/json" id="ktc-players">')
    start = text.index("[", start)
    ktc_rows, _ = json.JSONDecoder().raw_decode(text, start)
    blend = {row["playerName"]: row["superflexValues"]["tep"]["blendValue"] for row in ktc_rows}
    ktc = {
        row["sleeper_id"]: blend[row["name"]]
        for source in rankings["sources"]
        if source["id"] == "keeptradecut"
        for row in source["players"]
        if row["sleeper_id"]
    }
    ktc_picks = {row["playerName"]: blend[row["playerName"]] for row in ktc_rows if row["position"] == "RDP"}
    return ktc, ktc_picks


def market_values(picks: dict[str, int]) -> Market:
    """Blend, KTC, and FantasyCalc values by sleeper id, plus picks by name at their tier."""
    ktc, ktc_picks = ktc_blend()
    fc_rows = json.loads(FANTASYCALC_RAW.read_text())
    fc = {r["player"]["sleeperId"]: r["value"] for r in fc_rows if r["player"].get("sleeperId")}
    fc_picks = {r["player"]["name"]: r["value"] for r in fc_rows}
    shared = [sid for sid in ktc if sid in fc]
    scale = sum(ktc[s] for s in shared) / sum(fc[s] for s in shared)
    for name, slot in picks.items():
        year, rnd = name.split()[-2:]
        tier = pick_tier(slot)
        ktc[name] = ktc_picks[f"{year} {tier} {rnd}"]
        fc[name] = fc_picks[f"{year} {rnd} ({tier})"]
    blend = {}
    for key in set(ktc) | set(fc):
        both = [v for v in (ktc.get(key), fc[key] * scale if key in fc else None) if v is not None]
        blend[key] = sum(both) / len(both)
    return {"mkt": blend, "ktc": ktc, "fc": fc}


def worth(board: dict[str, float], assets) -> float:
    return sum(board.get(p.sleeper_id, 0) for p in assets)


# KeepTradeCut's trade calculator, ported from site.min.js (adjustPackageNew, ALGOTOUSE 2) and
# checked against the live page. Pick values are unscaled (pickVal 0).
def _ktc_round(x: float) -> int:
    return math.floor(x + 0.5)  # Math.round


def _ktc_shrunk(value: float, best: float) -> float:
    """A piece's value once consolidation is priced: near full for the best piece, much less below it."""
    return (0.1 * (value / 10099) ** 1.4 + 0.7 * (value / (1.05 * best)) ** 1.25 + 0.2) * value


def _ktc_single(gap: float, best: float, top: float) -> int:
    """The one-piece value whose shrunk value is `gap` (the site's Newton solve)."""
    a, t, x = top + 100, best, 5 * gap
    for _ in range(20):
        f = (0.1 * (x / a) ** 1.4 + 0.7 * (x / (1.05 * t)) ** 1.25 + 0.2) * x - gap
        d = 0.24 * x ** 1.4 / a ** 1.4 + 1.575 * x ** 1.25 / (1.05 ** 1.25 * t ** 1.25) + 0.2
        step = x - f / d
        if abs(step - x) < 1e-8:
            return _ktc_round(step)
        x = step
    return _ktc_round(x)


def ktc_close(e: float, a: float) -> bool:
    e, a = max(0.0, e), max(0.0, a)
    pct = min(100.0, abs(e - a) / (e + a) * 100) if e + a else 0.0
    return _ktc_round(10 * pct) / 10 <= KTC_VARIANCE


def ktc_calculator(board: dict[str, float], mine: list[Player], his: list[Player]) -> tuple[float, float]:
    """Adjusted totals (what I get, what he gets) as the calculator shows them, me as Team 1."""
    v1 = sorted((board.get(p.sleeper_id, 0) for p in mine), reverse=True)
    v2 = sorted((board.get(p.sleeper_id, 0) for p in his), reverse=True)
    total1, total2 = sum(v1), sum(v2)
    if not total1 or not total2:
        return total1, total2
    top, best = max(board.values()), max(v1[0], v2[0])
    e, a = sum(_ktc_shrunk(v, best) for v in v1), sum(_ktc_shrunk(v, best) for v in v2)
    gap = math.floor(abs(e - a))
    adj1 = adj2 = 0.0
    if ktc_close(total1, total2) and ktc_close(e, a):
        if e > a:
            b = total2 + _ktc_single(gap, best, top) - total1
            adj1, adj2 = (b, 0.0) if b > 0 else (0.0, -b)
        elif a > e:
            b = total1 + _ktc_single(gap, best, top) - total2
            adj1, adj2 = (0.0, b) if b > 0 else (-b, 0.0)
    elif e / total1 > a / total2:
        if e > a:
            b = total2 + _ktc_single(gap, best, top) - total1
            adj1, adj2 = (b, 0.0) if b > 0 else (0.0, -b)
        elif total1 != total2 and (w := _ktc_single(gap, best, top)) > 0:
            if total1 < total2:
                t = w - (total2 - total1)
                if t > 0:
                    adj2 = 0.0 if t > KTC_MAX_PLAYER else t
                else:
                    adj1 = -t
            else:
                adj2 = w - (total1 - total2)
    else:
        if a > e:
            b = total1 + _ktc_single(gap, best, top) - total2
            adj1, adj2 = (0.0, b) if b > 0 else (-b, 0.0)
        elif total1 != total2 and (w := _ktc_single(gap, best, top)) > 0:
            if total1 < total2:
                adj1 = w - (total2 - total1)
            else:
                t = w - (total1 - total2)
                if t > 0:
                    adj1 = 0.0 if t > KTC_MAX_PLAYER else t
                else:
                    adj2 = -t
    return max(0.0, total1 + adj1), max(0.0, total2 + adj2)


def load_team(league: dict, players: dict[str, Player], username: str) -> Team:
    row = next((r for r in league["rosters"] if r["username"] == username), None)
    if row is None:
        sys.exit(f"error: no roster for {username!r}; usernames: "
                 f"{[r['username'] for r in league['rosters']]}")
    missing = [sid for sid in row["players"] if sid not in players]
    if missing:
        sys.exit(f"error: {username}'s players {missing} have no Sleeper projection row")
    held = set(row["taxi"]) | set(row["reserve"])
    return Team(
        name=username,
        active=[players[s] for s in row["players"] if s not in held],
        taxi=[players[s] for s in row["taxi"]],
        reserve=[players[s] for s in row["reserve"]],
    )


class Pricer:
    def __init__(self, league: dict, free: list[Player], lotteries: dict[str, list[Player]]):
        # No QB fallback: the wire has had one playable QB all season and none behind him.
        free = [p for p in free if p.position != "QB"]
        self.top = top_free(free)
        self.wire = wire_levels(self.top)
        self.lotteries = lotteries
        self.active_slots = len(league["roster_positions"])
        self.taxi_slots = max(len(r["taxi"]) for r in league["rosters"])
        self.reserve_slots = max(len(r["reserve"]) for r in league["rosters"])
        self.free_pool = sorted(
            {p.player_id: p for by_pos in self.top.values() for ps in by_pos.values() for p in ps}
            .values(),
            key=lambda p: p.player_id,
        )

    def after(self, team: Team, give: list[Player], get: list[Player], sender: Team) -> tuple[Team, list[str]]:
        """The roster once `give` leaves, `get` arrives, and the shape is restored."""
        gone = set(id(p) for p in give)
        new = Team(
            team.name,
            [p for p in team.active if id(p) not in gone],
            [p for p in team.taxi if id(p) not in gone],
            [p for p in team.reserve if id(p) not in gone],
            [p for p in team.picks if id(p) not in gone],
        )
        for p in get:
            if p.player_id < 0:
                new.picks.append(p)
            elif p in sender.reserve and len(new.reserve) < self.reserve_slots:
                new.reserve.append(p)
            elif p in sender.taxi and len(new.taxi) < self.taxi_slots:
                new.taxi.append(p)
            else:
                new.active.append(p)
        moves: list[str] = []
        while len(new.active) > self.active_slots:
            cut = max(
                new.active,
                key=lambda p: sum(
                    roster_value([q for q in new.bodies() if q is not p],
                                 wire_levels(self.top, drop=p)).values()
                ),
            )
            new.active.remove(cut)
            moves.append(f"cut {cut.name}")
        while len(new.active) < self.active_slots:
            add = max(
                self.free_pool,
                key=lambda p: sum(roster_value(new.bodies() + [p], wire_levels(self.top, add=p)).values()),
            )
            new.active.append(add)
            moves.append(f"add FA {add.name}")
        return new, moves

    def pick_value(self, team: Team, pick: Player) -> float:
        return lottery_value(team.bodies(), self.lotteries[pick.name], self.wire)

    def price(self, me: Team, him: Team, give: list[Player], get: list[Player]):
        me2, my_moves = self.after(me, give, get, him)
        him2, his_moves = self.after(him, get, give, me)
        my_after, his_after = me2.value(self.wire), him2.value(self.wire)
        # A pick is worth its lottery on the sender's roster now, and on the receiver's once he has it.
        for p in give:
            if p.name in self.lotteries:
                my_after["yr23"] -= self.pick_value(me, p)
                his_after["yr23"] += self.pick_value(him2, p)
        for p in get:
            if p.name in self.lotteries:
                his_after["yr23"] -= self.pick_value(him, p)
                my_after["yr23"] += self.pick_value(me2, p)
        return my_after, his_after, my_moves, his_moves


def objective(value: dict[str, float], tanking: bool) -> float:
    return value["yr23"] if tanking else sum(value.values())


def find(team: Team, name: str) -> Player:
    hits = [p for p in team.players() if name.lower() in p.name.lower()]
    if len(hits) != 1:
        sys.exit(f"error: {name!r} matches {[p.name for p in hits]} on {team.name}")
    return hits[0]


def label(p: Player) -> str:
    return p.name if p.player_id < 0 else f"{p.position} {p.name}"


@dataclass
class Context:
    me: Team
    him: Team
    pricer: Pricer
    market: Market
    tanking: bool
    my_base: dict[str, float]
    his_base: dict[str, float]

    def score(self, give: list[Player], get: list[Player]):
        my_after, his_after, my_moves, his_moves = self.pricer.price(self.me, self.him, give, get)
        my_gain = {h: my_after[h] - self.my_base[h] for h in HORIZONS}
        his_gain = objective(his_after, self.tanking) - objective(self.his_base, self.tanking)
        deltas = {b: worth(v, get) - worth(v, give) for b, v in self.market.items()}  # my side
        t_me, t_him = ktc_calculator(self.market["ktc"], get, give)
        deltas["calc"] = t_me - t_him
        plausible = deltas["calc"] <= KTC_SLACK
        return my_gain, his_gain, deltas, plausible, my_moves, his_moves


# Set before the sweep forks its workers, which inherit it.
_CTX: Context | None = None


def _sweep(task: tuple[tuple[int, ...], int]) -> list[tuple]:
    give_index, n_get = task
    give = [_CTX.me.players()[i] for i in give_index]
    rows = []
    for get in itertools.combinations(_CTX.him.players(), n_get):
        my_gain, his_gain, deltas, plausible, _, _ = _CTX.score(give, list(get))
        if sum(my_gain.values()) < MIN_EV or not plausible:
            continue
        rows.append(
            (sum(my_gain.values()), tuple(map(label, give)), tuple(map(label, get)),
             my_gain, his_gain, deltas)
        )
    return rows


def main() -> int:
    global _CTX
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("opponent", help="Sleeper username")
    ap.add_argument("--tanking", action="store_true", help="score him on 2027-28 only")
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
    market = market_values(slots)
    me = load_team(league, players, league["me"]["username"])
    me.picks = [picks[n] for n in MY_PICKS]
    him = load_team(league, players, args.opponent)
    him.picks = [picks[n] for n in his_picks]

    rostered = {sid for r in league["rosters"] for sid in r["players"]}
    pricer = Pricer(league, [p for sid, p in players.items() if sid not in rostered], lotteries)
    ctx = Context(me, him, pricer, market, args.tanking, me.value(pricer.wire), him.value(pricer.wire))

    def line(give, get, my_gain, his_gain, deltas) -> str:
        return (
            f"{sum(my_gain.values()):+6.1f} {my_gain['yr1']:+6.1f} {my_gain['yr23']:+6.1f} | "
            f"{his_gain:+6.1f} | {deltas['mkt']:+6.0f} {deltas['ktc']:+6.0f} {deltas['fc']:+6.0f} "
            f"{deltas['calc']:+7.0f} | give {', '.join(give)} | get {', '.join(get)}"
        )

    print(
        f"{me.name}: ros {ctx.my_base['yr1']:.0f}, 2027-28 {ctx.my_base['yr23']:.0f}   "
        f"{him.name}: ros {ctx.his_base['yr1']:.0f}, 2027-28 {ctx.his_base['yr23']:.0f}"
        f"   (he is scored on {'2027-28 only' if args.tanking else 'both horizons'})"
    )
    for name, pick in picks.items():
        print(
            f"{name}: lottery over {len(lotteries[name])} rookies near slot {slots[name]}, "
            f"worth {pricer.pick_value(me, pick):+.1f} on my roster and "
            f"{pricer.pick_value(him, pick):+.1f} on his; "
            f"mkt {market['mkt'][name]:.0f} (KTC {market['ktc'][name]}, FC {market['fc'][name]})"
        )
    print("market deltas are from my side, in minus out; KTCcalc is my margin on KTC's calculator")
    header = (
        f"\n{'EV me':>6} {'ros':>6} {'y2-3':>6} | {'EV him':>6} | "
        f"{'mkt':>6} {'KTC':>6} {'FC':>6} {'KTCcalc':>7} | trade"
    )

    if args.give:
        give = [find(me, n) for n in args.give]
        get = [find(him, n) for n in args.get]
        my_gain, his_gain, deltas, plausible, my_moves, his_moves = ctx.score(give, get)
        print(header)
        print(line(list(map(label, give)), list(map(label, get)), my_gain, his_gain, deltas))
        for who, moves in ((me.name, my_moves), (him.name, his_moves)):
            if moves:
                print(f"  {who} then: {'; '.join(moves)}")
        if not plausible:
            print(f"  KTC's calculator favors me by {deltas['calc']:.0f}, past the {KTC_SLACK} he has conceded")
        return 0

    _CTX = ctx
    shapes = {
        "1-for-1": (1, 1), "2-for-1": (2, 1), "1-for-2": (1, 2), "2-for-2": (2, 2),
    }
    found: dict[str, list] = {}
    with multiprocessing.get_context("fork").Pool(WORKERS) as pool:
        for shape, (n_give, n_get) in shapes.items():
            tasks = [
                (give_index, n_get)
                for give_index in itertools.combinations(range(len(me.players())), n_give)
            ]
            rows = [row for chunk in pool.imap_unordered(_sweep, tasks, chunksize=4) for row in chunk]
            rows.sort(key=lambda r: (-r[0], r[1], r[2]))
            found[shape] = rows
            print(f"\n== {shape}: {len(rows)} plausible +EV packages ==", end="")
            print(header)
            for _, give, get, my_gain, his_gain, deltas in rows[:TOP_PER_SHAPE]:
                print(line(give, get, my_gain, his_gain, deltas))

    best: dict[str, tuple] = {}
    for rows in found.values():
        for row in rows:
            for target in row[2]:
                if target not in best or row[0] > best[target][0]:
                    best[target] = row
    print("\n== best package per target ==", end="")
    print(header)
    for target, row in sorted(best.items(), key=lambda kv: -kv[1][0])[:TOP_PER_TARGET]:
        _, give, get, my_gain, his_gain, deltas = row
        print(f"{target}: {line(give, get, my_gain, his_gain, deltas)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

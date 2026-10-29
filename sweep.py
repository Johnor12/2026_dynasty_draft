#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy"]
# ///
"""Sweep every small package with every opponent, scored in title odds over 2026-28.

trades.py prices one opponent's packages in lineup points and odds.py turns one package
into title odds. This does both for the whole league: every 1-for-1, 2-for-1, 1-for-2,
2-for-2, and depth-only 3-for-1 package with every opponent, scored as the change in my
cumulative title odds over the horizon, p(2026) + p(2027) + p(2028), with 2027 and 2028
the same today's-rosters season odds.py uses. Only packages KeepTradeCut's calculator
puts in play are scored: my margin at most KTC_SLACK, or he would refuse, and at least
-KTC_LOSS, since a package the market scores that far against me has never been worth
titles to me and pricing it is the cost. Only those adding at least MIN_TITLES are
printed, the best package per set of incoming pieces, and every kept package is dumped to
sweep_rows.json for verdicts.py to sort by the calculator's verdict. His own change in
cumulative title odds is shown but not filtered on: he judges a package on the
calculator and on what he is trying to do, not on this model. So a second list keeps only
the packages the calculator scores in his favor (its value adjustment already prices star
power, one strong piece over a package of equals) that also fit his stance, read off his
2026 playoff odds: a contender (CONTENDING or better) must not lose rest-of-season lineup
points, a tanker (TANKING or worse) must not lose 2027-28 points, and anyone between must
not lose their sum.

The season model is odds.py's, vectorized and run on common random numbers: every
evaluation reuses one set of weekly draws, so two rosters' odds differ by the rosters and
not by simulation noise. Each package is first priced in points, and the gradient of my
title odds (central differences per roster) turns that into a linear estimate; the season
is simulated exactly only when the estimate clears LINEAR_FLOOR. A 3-for-1 is
consolidation, so its three pieces come from my depth: players worth under DEPTH_TITLES
to me on their own.

numpy comes from the inline script metadata, so `uv run sweep.py` resolves it on its own
(`uv run --offline sweep.py` once it is cached). Reads what waivers.py refreshed, plus
Sleeper's live standings and schedule; run `uv run waivers.py` first.

Usage:
    uv run sweep.py
    uv run sweep.py UGrizz14 dirtydole
"""

from __future__ import annotations

import itertools
import json
import multiprocessing
import os
import random
import sys
import time
from dataclasses import dataclass

os.environ.setdefault("OMP_NUM_THREADS", "1")  # the worker pool is the parallelism, not BLAS
import numpy as np  # noqa: E402

from odds import FUTURE_WEEKS, PLAYOFF_TEAMS, live_season, random_schedule  # noqa: E402
from ranker.value import HORIZONS  # noqa: E402
from trades import (  # noqa: E402
    HIS_PICKS, KTC_SLACK, LEAGUE, MY_PICKS, PICK_HISTORY, POOL, REPO_ROOT, Pricer, Team,
    ktc_calculator, label, load_team, market_values, pick_lotteries, pick_players, worth,
)
from waivers import load_players, roster_value  # noqa: E402

SEASONS = 40000
MIN_TITLES = 0.10  # cumulative title odds over 2026-28 a package must add to be printed
LINEAR_FLOOR = MIN_TITLES / 2  # simulate exactly only when the linear estimate clears this
DEPTH_TITLES = 0.08  # a piece worth less than this to me on its own is depth
KTC_LOSS = 4000  # the calculator margin against me past which a package is not priced
CONTENDING = 2 / 3  # 2026 playoff odds at which he is buying this season
TANKING = 1 / 3  # and at which he has written it off
SHAPES = {"1-for-1": (1, 1), "2-for-1": (2, 1), "1-for-2": (1, 2), "2-for-2": (2, 2), "3-for-1": (3, 1)}
WORKERS = 16
TOP_PER_OPPONENT = 10
TOP_LEAGUE = 40
ROWS = REPO_ROOT / "sweep_rows.json"  # every package the last sweep kept, for verdicts.py


class Season:
    """One season's draws, reused for every roster valuation (common random numbers)."""

    def __init__(self, ids, schedule, wins0, points0, sd, seed):
        self.ids = list(ids)
        idx = {r: k for k, r in enumerate(self.ids)}
        T, W = len(self.ids), len(schedule)
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((SEASONS, W, T)) * sd
        self.zp = rng.standard_normal((SEASONS, 3, T)) * sd  # the three playoff rounds
        A = np.array([[idx[a] for a, b in wk] for wk in schedule])
        B = np.array([[idx[b] for a, b in wk] for wk in schedule])
        self.A, self.B = A.ravel(), B.ravel()
        w = np.repeat(np.arange(W), A.shape[1])
        # a beats b when mean_a - mean_b > z_b - z_a
        self.thresh = z[:, w, self.B] - z[:, w, self.A]
        self.MA = np.zeros((len(self.A), T))
        self.MA[np.arange(len(self.A)), self.A] = 1
        self.MB = np.zeros((len(self.B), T))
        self.MB[np.arange(len(self.B)), self.B] = 1
        self.noise_pts = z.sum(axis=1)
        self.W = W
        self.wins0 = np.array([wins0[r] for r in self.ids], float)
        self.pts0 = np.array([points0[r] for r in self.ids], float)
        self.ar = np.arange(SEASONS)

    def run(self, mean: dict[int, float]) -> dict[int, tuple[float, float, float]]:
        """Per roster id: playoff share, title share, and mean wins, as odds.py's season_odds."""
        m = np.array([mean[r] for r in self.ids])
        awin = (m[self.A] - m[self.B])[None, :] > self.thresh
        wins = self.wins0 + awin @ self.MA + (~awin) @ self.MB
        pts = self.pts0 + self.W * m + self.noise_pts
        seeds = np.argsort(-(wins * 1e5 + pts), axis=1)[:, :PLAYOFF_TEAMS]
        P = m[None, None, :] + self.zp

        def game(week, a, b):
            return np.where(P[self.ar, week, a] > P[self.ar, week, b], a, b)

        top = game(1, seeds[:, 0], game(0, seeds[:, 3], seeds[:, 4]))
        bottom = game(1, seeds[:, 1], game(0, seeds[:, 2], seeds[:, 5]))
        champ = game(2, top, bottom)
        T = len(self.ids)
        made = np.bincount(seeds.ravel(), minlength=T) / SEASONS
        titles = np.bincount(champ, minlength=T) / SEASONS
        mwins = wins.mean(axis=0)
        return {r: (made[k], titles[k], mwins[k]) for k, r in enumerate(self.ids)}


class Titles:
    """Cumulative title odds 2026-28, p2026 + 2 x p2027/28, for every roster from lineup values."""

    def __init__(self, league: dict):
        wins, points, schedule, self.sd, self.played = live_season(league)
        ids = sorted(wins)
        self.n_weeks = len(league["weeks"])
        self.now = Season(ids, schedule, wins, points, self.sd, seed=1)
        self.later = Season(
            ids, random_schedule(ids, random.Random(1)), {i: 0 for i in ids},
            {i: 0.0 for i in ids}, self.sd, seed=2,
        )

    def odds(self, values: dict[int, dict[str, float]]):
        now = self.now.run({i: v["yr1"] / self.n_weeks for i, v in values.items()})
        later = self.later.run({i: v["yr23"] / 2 / FUTURE_WEEKS for i, v in values.items()})
        return now, later

    def cumulative(self, values: dict[int, dict[str, float]]) -> dict[int, float]:
        now, later = self.odds(values)
        return {i: now[i][1] + 2 * later[i][1] for i in now}


@dataclass
class Context:
    me: Team
    teams: dict[str, Team]
    rid: dict[str, int]
    pricer: Pricer
    market: dict
    base: dict[int, dict[str, float]]
    titles: Titles
    base_now: dict
    base_later: dict
    opp: str = ""
    stance: str = ""
    grad: dict | None = None


# Set before the sweep forks its workers, which inherit it.
_CTX: Context | None = None


def gradient(ctx: Context, who: int) -> dict[tuple[int, str], float]:
    """d(my cumulative title odds) / d(point) of roster `who` per horizon, on common draws."""
    me = ctx.rid[ctx.me.name]
    out = {}
    for h, step in (("yr1", 20.0), ("yr23", 60.0)):
        up = {k: dict(v) for k, v in ctx.base.items()}
        dn = {k: dict(v) for k, v in ctx.base.items()}
        up[who][h] += step
        dn[who][h] -= step
        out[(who, h)] = (ctx.titles.cumulative(up)[me] - ctx.titles.cumulative(dn)[me]) / (2 * step)
    return out


def depth(ctx: Context, grad: dict) -> list[int]:
    """Indices of my pieces worth under DEPTH_TITLES on their own: the 3-for-1 pool."""
    me = ctx.rid[ctx.me.name]
    out = []
    for i, p in enumerate(ctx.me.players()):
        if p.player_id < 0:
            cost = grad[(me, "yr23")] * ctx.pricer.pick_value(ctx.me, p)
        else:
            without = roster_value([q for q in ctx.me.bodies() if q is not p], ctx.pricer.wire)
            cost = sum(grad[(me, h)] * (ctx.base[me][h] - without[h]) for h in HORIZONS)
        if cost < DEPTH_TITLES:
            out.append(i)
    return out


def stance(playoff_odds: float) -> str:
    return "contending" if playoff_odds >= CONTENDING else "tanking" if playoff_odds <= TANKING else "neutral"


def fits(stance: str, d_him: dict[str, float]) -> bool:
    """Whether the package serves what he is trying to do, in his own lineup points."""
    if stance == "contending":
        return d_him["yr1"] >= 0
    if stance == "tanking":
        return d_him["yr23"] >= 0
    return d_him["yr1"] + d_him["yr23"] >= 0


def _work(task: tuple[tuple[int, ...], int]) -> tuple[list[dict], list[int]]:
    """Rows worth printing, and how many packages reached each stage: all, priced, simulated."""
    give_index, n_get = task
    ctx = _CTX
    me_t, him_t = ctx.me, ctx.teams[ctx.opp]
    me, him = ctx.rid[me_t.name], ctx.rid[ctx.opp]
    give = [me_t.players()[i] for i in give_index]
    rows = []
    stages = [0, 0, 0]
    for get in itertools.combinations(him_t.players(), n_get):
        get = list(get)
        stages[0] += 1
        t_me, t_him = ktc_calculator(ctx.market["ktc"], get, give)
        calc = t_me - t_him
        if calc > KTC_SLACK or calc < -KTC_LOSS:
            continue
        stages[1] += 1
        my_after, his_after, my_moves, his_moves = ctx.pricer.price(me_t, him_t, give, get)
        d_me = {h: my_after[h] - ctx.base[me][h] for h in HORIZONS}
        d_him = {h: his_after[h] - ctx.base[him][h] for h in HORIZONS}
        linear = sum(ctx.grad[(me, h)] * d_me[h] + ctx.grad[(him, h)] * d_him[h] for h in HORIZONS)
        if linear < LINEAR_FLOOR:
            continue
        stages[2] += 1
        vals = dict(ctx.base)
        vals[me], vals[him] = my_after, his_after
        now, later = ctx.titles.odds(vals)
        d_now = now[me][1] - ctx.base_now[me][1]
        d_later = later[me][1] - ctx.base_later[me][1]
        if d_now + 2 * d_later < MIN_TITLES:
            continue
        fit = fits(ctx.stance, d_him)
        rows.append(dict(
            opp=ctx.opp, stance=ctx.stance, give=tuple(map(label, give)), get=tuple(map(label, get)),
            titles=d_now + 2 * d_later, now=d_now, later=d_later,
            his=(now[him][1] - ctx.base_now[him][1]) + 2 * (later[him][1] - ctx.base_later[him][1]),
            d_me=d_me, d_him=d_him, calc=calc, t_me=t_me, t_him=t_him, fits=fit, sendable=calc <= 0 and fit,
            mkt=worth(ctx.market["mkt"], get) - worth(ctx.market["mkt"], give),
            my_moves=my_moves, his_moves=his_moves,
        ))
    return rows, stages


def line(r: dict) -> str:
    text = (
        f"{r['titles']:+6.1%} (2026 {r['now']:+6.1%}, 27/28 {r['later']:+6.1%}/yr) him {r['his']:+6.1%} | "
        f"pts me {r['d_me']['yr1']:+4.0f}/{r['d_me']['yr23']:+4.0f} him {r['d_him']['yr1']:+4.0f}/{r['d_him']['yr23']:+4.0f} | "
        f"calc {r['calc']:+6.0f} mkt {r['mkt']:+6.0f} | {r['opp']}: give {', '.join(r['give'])} | get {', '.join(r['get'])}"
    )
    for who, moves in (("me", r["my_moves"]), ("him", r["his_moves"])):
        if moves:
            text += f"\n        {who} then: {'; '.join(moves)}"
    return text


def best_per_target(rows: list[dict]) -> list[dict]:
    best: dict[tuple, dict] = {}
    for r in rows:
        if r["get"] not in best or r["titles"] > best[r["get"]]["titles"]:
            best[r["get"]] = r
    return sorted(best.values(), key=lambda r: (-r["titles"], r["give"]))


def main() -> int:
    global _CTX
    league = json.loads(LEAGUE.read_text())
    players = load_players(league, json.loads(POOL.read_text()))
    me_name = league["me"]["username"]
    rid = {r["username"]: r["roster_id"] for r in league["rosters"]}
    opponents = sys.argv[1:] or [n for n in rid if n != me_name]
    unknown = [o for o in opponents if o not in rid or o == me_name]
    if unknown:
        sys.exit(f"error: no opponent {unknown}; usernames: {[n for n in rid if n != me_name]}")

    slots = dict(MY_PICKS)
    for picks in HIS_PICKS.values():
        slots |= picks
    pick_by_name = pick_players(slots)
    lotteries = pick_lotteries(json.loads(PICK_HISTORY.read_text()), slots)
    market = market_values(slots)
    teams: dict[str, Team] = {}
    for name in rid:
        team = load_team(league, players, name)
        own = MY_PICKS if name == me_name else HIS_PICKS.get(name, {})
        team.picks = [pick_by_name[n] for n in own]
        teams[name] = team
    rostered = {sid for r in league["rosters"] for sid in r["players"]}
    pricer = Pricer(league, [p for sid, p in players.items() if sid not in rostered], lotteries)
    base = {rid[n]: t.value(pricer.wire) for n, t in teams.items()}
    titles = Titles(league)
    base_now, base_later = titles.odds(base)
    ctx = Context(teams[me_name], teams, rid, pricer, market, base, titles, base_now, base_later)
    me = rid[me_name]

    print(
        f"weekly score spread {titles.sd:.1f} over {titles.played} played weeks; "
        f"{titles.n_weeks} weeks left; {SEASONS} seasons per evaluation\n"
        f"{'team':<14} {'ros':>5} {'27-28':>5} | {'2026 playoffs':>13} {'title':>6} {'wins':>5} | "
        f"{'27/28 playoffs':>14} {'title':>6} | {'titles 26-28':>12}"
    )
    for name, i in sorted(rid.items(), key=lambda kv: -(base_now[kv[1]][1] + 2 * base_later[kv[1]][1])):
        print(
            f"{name:<14} {base[i]['yr1']:5.0f} {base[i]['yr23']:5.0f} | "
            f"{base_now[i][0]:13.1%} {base_now[i][1]:6.1%} {base_now[i][2]:5.1f} | "
            f"{base_later[i][0]:14.1%} {base_later[i][1]:6.1%} | "
            f"{base_now[i][1] + 2 * base_later[i][1]:12.1%}"
        )
    my_grad = gradient(ctx, me)
    depth_ix = depth(ctx, my_grad)
    print(
        f"\none of my points is worth {my_grad[(me, 'yr1')]:.3%} of a title this season and "
        f"{my_grad[(me, 'yr23')]:.4%} in 2027-28; 3-for-1 pool: "
        f"{', '.join(label(ctx.me.players()[i]) for i in depth_ix)}\n"
        "columns: my cumulative title odds (2026 + 2 x 27/28), his; lineup points per horizon; "
        f"KTCcalc is my margin on the calculator, at most {KTC_SLACK}; mkt is my blended market delta"
    )

    everything: list[dict] = []
    for opp in opponents:
        ctx.opp = opp
        ctx.stance = stance(base_now[rid[opp]][0])
        ctx.grad = my_grad | gradient(ctx, rid[opp])
        _CTX = ctx
        rows: list[dict] = []
        # A pool per opponent: the workers fork with this opponent's context.
        with multiprocessing.get_context("fork").Pool(WORKERS) as pool:
            for shape, (n_give, n_get) in SHAPES.items():
                t0 = time.time()
                pool_ix = depth_ix if n_give == 3 else range(len(ctx.me.players()))
                tasks = [(g, n_get) for g in itertools.combinations(pool_ix, n_give)]
                found, stages = [], [0, 0, 0]
                for chunk, counts in pool.imap_unordered(_work, tasks, chunksize=2):
                    found += chunk
                    stages = [a + b for a, b in zip(stages, counts)]
                rows += found
                print(
                    f"  {opp} {shape}: {stages[0]} packages, {stages[1]} in the calculator's band, "
                    f"{stages[2]} simulated, {len(found)} kept, {time.time() - t0:.0f}s",
                    file=sys.stderr,
                )
        everything += rows
        print(
            f"\n== {opp} ({ctx.stance}, 2026 playoffs {base_now[rid[opp]][0]:.0%}): one of his points "
            f"costs me {ctx.grad[(rid[opp], 'yr1')]:.3%} this season, {ctx.grad[(rid[opp], 'yr23')]:.4%} "
            f"in 2027-28; {len(rows)} packages add {MIN_TITLES:.0%}+; best per target ==",
            flush=True,
        )
        for r in best_per_target(rows)[:TOP_PER_OPPONENT]:
            print("  " + line(r), flush=True)
        sendable = [r for r in rows if r["sendable"]]
        print(f"  -- {len(sendable)} where the calculator favors him and the package fits his stance --", flush=True)
        for r in best_per_target(sendable)[:TOP_PER_OPPONENT]:
            print("  " + line(r), flush=True)

    ROWS.write_text(json.dumps(everything))
    print(f"{len(everything)} kept packages written to {ROWS.name}", file=sys.stderr)
    for title, keep in (("", lambda r: True), (", sendable", lambda r: r["sendable"])):
        print(f"\n== league-wide: best per opponent and target, adding {MIN_TITLES:.0%}+{title} ==")
        best: dict[tuple, dict] = {}
        for r in everything:
            key = (r["opp"], r["get"])
            if keep(r) and (key not in best or r["titles"] > best[key]["titles"]):
                best[key] = r
        for r in sorted(best.values(), key=lambda r: -r["titles"])[:TOP_LEAGUE]:
            print(line(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

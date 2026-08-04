#!/usr/bin/env python3
"""Value over replacement for this league, with an optimal-drafter draft simulation.

    python3 rank_vor.py                    # pool.json -> rankings.json
    python3 rank_vor.py --report           # + validation/convergence summary on stderr
    python3 rank_vor.py --selftest         # verify the lineup solver against brute force
    python3 rank_vor.py --flat             # emit a bare JSON array instead of an object

Scope is this league and nothing else. The only value input is `points_3yr` from
`pool.json` (built by build_pool.py) — three-year points at 0.5/rec with a 0.5 TE
premium, which is this league's scoring. Draftsharks' 3D value is ignored entirely and
is not even carried into the pool: it is a provider-scaled ordinal that already bakes in
someone else's roster assumptions, and it is not in points, so it cannot be differenced
against a replacement level. Kickers and IDP are already dropped upstream because the
roster has no slot for them.

League (from README.md): 10 teams, 0.5 PPR + 0.5 TE premium, superflex. Starters are
1 QB / 2 RB / 3 WR / 1 TE / 2 W-R-T / 1 W-R-T-Q = 10. Then 15 bench and 4 rookie-only
taxi spots = 29 draftable roster spots = 29 rounds = 290 picks. Snake with a 3rd-round
reversal. My slot is 1.02.

WHY A SIMULATION IS NEEDED AT ALL
---------------------------------
Naive VOR picks a replacement rank per position out of thin air ("QB10, because 10 teams
start one QB"). That is wrong here in both directions: the superflex slot means well over
10 QBs start league-wide, and the two W-R-T slots mean the RB/WR/TE split is decided by
who is actually good, not by the slot names. The number of players at each position who
end up starting is an *outcome* of how the league drafts, so it has to be simulated, not
assumed.

That creates a circularity, which this script resolves as a fixed point:

    replacement levels -> what a player is worth to a team -> how the draft goes
                       -> how many of each position start -> replacement levels

Iteration 0 seeds replacement from a pure slot-assignment argument (assign the top of the
pool to the league's 100 starting slots optimally, replacement is the best player left
over). Then simulate, re-measure, repeat. The map is piecewise-constant — starter counts
are integers — so it cannot be damped to a point; it is iterated undamped until a starter
count vector repeats, and the levels are averaged over that limit cycle. `--report` prints
the trace and the cycle width.

HOW A SIMULATED TEAM VALUES A PLAYER
------------------------------------
Everything is in 3-year points above a baseline, never raw points, so that filling an empty
slot is never confused with a real gain. A team's value is

    V(roster) = sum over filled starting slots of (points - that slot's replacement level)
              + sum over bench players of  0.30 * 0.55^d * max(points - wire level, 0)

where d counts how many players that team already has at the same position. The value of a
player to that team is V(roster + player) - V(roster). Three choices in there each cost a
degenerate draft to learn, and are explained at their definitions:

  * Starting slots are priced against the *marginal-starter* level, the same baseline VOR
    reports, so the board and the simulated drafters cannot disagree. Pricing them off the
    wire instead made filling any empty slot look enormous and had the optimizer take a
    206-VOR QB at 1.02 over a 446-VOR TE sitting right there (`slot_replacement`).
  * A flex slot is priced at max over the positions it accepts, so a player is worth
    strictly less there than in his dedicated slot. That is where scarcity comes from,
    rather than being asserted.
  * Bench value is measured against the wire and floored at zero, discounted by depth at
    the player's own position. Unfloored, a near-zero depth weight makes a negative
    marginal vanish, so stacking a position becomes the least-bad move — one team drafted
    twenty tight ends (`team_value`).

Roster legality is a constraint, not a price: 1 QB / 2 RB / 3 WR / 1 TE come from slots
nothing else can cover, so once a team's remaining picks equal its unfilled mandatory
spots, its candidates narrow to what it still owes (`Draft.candidates`).

Picks are not myopic. Each team scores a candidate as

    value now  +  E[value of the best player still there at my next pick]

using survival probabilities from the number of intervening picks versus the candidate's
rank in the order players are actually coming off the board. This is what makes the
drafters "optimal" in the sense that matters for a board: they take the player who will not
be there later, not merely the highest number on the screen. It is a two-pick rollout with
an independence approximation across candidates, not equilibrium play — the honest name is
a strong greedy, and the two-pick horizon is the part most likely to understate how early a
truly scarce position gets attacked.

WHAT THE OTHER NINE TEAMS DO
----------------------------
They are pulled toward the source ADP (--market-weight, 0.5 by default; 0 recovers
all-drafters-fully-optimal). ADP never enters VOR. It is used only as the best available
*estimate of how this draft will actually run*, since that is what decides who is on the
board at my pick — and it is a fuzzy estimate rather than a biased one, because a true
10-team TE-premium superflex ADP does not exist and the closest thing on hand is a 12-team
ADP with no TE premium. Where this board and ADP disagree, that is measurement error in the
signal, not a detected opponent error: the other managers know their own scoring settings.
Do not read `adp_rank_delta` as "he will fall to me". See `market_value`.

A CAVEAT THE DATA IMPOSES, NOT THIS MODEL
-----------------------------------------
QB replacement lands around QB20-21 = ~820 points, which compresses elite QB value hard
(Josh Allen is ~+225). That follows from the provider projecting backup and rookie QBs at
starter-grade volume — Malik Willis at 273 points in year one, rookie Fernando Mendoza at
812 over three years. If those are not credible, QB replacement is overstated and every
elite QB is underrated here. The lever is the projections, not the ranking method.

WHAT COMES OUT
--------------
`vor` is the headline number and the sort key: 3-year points minus the converged
replacement level for that position. The replacement level is the (S_p + 1)-th best
player at position p in the pool, where S_p is how many players at p hold a starting slot
across all 10 teams in the simulated draft. So VOR answers "how many points over three
years does this player add versus the best guy any team could have had at his position
without spending a starting-caliber pick".

`--baseline wire` swaps in the best-undrafted player instead. It is offered for comparison
but is not a sane ranking: because the wire is picked clean, it values a QB at nearly his
full point total and returns a board whose top 18 are all quarterbacks, Matthew Stafford at
38 years old ahead of Ja'Marr Chase. The wire level is the right price for an empty slot and
for bench depth; it is the wrong yardstick for comparing players.

Two simulations are reported per player, and they answer different questions:
  * `sim_pick` is from the single deterministic draft, no noise.
  * `sim_adp` / `p_available_at_my_picks` come from --sims noisy drafts (Gumbel noise on
    the other teams' scores, --noise). Under deterministic play availability is 0 or 1,
    which tells you nothing about risk, so the noise band is what makes the columns
    usable at the table. It is also where the uncertainty in the ADP signal belongs.

Python stdlib only. Deterministic: every tie breaks on player_id and the RNG is seeded.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

# --- league definition (README.md) -------------------------------------------------

SCHEME = "half_ppr_te_premium_superflex"
HORIZON = "3yr"
POINTS_FIELD = f"points_{HORIZON}"  # the one value column in pool.json
POSITIONS = ("QB", "RB", "WR", "TE")

TEAMS = 10
MY_SLOT = 2  # 1.02
STARTING_SLOTS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 2, "SF": 1}
# Slots no other position can cover, so every roster must end up with at least these.
DEDICATED_SLOTS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}
BENCH_SLOTS = 15
TAXI_SLOTS = 4  # rookie-only
ROSTER_SLOTS = sum(STARTING_SLOTS.values()) + BENCH_SLOTS + TAXI_SLOTS  # 29
ROUNDS = ROSTER_SLOTS
TOTAL_PICKS = TEAMS * ROUNDS  # 290

# Most restrictive slot first. Replacement level is non-decreasing along each chain
# (a dedicated slot is always the cheapest place to put a player), which is what lets the
# greedy lineup solver be exact; --selftest checks that against brute force.
SLOT_CHAIN = {
    "QB": ("QB", "SF"),
    "RB": ("RB", "FLEX", "SF"),
    "WR": ("WR", "FLEX", "SF"),
    "TE": ("TE", "FLEX", "SF"),
}
SLOT_ELIGIBLE = {
    "QB": ("QB",),
    "RB": ("RB",),
    "WR": ("WR",),
    "TE": ("TE",),
    "FLEX": ("RB", "WR", "TE"),
    "SF": ("QB", "RB", "WR", "TE"),
}

# --- strategy knobs ---------------------------------------------------------------

DEPTH_BASE = 0.30  # a position's first backup is worth this fraction of his VOR
POSITION_DEPTH_DECAY = 0.55  # ...and each further body at the same position much less
SURVIVAL_SIGMA = 3.5  # softness of "will he last until my next pick"
LOOKAHEAD_PER_POS = 2  # candidates per position considered for the next pick
NOISE = 0.35  # Gumbel scale, as a fraction of the spread between a pick's candidates
MAX_ITERS = 24  # cap on fixed-point iterations before a cycle must have closed
MARKET_WEIGHT = 0.8  # how far the other nine teams are pulled toward the source ADP
BASELINES = ("marginal-starter", "wire")
SIMS = 200
SEED = 20260804


@dataclass(slots=True)
class Player:
    player_id: int
    name: str
    position: str
    team: str
    age: float | None
    bye_week: int | None
    is_rookie: bool
    points: int
    provider_adp: float | None
    pos_index: int = 0  # rank within position by points, 0-based
    vor_index: int = 0  # rank in the pool by current VOR, 0-based


# --- draft order ------------------------------------------------------------------


def draft_order(teams: int = TEAMS, rounds: int = ROUNDS) -> list[int]:
    """Slot (1-based) picking at each overall pick. Snake with a 3rd-round reversal.

    Round 1 forward, rounds 2 and 3 both reverse (that is the reversal), then the snake
    resumes: even rounds forward, odd rounds reverse. Pinned to the README's stated picks
    for slot 2 (1.02, 2.09, 3.09, 4.02, 5.09, 6.02, ..., 28.02, 29.09) in --report.
    """
    order: list[int] = []
    for rnd in range(1, rounds + 1):
        forward = rnd == 1 or (rnd >= 4 and rnd % 2 == 0)
        order.extend(range(1, teams + 1) if forward else range(teams, 0, -1))
    return order


def pick_label(pick_no: int, teams: int = TEAMS) -> str:
    """1-based overall pick number -> 'round.slot-in-round' as the draft room shows it."""
    rnd, idx = divmod(pick_no - 1, teams)
    return f"{rnd + 1}.{idx + 1:02d}"


def picks_for_slot(slot: int, order: list[int]) -> list[int]:
    return [i + 1 for i, s in enumerate(order) if s == slot]


# --- pool -------------------------------------------------------------------------


def load_pool(path: Path) -> tuple[list[Player], dict]:
    """Read pool.json: already filtered to QB/RB/WR/TE with a usable 3-year projection.

    build_pool.py does the filtering — positions with no roster slot, the source's zeros
    where a null belongs, and everything past rank 350 — so this only re-checks the
    invariants it promises rather than re-deriving them. The guards stay because a
    hand-edited or stale pool is the likeliest way this ever gets bad input.
    """
    raw = json.loads(path.read_text())
    players: list[Player] = []
    dropped = {"non_offense": 0, "zero_projection": []}
    for rec in raw["players"]:
        if rec["position"] not in POSITIONS:
            dropped["non_offense"] += 1
            continue
        points = rec[POINTS_FIELD]
        if not points or points <= 0:
            dropped["zero_projection"].append(rec["name"])
            continue
        players.append(
            Player(
                player_id=rec["player_id"],
                name=rec["name"],
                position=rec["position"],
                team=rec["team"],
                age=rec.get("age"),
                bye_week=rec.get("bye_week"),
                is_rookie=bool(rec.get("is_rookie")),
                points=points,
                provider_adp=rec.get("adp"),
            )
        )

    players.sort(key=lambda p: (-p.points, p.player_id))
    per_pos: dict[str, list[Player]] = {pos: [] for pos in POSITIONS}
    for p in players:
        p.pos_index = len(per_pos[p.position])
        per_pos[p.position].append(p)

    meta = {
        "source_file": str(path),
        "source_player_count": raw.get("player_count", len(raw["players"])),
        "source_of_pool": raw.get("source_file"),
        "pool_size": len(players),
        "by_position": {pos: len(v) for pos, v in per_pos.items()},
        "dropped_non_offense": dropped["non_offense"],
        "dropped_zero_projection": sorted(dropped["zero_projection"]),
    }
    return players, meta


def by_position(players: list[Player]) -> dict[str, list[Player]]:
    out: dict[str, list[Player]] = {pos: [] for pos in POSITIONS}
    for p in players:
        out[p.position].append(p)
    return out


# --- lineup solver ----------------------------------------------------------------


def slot_replacement(rep: dict[str, float]) -> dict[str, float]:
    """What each starting slot yields without spending a valuable pick on it.

    Built from the same baseline as VOR, deliberately, so that the board and the simulated
    drafters cannot disagree. Pricing these off the wire levels instead was tried and is
    wrong: the wire QB is worth 223, so filling an empty QB slot scored ~795 and the
    optimizer took Jayden Daniels (VOR 206) at 1.02 over Trey McBride (VOR 446) sitting
    right there. The flaw is the counterfactual — if you skip a QB early you do not end the
    draft with nothing at QB, you end it with a late-round QB, which is the marginal-starter
    level by construction. So that is the price of an empty slot.

    What this pricing does *not* capture is that filling the slot late still costs a pick;
    that is a roster-legality matter, handled by reserving picks in `Draft.candidates`
    rather than by distorting the value here.
    """
    return {slot: max(rep[pos] for pos in elig) for slot, elig in SLOT_ELIGIBLE.items()}


def lineup_surplus(
    roster: list[Player], slot_rep: dict[str, float]
) -> tuple[float, list[Player], dict[str, int]]:
    """Points above replacement from the optimal starting lineup, plus the bench.

    Greedy in descending points, each player taking the most restrictive slot still open.
    Exact because slot eligibility is laminar and replacement level is non-decreasing
    along every chain, so a player never gains by moving to a looser slot. A player whose
    surplus is already non-positive in the tightest open slot is benched and the slot is
    streamed (worth 0) rather than filled at a loss.
    """
    caps = dict(STARTING_SLOTS)
    surplus = 0.0
    bench: list[Player] = []
    started = {pos: 0 for pos in POSITIONS}
    for p in sorted(roster, key=lambda q: (-q.points, q.player_id)):
        placed = False
        for slot in SLOT_CHAIN[p.position]:
            if caps[slot] == 0:
                continue
            gain = p.points - slot_rep[slot]
            if gain <= 0:
                break
            caps[slot] -= 1
            surplus += gain
            started[p.position] += 1
            placed = True
            break
        if not placed:
            bench.append(p)
    return surplus, bench, started


def starting_positions(roster: list[Player]) -> list[str]:
    """Which positions fill the 10 slots when a team must field everyone it can.

    No surplus gate here: a team starts its best available body every week even when that
    body is below replacement. This is the measurement used for replacement levels, so
    gating it would undercount starters and bias replacement upward.
    """
    caps = dict(STARTING_SLOTS)
    out: list[str] = []
    for p in sorted(roster, key=lambda q: (-q.points, q.player_id)):
        for slot in SLOT_CHAIN[p.position]:
            if caps[slot]:
                caps[slot] -= 1
                out.append(p.position)
                break
    return out


def team_value(
    roster: list[Player], slot_rep: dict[str, float], depth_value: dict[int, float]
) -> float:
    """Starting-lineup surplus plus bench depth value. Every term here is load-bearing.

    Starters are priced against the marginal-starter baseline (via `slot_rep`) because that
    is what you would otherwise have in the slot. Bench players are priced against the
    *wire* — `depth_value` is max(points - wire level, 0) — because a backup's job is to
    beat what you would otherwise have to sign, and then discounted by how likely he ever
    plays.

    The discount is by depth at the player's own position, not by a running bench index: a
    team's fifth QB cannot start in a league with two QB-capable slots, however large his
    value looks. Using a raw bench index let one team draft ten QBs.

    `depth_value` is floored at zero, and that floor is essential rather than cosmetic. An
    earlier version used unfloored VOR to keep deep picks ordered, which quietly inverted
    the incentive: when a candidate's value is negative, a *smaller* depth weight makes the
    marginal impact closer to zero and therefore better, so stacking a position until its
    weight vanished became the optimal move. One team drafted twenty tight ends. Measuring
    against the wire instead keeps the quantity non-negative — so no perverse stacking —
    while staying informative deep into the draft, since nearly every drafted player clears
    the wire even when he is far below the marginal starter.
    """
    surplus, bench, started = lineup_surplus(roster, slot_rep)
    bench.sort(key=lambda p: (-p.points, p.player_id))
    seen = dict(started)
    for p in bench:
        depth = seen[p.position] - started[p.position]
        surplus += DEPTH_BASE * POSITION_DEPTH_DECAY**depth * depth_value[p.player_id]
        seen[p.position] += 1
    return surplus


# --- replacement levels -----------------------------------------------------------

def _apportion(means: dict[str, float], total: int) -> dict[str, int]:
    """Round fractional starter counts to integers that still sum to `total`.

    Averaging a limit cycle gives fractions like 20.5 RB starters. Rounding each position
    independently does not preserve the sum — it reported 101 starters for a 100-slot
    league — so allocate floors first and hand out the remainder to the largest fractional
    parts (Hare/largest-remainder).
    """
    floors = {k: int(math.floor(v)) for k, v in means.items()}
    leftover = total - sum(floors.values())
    order = sorted(means, key=lambda k: (-(means[k] - floors[k]), k))
    for k in order[:max(leftover, 0)]:
        floors[k] += 1
    return floors


def _rep_at_rank(pos_players: list[Player], rank: int) -> float:
    """Points of the rank-th best player at a position (1-based), clamped to the pool."""
    if not pos_players:
        return 0.0
    return float(pos_players[min(max(rank, 1), len(pos_players)) - 1].points)


def seed_replacement(players: list[Player]) -> dict[str, float]:
    """Iteration-0 replacement from pure slot counting, no draft behaviour assumed.

    Assign the top of the pool to the league's 100 starting slots the way a perfectly
    efficient market would — dedicated slots by positional rank, then the 20 flex slots
    and 10 superflex slots to the best players still eligible — and take the best player
    at each position who did not earn a slot.
    """
    pos = by_position(players)
    caps = {slot: n * TEAMS for slot, n in STARTING_SLOTS.items()}
    used = {p: 0 for p in POSITIONS}
    for p in players:  # already sorted by points desc
        for slot in SLOT_CHAIN[p.position]:
            if caps[slot]:
                caps[slot] -= 1
                used[p.position] += 1
                break
    return {k: _rep_at_rank(pos[k], used[k] + 1) for k in POSITIONS}


def replacement_from_draft(rosters: list[list[Player]], pos: dict[str, list[Player]]) -> tuple[dict[str, float], dict[str, int]]:
    """Replacement = the best player at each position who is not starting-caliber.

    Count how many players at each position hold a starting slot across all 10 simulated
    teams; the next best player at that position in the pool is the replacement level.
    """
    counts = {p: 0 for p in POSITIONS}
    for roster in rosters:
        for position in starting_positions(roster):
            counts[position] += 1
    rep = {k: _rep_at_rank(pos[k], counts[k] + 1) for k in POSITIONS}
    return rep, counts


def wire_replacement(draft: Draft, pos: dict[str, list[Player]]) -> dict[str, float]:
    """The best player at each position actually left undrafted — the free-agent baseline.

    Distinct from the starting-caliber baseline above, and much lower: 290 of 444 players
    get rostered, so the wire is picked clean. 29 QBs go in the simulated draft against 20
    QB starting slots, which means QB21 (the marginal starter) is somebody's backup, not a
    free add. Both numbers are reported because they answer different questions —
    `replacement.levels` asks "how much better is he than the worst player good enough to
    start in this league", `replacement.wire_levels` asks "how much better is he than what
    I could sign for nothing after the draft". The former is the conventional VBD baseline
    and the sort key here, because starting slots are the scarce resource a draft pick buys.
    """
    out: dict[str, float] = {}
    for position, players in pos.items():
        left = [p for p in players if p.player_id not in draft.taken]
        out[position] = float(left[0].points) if left else 0.0
    return out


# --- availability model -----------------------------------------------------------


class Fenwick:
    """Availability counter over the VOR-sorted pool, for O(log n) 'how many better'."""

    __slots__ = ("n", "tree")

    def __init__(self, n: int) -> None:
        self.n = n
        self.tree = [0] * (n + 1)

    def add(self, i: int, delta: int) -> None:
        i += 1
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i

    def prefix(self, i: int) -> int:
        """Count of set entries strictly before index i."""
        total = 0
        while i > 0:
            total += self.tree[i]
            i -= i & -i
        return total


def survival(better_available: int, gap: int) -> float:
    """P(a player is still there `gap` picks later), given how many rank above him.

    If drafters followed the VOR order exactly he lasts iff more than `gap` players rank
    above him. Softened into a logistic because they follow their own roster needs, not a
    single global list.
    """
    if gap <= 0:
        return 1.0
    return 1.0 / (1.0 + math.exp((gap - better_available) / SURVIVAL_SIGMA))


# --- draft simulation -------------------------------------------------------------


class Draft:
    """One simulated draft. `noise` > 0 perturbs the other teams' scores only."""

    def __init__(
        self,
        players: list[Player],
        rep: dict[str, float],
        vor: dict[int, float],
        order: list[int],
        wire: dict[str, float] | None = None,
        noise: float = 0.0,
        rng: random.Random | None = None,
        market_vor: dict[int, float] | None = None,
        market_weight: float = 0.0,
    ) -> None:
        self.players = players
        self.rep = rep
        self.slot_rep = slot_replacement(rep)
        self.vor = vor
        # Bench depth is measured against the wire, floored at zero. Before any draft exists
        # to read a wire off, fall back to the VOR baseline.
        self.wire = rep if wire is None else wire
        self.depth_value = {
            p.player_id: max(p.points - self.wire[p.position], 0.0) for p in players
        }
        self.order = order
        self.noise = noise
        self.rng = rng
        self.market_vor = market_vor or {}
        self.market_weight = market_weight if market_vor else 0.0
        # The nine other teams are market-influenced, so the order players actually come
        # off the board is the blended one — and that is what the survival model has to
        # follow, for me as much as for them. Predicting opponents correctly is the whole
        # point: it is what lets my picks exploit a TE the market is sleeping on.
        w = self.market_weight
        perceived = {
            p.player_id: (1 - w) * vor[p.player_id] + w * self.market_vor[p.player_id]
            if w
            else vor[p.player_id]
            for p in players
        }
        self.perceived = perceived
        # pool sorted by perceived value desc drives the survival model; per-position lists
        # sorted by points drive candidate generation (within a position, more points is
        # always at least as valuable, so only the head of each list can be the best pick).
        self.vor_sorted = sorted(players, key=lambda p: (-perceived[p.player_id], p.player_id))
        for i, p in enumerate(self.vor_sorted):
            p.vor_index = i
        self.avail_bits = Fenwick(len(players))
        for p in players:
            self.avail_bits.add(p.vor_index, 1)
        self.pos_lists = {
            pos: sorted(v, key=lambda p: (-p.points, p.player_id))
            for pos, v in by_position(players).items()
        }
        self.heads = {pos: 0 for pos in POSITIONS}
        self.taken: set[int] = set()
        self.rosters: list[list[Player]] = [[] for _ in range(TEAMS)]
        self.pick_of: dict[int, int] = {}
        self.next_pick = self._next_pick_table()
        self.picks_left = [ROUNDS] * TEAMS

    def _next_pick_table(self) -> list[int | None]:
        """For each pick index, the slot's following pick index (None if it is their last)."""
        nxt: list[int | None] = [None] * len(self.order)
        last_seen: dict[int, int] = {}
        for i in range(len(self.order) - 1, -1, -1):
            slot = self.order[i]
            nxt[i] = last_seen.get(slot)
            last_seen[slot] = i
        return nxt

    # -- availability helpers

    def _advance(self, pos: str) -> None:
        lst = self.pos_lists[pos]
        i = self.heads[pos]
        while i < len(lst) and lst[i].player_id in self.taken:
            i += 1
        self.heads[pos] = i

    def candidates(
        self, roster: list[Player], per_pos: int = 1, picks_left: int | None = None
    ) -> list[Player]:
        """Best available at each position, honouring taxi and roster-legality limits.

        A lineup needs 1 QB, 2 RB, 3 WR and 1 TE from positions that nothing else can
        cover, so a manager cannot spend every pick on the best name available and end up
        without a quarterback. Once the picks remaining are exactly the unfilled mandatory
        spots, candidates narrow to the positions still owed. This is the honest way to stop
        a team punting a position — the earlier attempt distorted the value of an empty slot
        instead, which broke the board.
        """
        eligible = POSITIONS
        if picks_left is not None:
            have = {pos: 0 for pos in POSITIONS}
            for p in roster:
                have[p.position] += 1
            owed = {pos: max(0, DEDICATED_SLOTS[pos] - have[pos]) for pos in POSITIONS}
            if picks_left <= sum(owed.values()):
                eligible = tuple(pos for pos in POSITIONS if owed[pos]) or POSITIONS
        taxi = len(roster) >= sum(STARTING_SLOTS.values()) + BENCH_SLOTS

        # Both restrictions can be simultaneously unsatisfiable — a team owed a QB in the
        # taxi rounds when none of the 9 rookie QBs are left, which used to assert out as
        # "pool exhausted". Every one of the 29 picks is mandatory, so relax in order of
        # what a manager would actually give up: taxi eligibility first (they can carry the
        # player on the bench and cut elsewhere), then the positional requirement.
        for positions, rookies_only in ((eligible, taxi), (eligible, False), (POSITIONS, False)):
            out: list[Player] = []
            for pos in positions:
                self._advance(pos)
                found = 0
                for p in self.pos_lists[pos][self.heads[pos] :]:
                    if p.player_id in self.taken:
                        continue
                    if rookies_only and not p.is_rookie:
                        continue
                    out.append(p)
                    found += 1
                    if found == per_pos:
                        break
            if out:
                return out
        return []

    def better_available(self, p: Player) -> int:
        return self.avail_bits.prefix(p.vor_index)

    # -- valuation

    def marginal(self, roster: list[Player], p: Player, base: float) -> float:
        return team_value(roster + [p], self.slot_rep, self.depth_value) - base

    def lookahead(self, roster: list[Player], taking: Player, gap: int, left: int) -> float:
        """E[value of the best player still available at this team's next pick].

        Order statistic over the plausible next-pick candidates: the best surviving one is
        candidate i if i survives and everyone better does not. Candidates are treated as
        independent, which slightly understates the chance that a whole position gets
        cleared out between picks.
        """
        if gap is None:
            return 0.0
        future_roster = roster + [taking]
        base = team_value(future_roster, self.slot_rep, self.depth_value)
        scored: list[tuple[float, float]] = []
        for cand in self.candidates(future_roster, per_pos=LOOKAHEAD_PER_POS, picks_left=left):
            if cand.player_id == taking.player_id:
                continue
            gain = self.marginal(future_roster, cand, base)
            scored.append((gain, survival(self.better_available(cand), gap)))
        scored.sort(key=lambda t: -t[0])
        expected = 0.0
        mass = 1.0
        for gain, surv in scored:
            expected += mass * surv * gain
            mass *= 1.0 - surv
            if mass < 1e-4:
                break
        return expected

    def choose(self, pick_index: int, slot: int) -> Player:
        roster = self.rosters[slot - 1]
        base = team_value(roster, self.slot_rep, self.depth_value)
        nxt = self.next_pick[pick_index]
        gap = None if nxt is None else nxt - pick_index - 1
        # I draft on my own board; the other nine are pulled toward the market's ordering.
        # Blended at the decision, not inside the valuation, so their roster logic stays
        # intact — they still fill needs, they just rank players closer to consensus. At
        # market_weight 1.0 they are pure best-available-by-ADP drafters.
        w = self.market_weight if slot != MY_SLOT else 0.0
        left = self.picks_left[slot - 1]
        scored: list[tuple[float, Player]] = []
        for cand in self.candidates(roster, per_pos=1, picks_left=left):
            score = self.marginal(roster, cand, base)
            if gap is not None:
                score += self.lookahead(roster, cand, gap, left - 1)
            if w:
                score = (1 - w) * score + w * self.market_vor[cand.player_id]
            scored.append((score, cand))
        assert scored, "pool exhausted"

        if self.noise and self.rng is not None and slot != MY_SLOT:
            # Gumbel noise -> the other nine teams follow a softmax over their own scores
            # instead of a strict argmax, which is what turns 0/1 availability under
            # deterministic play into a usable probability band. Scaled to the spread
            # between this pick's candidates, so it expresses "the ordering among
            # near-equals is uncertain" rather than a fixed number of points; a pick with
            # one clear best option stays nearly deterministic.
            spread = max(s for s, _ in scored) - min(s for s, _ in scored)
            scale = self.noise * max(spread, 1.0)
            scored = [
                (s - scale * math.log(-math.log(self.rng.random())), c) for s, c in scored
            ]

        return max(scored, key=lambda t: (t[0], -t[1].player_id))[1]

    def run(self) -> None:
        for i, slot in enumerate(self.order):
            pick = self.choose(i, slot)
            self.taken.add(pick.player_id)
            self.avail_bits.add(pick.vor_index, -1)
            self.rosters[slot - 1].append(pick)
            self.picks_left[slot - 1] -= 1
            self.pick_of[pick.player_id] = i + 1


def compute_vor(players: list[Player], rep: dict[str, float]) -> dict[int, float]:
    return {p.player_id: p.points - rep[p.position] for p in players}


def market_value(players: list[Player], vor: dict[int, float]) -> dict[int, float]:
    """Quantile-map the market's ADP ordering onto the VOR scale.

    The source ADP cannot be used as a value directly — it is a pick number, and per README
    it is a *12-team, no-TE-premium* superflex ADP, so neither its scale nor its scoring
    matches this league. What it does carry is a credible *ordering* of how the field ranks
    players. So take that ordering and pour it into the shape of the VOR distribution: the
    player the market likes best is assigned the top VOR value, the second-best the second,
    and so on. Rank-based, so the 12-team pick numbering cancels out and the raw ADP values
    (which run continuously into the thousands rather than stopping at a clean unranked
    sentinel) never need interpreting as real picks.

    This is the best available estimate of how the draft will actually run, not a claim that
    the field is wrong. The other managers know their own scoring; a true 10-team TE-premium
    superflex ADP simply does not exist to be had. So where this ordering diverges from the
    board — TEs above all — treat it as a gap in the signal rather than an opponent error,
    and let --noise carry that uncertainty. What it must never do is feed back into VOR
    itself; it only shapes how the opponents behave.
    """
    by_adp = sorted(
        players,
        key=lambda p: (p.provider_adp if p.provider_adp is not None else math.inf, p.player_id),
    )
    scale = sorted((vor[p.player_id] for p in players), reverse=True)
    return {p.player_id: scale[i] for i, p in enumerate(by_adp)}


def converge(
    players: list[Player],
    order: list[int],
    max_iters: int,
    report: bool,
    market_weight: float = 0.0,
    baseline: str = "marginal-starter",
) -> tuple[dict[str, float], dict[str, float], dict[str, int], Draft, dict]:
    """Fixed point: replacement -> valuation -> draft -> starter counts -> replacement.

    The map is piecewise-constant: starter counts are integers, so the replacement level
    jumps between adjacent players in the pool (RB21 and RB22 are 18 points apart) instead
    of moving continuously. That means gradient-style damping cannot settle it — it just
    orbits the discontinuity — while plain undamped iteration is eventually periodic.

    So iterate undamped, watch for a repeated starter-count vector, and average the
    replacement levels over the cycle once one closes. A cycle of length 1 is an exact
    fixed point; a longer cycle means the draft genuinely alternates between neighbouring
    league shapes (e.g. 20 vs 21 RBs starting) and the average across it is the honest
    answer. The cycle is reported rather than hidden.

    Two levels come out of each draft and both are iterated: the VOR baseline selected by
    --baseline, and the wire level that prices an empty starting slot. They are separate
    quantities and must not be collapsed into one (see `slot_replacement`).
    """
    pos = by_position(players)
    rep = seed_replacement(players)
    stream = dict(rep)  # no draft to read a wire off yet
    trace = [{"iteration": 0, "source": "slot_assignment", "replacement": dict(rep)}]
    seen: dict[tuple[int, ...], int] = {}
    observations: list[dict[str, float]] = []
    wire_observations: list[dict[str, float]] = []
    counts_by_iter: list[dict[str, int]] = []
    cycle_start: int | None = None

    for it in range(1, max_iters + 1):
        vor = compute_vor(players, rep)
        mkt = market_value(players, vor) if market_weight else None
        draft = Draft(
            players, rep, vor, order, wire=stream,
            market_vor=mkt, market_weight=market_weight,
        )
        draft.run()
        starter_rep, counts = replacement_from_draft(draft.rosters, pos)
        wire = wire_replacement(draft, pos)
        observed = starter_rep if baseline == "marginal-starter" else wire
        observations.append(observed)
        wire_observations.append(wire)
        counts_by_iter.append(counts)
        key = tuple(counts[k] for k in POSITIONS)
        trace.append(
            {
                "iteration": it,
                "source": "draft_simulation",
                "starters_by_position": dict(counts),
                "observed_replacement": {k: round(v, 1) for k, v in observed.items()},
            }
        )
        if report:
            print(
                f"  iter {it}: starters={counts} observed rep="
                + ", ".join(f"{k} {observed[k]:.0f}" for k in POSITIONS),
                file=sys.stderr,
            )
        if key in seen:
            cycle_start = seen[key]
            break
        seen[key] = it
        rep = observed
        stream = wire

    assert cycle_start is not None, f"no cycle within {max_iters} iterations"
    cycle = observations[cycle_start - 1 : -1] or observations[cycle_start - 1 :]
    wire_cycle = wire_observations[cycle_start - 1 : -1] or wire_observations[cycle_start - 1 :]
    cycle_counts = counts_by_iter[cycle_start - 1 : len(observations) - 1] or counts_by_iter[-1:]
    rep = {k: sum(o[k] for o in cycle) / len(cycle) for k in POSITIONS}
    stream = {k: sum(o[k] for o in wire_cycle) / len(wire_cycle) for k in POSITIONS}
    counts = _apportion(
        {k: sum(c[k] for c in cycle_counts) / len(cycle_counts) for k in POSITIONS},
        TEAMS * sum(STARTING_SLOTS.values()),
    )
    if report:
        print(
            f"  cycle of length {len(cycle)} closed at iteration {cycle_start}; "
            "averaging replacement across it",
            file=sys.stderr,
        )

    # Final deterministic draft at the settled levels, so sim_pick and the reported
    # replacement levels describe one and the same draft.
    vor = compute_vor(players, rep)
    mkt = market_value(players, vor) if market_weight else None
    draft = Draft(
        players, rep, vor, order, wire=stream,
        market_vor=mkt, market_weight=market_weight,
    )
    draft.run()
    _, final_counts = replacement_from_draft(draft.rosters, pos)
    history = {
        "baseline": baseline,
        "market_weight": market_weight,
        "method": "undamped iteration to a limit cycle, averaged over the cycle",
        "cycle_length": len(cycle),
        "cycle_first_seen_at_iteration": cycle_start,
        "iterations_run": len(observations),
        "starters_in_final_draft": final_counts,
        # Spread of the cycle the levels were averaged over. A wide band means the league
        # shape genuinely wobbles between neighbouring configurations rather than settling.
        "cycle_replacement_range": {
            k: [round(min(o[k] for o in cycle), 1), round(max(o[k] for o in cycle), 1)]
            for k in POSITIONS
        },
        "trace": trace,
    }
    return rep, stream, counts, draft, history


def monte_carlo(
    players: list[Player],
    rep: dict[str, float],
    order: list[int],
    stream: dict[str, float],
    sims: int,
    noise: float,
    seed: int,
    market_weight: float = 0.0,
) -> tuple[dict[int, list[int]], dict[int, int]]:
    """Noisy redraws -> pick distribution per player. Returns picks seen and draft counts."""
    vor = compute_vor(players, rep)
    mkt = market_value(players, vor) if market_weight else None
    picks: dict[int, list[int]] = {p.player_id: [] for p in players}
    drafted: dict[int, int] = {p.player_id: 0 for p in players}
    for s in range(sims):
        rng = random.Random(seed + s)
        d = Draft(
            players, rep, vor, order, wire=stream, noise=noise, rng=rng,
            market_vor=mkt, market_weight=market_weight,
        )
        d.run()
        for pid, pick in d.pick_of.items():
            picks[pid].append(pick)
            drafted[pid] += 1
    return picks, drafted


# --- output -----------------------------------------------------------------------


def build_rankings(
    players: list[Player],
    rep: dict[str, float],
    wire: dict[str, float],
    draft: Draft,
    picks: dict[int, list[int]],
    drafted: dict[int, int],
    sims: int,
    my_picks: list[int],
) -> list[dict]:
    vor = compute_vor(players, rep)
    ranked = sorted(players, key=lambda p: (-vor[p.player_id], p.player_id))
    market_rank = {
        p.player_id: i
        for i, p in enumerate(
            sorted(
                players,
                key=lambda p: (
                    p.provider_adp if p.provider_adp is not None else math.inf,
                    p.player_id,
                ),
            ),
            start=1,
        )
    }
    pos_rank: dict[str, int] = {pos: 0 for pos in POSITIONS}
    rows: list[dict] = []
    for i, p in enumerate(ranked, start=1):
        pos_rank[p.position] += 1
        seen = picks[p.player_id]
        # Availability at each of my picks, across the noisy redraws. Only the uncertain
        # ones are emitted; a 0.0 or 1.0 entry carries no information for a draft board.
        # `latest` is the actionable one — how long I can wait on him — so it is the last
        # pick still at even odds, not the first (which is trivially 1.02 for everybody
        # except whoever goes 1.01).
        avail = {}
        latest = None
        for pick in my_picks:
            # `>=`, not `>`: a player taken *at* one of my picks was taken by me, so he was
            # on the board when I got there. Using `>` reported the board's top TE as
            # unavailable at 1.02 in exactly the sims where he was my 1.02 pick.
            prob = (sum(1 for s in seen if s >= pick) + (sims - len(seen))) / sims
            if 0.01 < prob < 0.99:
                avail[pick_label(pick)] = round(prob, 3)
            if prob >= 0.5:
                latest = pick_label(pick)
        sim_pick = draft.pick_of.get(p.player_id)
        mean = sum(seen) / len(seen) if seen else None
        sd = (
            math.sqrt(sum((x - mean) ** 2 for x in seen) / len(seen))
            if seen and len(seen) > 1
            else None
        )
        rows.append(
            {
                "vor_rank": i,
                "player_id": p.player_id,
                "name": p.name,
                "position": p.position,
                "positional_vor_rank": pos_rank[p.position],
                "team": p.team,
                "age": p.age,
                "bye_week": p.bye_week,
                "is_rookie": p.is_rookie,
                "points_3yr": p.points,
                "replacement_points": round(rep[p.position], 1),
                "vor": round(vor[p.player_id], 1),
                "vor_vs_wire": round(p.points - wire[p.position], 1),
                "sim_pick": sim_pick,
                "sim_pick_label": pick_label(sim_pick) if sim_pick else None,
                "sim_adp": round(mean, 1) if mean is not None else None,
                "sim_adp_sd": round(sd, 1) if sd is not None else None,
                "p_drafted": round(drafted[p.player_id] / sims, 3),
                "latest_my_pick_likely_available": latest,
                "p_available_at_my_picks": avail,
                "provider_adp": p.provider_adp,
                "market_rank": market_rank[p.player_id],
                # Divergence from a fuzzy proxy, not a measured mispricing. The source ADP
                # is a 12-team no-TE-premium board, so a large delta at TE mostly reflects
                # what the signal cannot see rather than what opponents will do.
                "adp_rank_delta": market_rank[p.player_id] - i,
            }
        )
    return rows


def validate(
    rows: list[dict],
    players: list[Player],
    rep: dict[str, float],
    counts: dict[str, int],
    draft: Draft,
    order: list[int],
    history: list[dict],
    my_picks: list[int],
) -> list[str]:
    problems: list[str] = []

    def check(ok: bool, msg: str) -> None:
        if not ok:
            problems.append(msg)

    readme = ["1.02", "2.09", "3.09", "4.02", "5.09", "6.02", "28.02", "29.09"]
    labels = [pick_label(p) for p in my_picks]
    check(labels[:6] == readme[:6], f"draft order head {labels[:6]} != README {readme[:6]}")
    check(labels[-2:] == readme[-2:], f"draft order tail {labels[-2:]} != README {readme[-2:]}")
    check(len(my_picks) == ROUNDS, f"{len(my_picks)} picks for slot {MY_SLOT}, want {ROUNDS}")
    check(len(order) == TOTAL_PICKS, f"{len(order)} picks total, want {TOTAL_PICKS}")

    check(
        sum(counts.values()) == TEAMS * sum(STARTING_SLOTS.values()),
        f"starters sum to {sum(counts.values())}, want {TEAMS * sum(STARTING_SLOTS.values())}",
    )
    check(len(draft.taken) == TOTAL_PICKS, f"{len(draft.taken)} unique players drafted")
    sizes = {len(r) for r in draft.rosters}
    check(sizes == {ROUNDS}, f"roster sizes {sizes}, want {{{ROUNDS}}}")
    for i, roster in enumerate(draft.rosters, start=1):
        # The real constraint is the 25 non-taxi spots; which specific picks land on taxi is
        # not fixed, and a mandatory pick may have to be a non-rookie when no rookie at a
        # required position is left.
        non_rookies = sum(1 for p in roster if not p.is_rookie)
        check(
            non_rookies <= sum(STARTING_SLOTS.values()) + BENCH_SLOTS,
            f"slot {i} holds {non_rookies} non-rookies, over the "
            f"{sum(STARTING_SLOTS.values()) + BENCH_SLOTS} non-taxi spots",
        )
        starters = starting_positions(roster)
        check(
            len(starters) == sum(STARTING_SLOTS.values()),
            f"slot {i} cannot field a full lineup ({len(starters)}/10)",
        )
        for pos, need in DEDICATED_SLOTS.items():
            have = sum(1 for p in roster if p.position == pos)
            check(have >= need, f"slot {i} has {have} {pos}, needs {need}")

    vors = [r["vor"] for r in rows]
    check(vors == sorted(vors, reverse=True), "rows are not sorted by VOR descending")
    check(len(rows) == len(players), f"{len(rows)} rows for {len(players)} pool players")
    # The reported level is a mean over the limit cycle, so the invariant that actually
    # holds is that it lies inside the range the cycle spanned — not that it coincides with
    # any single draft's level, which a long cycle can genuinely straddle.
    pos = by_position(players)
    for k in POSITIONS:
        lo, hi = history["cycle_replacement_range"][k]
        check(
            lo - 1e-6 <= rep[k] <= hi + 1e-6,
            f"{k} replacement {rep[k]:.1f} outside its cycle range [{lo}, {hi}]",
        )
        check(
            pos[k][-1].points <= rep[k] <= pos[k][0].points,
            f"{k} replacement {rep[k]:.1f} outside the {k} pool range",
        )
    check(
        history["iterations_run"] < MAX_ITERS,
        f"no limit cycle found within {MAX_ITERS} iterations",
    )
    return problems


# --- selftest ---------------------------------------------------------------------


def brute_force_surplus(roster: list[Player], slot_rep: dict[str, float]) -> float:
    """Exhaustive best assignment of a small roster to slots, for --selftest."""
    slots = [s for s, n in STARTING_SLOTS.items() for _ in range(n)]
    best = 0.0
    for combo in itertools.product(range(len(slots) + 1), repeat=len(roster)):
        used: set[int] = set()
        total = 0.0
        ok = True
        for p, s in zip(roster, combo):
            if s == len(slots):  # benched
                continue
            if s in used or p.position not in SLOT_ELIGIBLE[slots[s]]:
                ok = False
                break
            used.add(s)
            total += p.points - slot_rep[slots[s]]
        if ok:
            best = max(best, total)
    return best


def selftest(players: list[Player], trials: int = 300, seed: int = SEED) -> int:
    """The greedy lineup solver must match brute force on random rosters."""
    rng = random.Random(seed)
    rep = seed_replacement(players)
    slot_rep = slot_replacement(rep)
    worst = 0.0
    for _ in range(trials):
        roster = rng.sample(players, rng.randint(1, 6))
        greedy, _, _ = lineup_surplus(roster, slot_rep)
        exact = brute_force_surplus(roster, slot_rep)
        worst = max(worst, exact - greedy)
        if exact - greedy > 1e-6:
            print(
                "MISMATCH: "
                + ", ".join(f"{p.name}({p.position},{p.points})" for p in roster)
                + f"  greedy={greedy:.1f} exact={exact:.1f}",
                file=sys.stderr,
            )
            return 1
    print(
        f"selftest ok: greedy lineup solver matched brute force on {trials} random "
        f"rosters (max shortfall {worst:.2e})",
        file=sys.stderr,
    )
    return 0


# --- cli --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("input", nargs="?", default=Path("pool.json"), type=Path)
    ap.add_argument("-o", "--output", default=Path("rankings.json"), type=Path)
    ap.add_argument("--report", action="store_true", help="validation summary on stderr")
    ap.add_argument("--selftest", action="store_true", help="check the lineup solver, then exit")
    ap.add_argument("--flat", action="store_true", help="emit a bare array, no metadata")
    ap.add_argument("--sims", type=int, default=SIMS, help="noisy redraws for availability")
    ap.add_argument("--noise", type=float, default=NOISE, help="other teams' Gumbel scale")
    ap.add_argument(
        "--market-weight",
        type=float,
        default=MARKET_WEIGHT,
        help="how far the other nine teams follow the source ADP instead of their own "
        "board: 0 = all drafters fully optimal, 1 = pure best-available-by-ADP",
    )
    ap.add_argument(
        "--baseline",
        choices=BASELINES,
        default="marginal-starter",
        help="replacement level: 'marginal-starter' (worst startable player league-wide, "
        "the conventional VBD baseline) or 'wire' (best player left undrafted)",
    )
    ap.add_argument("--max-iters", type=int, default=MAX_ITERS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--indent", type=int, default=2, help="JSON indent; 0 for compact")
    args = ap.parse_args(argv)

    if not args.input.exists():
        print(f"missing {args.input} - run pipeline.py first", file=sys.stderr)
        return 1

    players, pool_meta = load_pool(args.input)
    if args.selftest:
        return selftest(players)

    order = draft_order()
    my_picks = picks_for_slot(MY_SLOT, order)

    if args.report:
        print(
            f"pool: {len(players)} players "
            + ", ".join(f"{k} {v}" for k, v in pool_meta["by_position"].items())
            + f"; dropped {pool_meta['dropped_non_offense']} non-offense and "
            f"{len(pool_meta['dropped_zero_projection'])} zero-projection",
            file=sys.stderr,
        )
        print("converging replacement levels:", file=sys.stderr)

    rep, stream, counts, draft, history = converge(
        players, order, args.max_iters, args.report, args.market_weight, args.baseline
    )

    if args.report:
        print(
            f"monte carlo: {args.sims} noisy drafts (noise={args.noise}, "
            f"market_weight={args.market_weight})",
            file=sys.stderr,
        )
    picks, drafted = monte_carlo(
        players, rep, order, stream, args.sims, args.noise, args.seed, args.market_weight
    )
    rows = build_rankings(players, rep, stream, draft, picks, drafted, args.sims, my_picks)

    problems = validate(rows, players, rep, counts, draft, order, history, my_picks)

    payload: object
    if args.flat:
        payload = rows
    else:
        payload = {
            "generated_from": pool_meta["source_file"],
            "scoring_scheme": SCHEME,
            "value_input": f"pool.json {POINTS_FIELD} ({SCHEME})",
            "value_note": (
                "3-year projected points in this league's scheme. Draftsharks' 3D value is "
                "deliberately unused: it is a provider-scaled ordinal, not points, so it "
                "cannot be differenced against a replacement level."
            ),
            "league": {
                "teams": TEAMS,
                "starting_slots": STARTING_SLOTS,
                "bench_slots": BENCH_SLOTS,
                "taxi_slots": TAXI_SLOTS,
                "rounds": ROUNDS,
                "total_picks": TOTAL_PICKS,
                "draft_type": "snake with 3rd-round reversal",
                "my_slot": MY_SLOT,
                "my_picks": [pick_label(p) for p in my_picks],
            },
            "pool": pool_meta,
            "market_model": {
                "market_weight": MARKET_WEIGHT,
                "who": "the other nine teams; my slot always drafts on this board",
                "how": (
                    "Each opposing pick scores candidates as (1 - w) * its own optimal value "
                    "+ w * the market's implied value, where the market's implied value comes "
                    "from pouring the ADP *ordering* into the shape of the VOR distribution "
                    "(rank-based, so the 12-team pick numbering cancels). Blended at the "
                    "decision rather than inside the valuation, so opponents still fill "
                    "roster needs — they just rank players closer to consensus."
                ),
                "why": (
                    "ADP never touches VOR, and it is not treated as the right answer. It is "
                    "the best available *estimate of how this draft will actually behave*, "
                    "which is what decides who is still on the board at my pick. It is a "
                    "fuzzy estimate, not a biased one: a true 10-team TE-premium superflex ADP "
                    "does not exist, so the closest thing on hand is a 12-team ADP with no TE "
                    "premium. Set --market-weight 0 to recover the all-drafters-optimal "
                    "assumption."
                ),
                "what_this_is_not": (
                    "This is NOT an assumption that the other nine managers will misprice "
                    "tight ends. They know their own scoring settings and will price the "
                    "premium themselves; we simply cannot observe how. So the TE gap between "
                    "this board and `market_rank` is measurement error in our signal, not a "
                    "detected edge, and it should not be read as 'elite TEs will fall to me'. "
                    "Where the signal is weakest is where --noise, not confidence, belongs."
                ),
                "adp_note": (
                    "`provider_adp` is passthrough reference only — an overall pick number "
                    "in the source's 12-team draft. Its values run continuously into the "
                    "thousands rather than stopping at a clean unranked sentinel, which is "
                    "why only its rank is ever used. "
                    "`market_rank` is that rank; `adp_rank_delta` is market_rank - vor_rank, "
                    "a divergence between this board and a fuzzy proxy — positive means the "
                    "proxy ranks him later than this board does, which is a prompt to check "
                    "why, not a free win."
                ),
            },
            "replacement": {
                "definition": (
                    "Points of the (S+1)-th best player at a position, where S is how many "
                    "players at that position hold a starting slot across all 10 teams in "
                    "the simulated draft. This is the marginal starter, i.e. the worst "
                    "player still good enough to start somewhere in this league."
                ),
                "levels": {k: round(v, 1) for k, v in rep.items()},
                "starters_by_position": counts,
                "marginal_starter": {
                    k: {
                        "rank": f"{k}{counts[k] + 1}",
                        "player": by_position(players)[k][
                            min(counts[k], len(by_position(players)[k]) - 1)
                        ].name,
                    }
                    for k in POSITIONS
                },
                "wire_levels": {k: round(v, 1) for k, v in stream.items()},
                "wire_note": (
                    "The best player at each position left undrafted — the post-draft free "
                    "agent baseline, used by `vor_vs_wire` and, importantly, to price empty "
                    "starting slots inside the simulation. Far below the marginal-starter "
                    "level because 290 of 444 pool players get rostered."
                ),
                "slot_levels": {k: round(v, 1) for k, v in slot_replacement(stream).items()},
                "slot_levels_note": (
                    "What each starting slot pays if left empty and streamed, built from the "
                    "wire levels: a flex slot is max(wire_RB, wire_WR, wire_TE) and the "
                    "superflex slot the max over all four, so a player is worth strictly less "
                    "in a flex slot than in his dedicated one. That is where positional "
                    "scarcity comes from. These are deliberately NOT the marginal-starter "
                    "levels — pricing an empty QB slot at the marginal starting QB let a team "
                    "punt the position, collect the points anyway, and finish unable to field "
                    "a lineup."
                ),
                "convergence": history,
            },
            "strategy": {
                "objective": "starting-lineup points above replacement + decayed bench VOR",
                "depth_base": DEPTH_BASE,
                "position_depth_decay": POSITION_DEPTH_DECAY,
                "depth_note": (
                    "Bench value decays with how many players that team already has at the "
                    "same position, not with a running bench index — a fifth QB cannot start "
                    "in a two-QB-slot league however large his VOR looks."
                ),
                "lookahead": "value now + E[best value still available at my next pick]",
                "survival_sigma": SURVIVAL_SIGMA,
                "note": (
                    "Two-pick rollout with an independence approximation across candidates; "
                    "a strong greedy, not equilibrium play."
                ),
            },
            "monte_carlo": {
                "sims": args.sims,
                "noise": args.noise,
                "seed": args.seed,
                "note": (
                    "Gumbel noise on the other 9 teams only, to turn 0/1 availability under "
                    "deterministic play into a usable probability band. sim_pick is from the "
                    "noiseless draft; sim_adp and p_available_at_my_picks are from these."
                ),
            },
            "validation": {"problems": problems, "ok": not problems},
            "count": len(rows),
            "rankings": rows,
        }

    text = json.dumps(payload, indent=args.indent or None, ensure_ascii=False)
    args.output.write_text(text + "\n")

    if args.report:
        top = rows[:12]
        width = max(len(r["name"]) for r in top)
        print("\ntop of the board:", file=sys.stderr)
        for r in top:
            print(
                f"  {r['vor_rank']:>3}. {r['name']:<{width}}  {r['position']}"
                f"{r['positional_vor_rank']:<3} vor {r['vor']:>7.1f}"
                f"  pts {r['points_3yr']:>5}  sim {r['sim_pick_label'] or '--':>6}"
                f"  provider adp {r['provider_adp'] or float('nan'):>5}",
                file=sys.stderr,
            )
        print(
            "\nreplacement: "
            + ", ".join(f"{k}{counts[k] + 1} = {rep[k]:.0f}" for k in POSITIONS),
            file=sys.stderr,
        )
        if problems:
            print(f"\n{len(problems)} VALIDATION PROBLEM(S):", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
        else:
            print("\nvalidation: all checks passed", file=sys.stderr)

    print(f"wrote {args.output} ({len(rows)} players)", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

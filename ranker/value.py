"""What a roster is worth, and where replacement levels come from.

Everything is in 3-year points above a baseline, never raw points, so that filling an
empty slot is never confused with a real gain. A team's value is

    V(roster) = sum over filled starting slots of (points - that slot's replacement level)
              + sum over bench players of  0.55^d * (   0.30 * max(upside - wire, 0)
                                                     + ins[pos] * max(points - wire, 0) )

The bench term prices a backup's two jobs, both against the wire. Growth: `upside` is his
3-year total re-projected at his years-2-3 pace (`upside_points`), so a backloaded rookie
outranks a flat veteran with the same 3-year sum. Insurance: his full 3-year sum — year 1
included — weighted by `INSURANCE_BASE`, his position's expected share of starter games
missed (byes + injuries), which is what a startable veteran backup is for. The two jobs
cover different weeks, so they add rather than compete.

d counts the bench players that team already has at the same position. The value of
a player to a team is V(roster + player) - V(roster). Three choices in there each cost a
degenerate draft to learn, and are explained at their definitions:

  * Starting slots are priced against the *marginal-starter* level, the same baseline VOR
    reports, so the board and the simulated drafters cannot disagree (`slot_replacement`).
  * A flex slot is priced at max over the positions it accepts, so a player is worth
    strictly less there than in his dedicated slot. That is where scarcity comes from,
    rather than being asserted.
  * Bench value is measured against the wire and floored at zero, discounted by depth at
    the player's own position (`team_value`).

Roster legality is a constraint, not a price: 1 QB / 2 RB / 3 WR / 1 TE come from slots
nothing else can cover, so once a team's remaining picks equal its unfilled mandatory
spots, its candidates narrow to what it still owes (`Draft.candidates` in sim.py).
"""

from __future__ import annotations

import math

from .league import (
    POSITION_DEPTH_DECAY,
    POSITIONS,
    SLOT_CHAIN,
    SLOT_ELIGIBLE,
    STARTING_SLOTS,
    TEAMS,
)
from .pool import Player, by_position

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
    *wire* — `depth_value` is the growth + insurance value built in `Draft.__init__`, both
    terms measured over the wire level — because a backup's job is to beat what you would
    otherwise have to sign, and then discounted by how likely he ever plays.

    The discount is by depth at the player's own position, not by a running bench index: a
    team's fifth QB cannot start in a league with two QB-capable slots, however large his
    value looks. Using a raw bench index let one team draft ten QBs.

    Each `depth_value` term is floored at zero, and that floor is essential rather than
    cosmetic. An
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
        surplus += POSITION_DEPTH_DECAY**depth * depth_value[p.player_id]
        seen[p.position] += 1
    return surplus


def compute_vor(players: list[Player], rep: dict[str, float]) -> dict[int, float]:
    return {p.player_id: p.points - rep[p.position] for p in players}


def upside_points(p: Player) -> float:
    """Bench players are priced on their years-2-3 pace, not their 3-year sum.

    With a full starter squad, a bench pick's year-1 points are close to worthless — his
    job is to grow past a starter in years 2-3. So a rookie projected 0/100/200 must
    outrank a veteran projected 100/100/100 on the bench, even though their 3-year sums
    tie. The provider publishes cumulative 1- and 3-year totals, so the years-2-3 pace
    is (points - points_1yr) / 2, and re-projecting that pace over the whole 3-year
    horizon keeps the quantity on the scale of the wire level it is differenced against:
    a perfectly flat scorer's upside points equal his `points` exactly, so only the
    growth shape moves the number — backloaded players up, declining veterans down.

    Starters are untouched: they play all three years, so their 3-year sum is already
    the right price. The source data guarantees points_1yr <= points (the horizons are
    cumulative), so this is never negative.
    """
    return 1.5 * (p.points - p.points_1yr)


# --- replacement levels -----------------------------------------------------------


def apportion(means: dict[str, float], total: int) -> dict[str, int]:
    """Round fractional starter counts to integers that still sum to `total`.

    Averaging a limit cycle gives fractions like 20.5 RB starters. Rounding each position
    independently does not preserve the sum — it reported 101 starters for a 100-slot
    league — so allocate floors first and hand out the remainder to the largest fractional
    parts (Hare/largest-remainder).
    """
    floors = {k: int(math.floor(v)) for k, v in means.items()}
    leftover = total - sum(floors.values())
    order = sorted(means, key=lambda k: (-(means[k] - floors[k]), k))
    for k in order[: max(leftover, 0)]:
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


def replacement_from_draft(
    rosters: list[list[Player]], pos: dict[str, list[Player]]
) -> tuple[dict[str, float], dict[str, int]]:
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


def wire_replacement(taken: set[int], pos: dict[str, list[Player]]) -> dict[str, float]:
    """The best player at each position actually left undrafted — the free-agent baseline.

    Distinct from the starting-caliber baseline above, and much lower: 290 of 350 pool
    players get rostered, so the wire is picked clean. 29 QBs go in the simulated draft
    against 20 QB starting slots, which means QB21 (the marginal starter) is somebody's
    backup, not a free add. Both numbers are reported because they answer different
    questions — `replacement.levels` asks "how much better is he than the worst player
    good enough to start in this league", `replacement.wire_levels` asks "how much better
    is he than what I could sign for nothing after the draft". The former is the
    conventional VBD baseline and the sort key here, because starting slots are the scarce
    resource a draft pick buys; the latter prices bench depth (`team_value`).
    """
    out: dict[str, float] = {}
    for position, players in pos.items():
        left = [p for p in players if p.player_id not in taken]
        out[position] = float(left[0].points) if left else 0.0
    return out

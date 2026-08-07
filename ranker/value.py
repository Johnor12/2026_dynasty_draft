"""What a roster is worth, and where replacement levels come from.

Everything is in points above a baseline, never raw points, so that filling an empty
slot is never confused with a real gain — and everything is priced *per horizon*.
The provider's cumulative 1- and 3-year projections split each player into two
components: year 1 (`points_1yr`) and years 2-3 (`points - points_1yr`). Lineups are
fielded per season, so a 69-point injury year cannot hide inside a healthy 3-year sum:
the starting lineup is solved separately on each component against that horizon's own
replacement levels, and the surpluses add. A team's value is

    V(roster) = sum over horizons h of
        sum over filled starting slots of (points_h - that slot's replacement_h)
      + sum over players benched in h of  0.55^d * depth_value_h

where the year-1 bench job is insurance only — ins[pos] * max(points_yr1 - wire_yr1, 0),
a backup who cannot play this season cannot cover starter games missed this season —
and the years-2-3 bench jobs are growth plus insurance on the years-2-3 excess,
(0.30 + ins[pos]) * max(points_yr23 - wire_yr23, 0). Both are measured against that
horizon's wire and floored at zero. A backloaded rookie still outranks a flat veteran
with the same 3-year sum on the bench — his value just lives in the horizon where he
actually produces it.

A starting slot no rostered body can beat is *streamed*, and streaming is not free: the
slot yields the stream level — the best player still available at a position the slot
accepts, capped at the replacement level — and is charged (stream - replacement) <= 0.
Early in a draft the cap binds and streaming costs nothing, which is the documented
counterfactual for skipping a position (you end the draft with a late-round starter,
not with nothing). As startable bodies at a horizon dry up, the stream level falls and
every unfilled slot at that horizon goes negative, which is what stops a roster from
punting year 1 for free while it stacks years-2-3 value: by the end of the draft the
stream level *is* the wire, so a year-1 hole is charged what it actually costs.

d counts the bench players that team already has at the same position, per horizon. The
value of a player to a team is V(roster + player) - V(roster). Three choices in there
each cost a degenerate draft to learn, and are explained at their definitions:

  * Starting slots are priced against the *marginal-starter* level, the same baseline VOR
    reports, so the board and the simulated drafters cannot disagree (`slot_replacement`).
  * A flex slot is priced at max over the positions it accepts, so a player is worth
    strictly less there than in his dedicated slot. That is where scarcity comes from,
    rather than being asserted.
  * Bench value is measured against the wire and floored at zero, discounted by depth at
    the player's own position (`team_value`).

Roster legality is a constraint, not a price: 1 QB / 2 RB / 3 WR / 1 TE come from slots
nothing else can cover, so once a team's remaining picks equal its unfilled mandatory
spots, its candidates narrow to what it still owes (`Draft.candidates` in simulation.py).
"""

from __future__ import annotations

import math

from .league import (
    POSITION_DEPTH_DECAY,
    POSITIONS,
    ROSTER_SLOTS,
    SLOT_CHAIN,
    SLOT_ELIGIBLE,
    STARTING_SLOTS,
    TEAMS,
)
from .pool import Player, by_position

# The two value horizons the provider's cumulative projections can distinguish:
# year 1, and years 2-3 as one block. Every level dict in the ranker — replacement,
# wire, slot levels — is keyed by horizon first, position/slot second.
HORIZONS = ("yr1", "yr23")
_DEPTH_WEIGHTS = tuple(POSITION_DEPTH_DECAY**depth for depth in range(ROSTER_SLOTS))


def horizon_points(p: Player, h: str) -> float:
    """A player's points in one horizon (precomputed on Player — see pool.py)."""
    return p.points_yr1 if h == "yr1" else p.points_yr23


# Sort keys per horizon, bound once: the lineup solver runs inside every marginal-value
# evaluation, so per-call lambda construction and string dispatch are worth avoiding.
_HKEY = {
    "yr1": lambda q: (-q.points_yr1, q.player_id),
    "yr23": lambda q: (-q.points_yr23, q.player_id),
}


def pos_by_horizon(players: list[Player]) -> dict[str, dict[str, list[Player]]]:
    """Per-horizon position lists, each sorted by that horizon's points descending."""
    pos = by_position(players)
    return {
        h: {k: sorted(v, key=_HKEY[h]) for k, v in pos.items()} for h in HORIZONS
    }


def sorted_by_horizon(roster: list[Player]) -> dict[str, list[Player]]:
    """A roster pre-sorted per horizon — the form `team_value` consumes.

    Valuing a candidate means re-valuing the roster with him added, thousands of times
    per pick, and re-sorting 25 players to place 1 was the single largest cost in the
    simulation. So the roster is sorted once per pick and each candidate is merged in
    by `insert_sorted`.
    """
    return {h: sorted(roster, key=_HKEY[h]) for h in HORIZONS}


def insert_sorted(seq: list[Player], p: Player, h: str) -> list[Player]:
    """A new list with `p` merged into an already-sorted horizon list."""
    points = p.points_yr1 if h == "yr1" else p.points_yr23
    lo, hi = 0, len(seq)
    while lo < hi:
        mid = (lo + hi) // 2
        q = seq[mid]
        q_points = q.points_yr1 if h == "yr1" else q.points_yr23
        if q_points > points or (q_points == points and q.player_id < p.player_id):
            lo = mid + 1
        else:
            hi = mid
    out = seq.copy()
    out.insert(lo, p)
    return out


# --- lineup solver ----------------------------------------------------------------


def slot_replacement(rep: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """What each starting slot yields without spending a valuable pick on it, per horizon.

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
    return {
        h: {slot: max(rep[h][pos] for pos in elig) for slot, elig in SLOT_ELIGIBLE.items()}
        for h in HORIZONS
    }


def lineup_surplus(
    roster: list[Player],
    slot_rep: dict[str, float],
    stream: dict[str, float],
    h: str,
) -> tuple[float, list[Player], dict[str, int]]:
    """Points above replacement from the optimal starting lineup in one horizon.

    `roster` must already be sorted by this horizon's points descending (`_HKEY[h]` —
    see `sorted_by_horizon`); the greedy's exactness depends on walking it in that order.

    `slot_rep` is that horizon's slot levels and `stream` what each slot yields unfilled
    (the best available body the slot accepts, capped at `slot_rep` — see the module
    docstring). A slot is filled when a rostered player beats its stream level, earning
    (points - slot_rep) even when that is negative — a below-replacement starter still
    beats signing the stream — and every slot left open is charged (stream - slot_rep).

    Greedy in descending horizon points, each player taking the most restrictive slot
    still open. Exact because slot eligibility is laminar and both levels are
    non-decreasing along every chain (`stream` is a min against `slot_rep` of a max over
    a growing eligibility set), so a player never gains by moving to a looser slot;
    --selftest checks this against brute force. A player who cannot beat the stream in
    the tightest open slot cannot beat it anywhere looser and is benched. The lineups
    can differ between horizons — that is the point: a player recovering from injury is
    benched in year 1 and started in years 2-3. The bench comes back in horizon-points
    order because the roster is walked in that order.
    """
    caps = dict(STARTING_SLOTS)
    surplus = 0.0
    bench: list[Player] = []
    started = {pos: 0 for pos in POSITIONS}
    year1 = h == "yr1"
    for p in roster:
        pts = p.points_yr1 if year1 else p.points_yr23
        placed = False
        for slot in SLOT_CHAIN[p.position]:
            if caps[slot] == 0:
                continue
            if pts <= stream[slot]:
                break
            caps[slot] -= 1
            surplus += pts - slot_rep[slot]
            started[p.position] += 1
            placed = True
            break
        if not placed:
            bench.append(p)
    for slot, n in caps.items():
        if n:
            surplus += n * (stream[slot] - slot_rep[slot])
    return surplus, bench, started


def _lineup_value(
    roster: list[Player],
    slot_rep: dict[str, float],
    stream: dict[str, float],
    depth_value: dict[int, float],
    h: str,
) -> float:
    """Lineup surplus plus bench value without walking the roster twice."""
    caps = dict(STARTING_SLOTS)
    surplus = 0.0
    bench_depth = {pos: 0 for pos in POSITIONS}
    bench_values: list[float] = []
    year1 = h == "yr1"
    for p in roster:
        pts = p.points_yr1 if year1 else p.points_yr23
        placed = False
        for slot in SLOT_CHAIN[p.position]:
            if caps[slot] == 0:
                continue
            if pts <= stream[slot]:
                break
            caps[slot] -= 1
            surplus += pts - slot_rep[slot]
            placed = True
            break
        if not placed:
            depth = bench_depth[p.position]
            bench_values.append(_DEPTH_WEIGHTS[depth] * depth_value[p.player_id])
            bench_depth[p.position] = depth + 1
    for slot, n in caps.items():
        if n:
            surplus += n * (stream[slot] - slot_rep[slot])
    for value in bench_values:
        surplus += value
    return surplus


def starting_positions(roster: list[Player], h: str) -> list[str]:
    """Which positions fill the 10 slots when a team must field everyone it can, per horizon.

    No surplus gate here: a team starts its best available body every week even when that
    body is below replacement. This is the measurement used for replacement levels, so
    gating it would undercount starters and bias replacement upward. The horizon decides
    the pecking order into the flex slots, so year-1 and years-2-3 starter counts can
    genuinely differ.
    """
    caps = dict(STARTING_SLOTS)
    out: list[str] = []
    for p in sorted(roster, key=_HKEY[h]):
        for slot in SLOT_CHAIN[p.position]:
            if caps[slot]:
                caps[slot] -= 1
                out.append(p.position)
                break
    return out


def team_value(
    roster: dict[str, list[Player]],
    slot_rep: dict[str, dict[str, float]],
    depth_value: dict[str, dict[int, float]],
    streams: dict[str, dict[str, float]],
    extra: Player | None = None,
) -> float:
    """Per-horizon starting-lineup surplus plus bench depth value, summed over horizons.

    `roster` is the pre-sorted per-horizon form from `sorted_by_horizon`; `extra` prices
    the roster with one more player without re-sorting, which is how every candidate at
    every pick is valued.

    Starters are priced against the marginal-starter baseline (via `slot_rep`) because that
    is what you would otherwise have in the slot, and slots nobody fills are charged their
    stream level (`streams`, built by `Draft.stream_levels` from what is actually still
    available). Bench players are priced against the
    *wire* — `depth_value` is the per-horizon growth + insurance value built in
    `Draft.__init__` — because a backup's job is to beat what you would otherwise have to
    sign, and then discounted by how likely he ever plays.

    The discount is by depth at the player's own position, not by a running bench index: a
    team's fifth QB cannot start in a league with two QB-capable slots, however large his
    value looks. Using a raw bench index let one team draft ten QBs. Depth is counted
    within each horizon, in that horizon's pecking order — the injured veteran is the
    year-1 bench's last body and the years-2-3 lineup's starter, not one or the other.

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
    total = 0.0
    for h in HORIZONS:
        seq = roster[h] if extra is None else insert_sorted(roster[h], extra, h)
        total += _lineup_value(seq, slot_rep[h], streams[h], depth_value[h], h)
    return total


def compute_vor(players: list[Player], rep: dict[str, dict[str, float]]) -> dict[int, float]:
    """Sum over horizons of (horizon points - that horizon's replacement level)."""
    return {
        p.player_id: sum(horizon_points(p, h) - rep[h][p.position] for h in HORIZONS)
        for p in players
    }


def upside_points(p: Player) -> float:
    """Reporting diagnostic: the 3-year total re-projected at the years-2-3 pace.

    Equal to `points` for a perfectly flat scorer, above it for a backloaded player,
    below it for a declining veteran. No longer a pricing input — bench growth is
    measured directly on the years-2-3 component against the years-2-3 wire in
    `Draft.__init__` — but kept in the output because the gap against `points_3yr`
    reads as the provider's implied growth per season.
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


def _rep_at_rank(pos_players: list[Player], rank: int, h: str) -> float:
    """Horizon points of the rank-th best player at a position by that horizon (1-based),
    clamped to the pool."""
    if not pos_players:
        return 0.0
    return horizon_points(pos_players[min(max(rank, 1), len(pos_players)) - 1], h)


def seed_replacement(players: list[Player]) -> dict[str, dict[str, float]]:
    """Iteration-0 replacement from pure slot counting, no draft behaviour assumed.

    Per horizon: assign the top of the pool to the league's 100 starting slots the way a
    perfectly efficient market would — dedicated slots by that horizon's positional rank,
    then the 20 flex slots and 10 superflex slots to the best players still eligible —
    and take the best player at each position who did not earn a slot.
    """
    pos_h = pos_by_horizon(players)
    out: dict[str, dict[str, float]] = {}
    for h in HORIZONS:
        caps = {slot: n * TEAMS for slot, n in STARTING_SLOTS.items()}
        used = {p: 0 for p in POSITIONS}
        for p in sorted(players, key=_HKEY[h]):
            for slot in SLOT_CHAIN[p.position]:
                if caps[slot]:
                    caps[slot] -= 1
                    used[p.position] += 1
                    break
        out[h] = {k: _rep_at_rank(pos_h[h][k], used[k] + 1, h) for k in POSITIONS}
    return out


def replacement_from_draft(
    rosters: list[list[Player]], pos_h: dict[str, dict[str, list[Player]]]
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, int]]]:
    """Replacement = the best player at each position who is not starting-caliber, per horizon.

    For each horizon, count how many players at each position hold a starting slot across
    all 10 simulated teams when lineups are set on that horizon's points; the next best
    player at that position by those points is the replacement level.
    """
    rep: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    for h in HORIZONS:
        c = {p: 0 for p in POSITIONS}
        for roster in rosters:
            for position in starting_positions(roster, h):
                c[position] += 1
        counts[h] = c
        rep[h] = {k: _rep_at_rank(pos_h[h][k], c[k] + 1, h) for k in POSITIONS}
    return rep, counts


def wire_replacement(
    taken: set[int], pos: dict[str, list[Player]]
) -> dict[str, dict[str, float]]:
    """The best player at each position actually left undrafted, per horizon.

    Distinct from the starting-caliber baseline above, and much lower: 290 of 350 pool
    players get rostered, so the wire is picked clean. 29 QBs go in the simulated draft
    against 20 QB starting slots, which means QB21 (the marginal starter) is somebody's
    backup, not a free add. Both numbers are reported because they answer different
    questions — `replacement.levels` asks "how much better is he than the worst player
    good enough to start in this league", `replacement.wire_levels` asks "how much better
    is he than what I could sign for nothing after the draft". The former is the
    conventional VBD baseline and the sort key here, because starting slots are the scarce
    resource a draft pick buys; the latter prices bench depth (`team_value`). The best
    year-1 add and the best years-2-3 stash can be different players — post-draft you can
    sign either, so each horizon takes its own max.
    """
    out: dict[str, dict[str, float]] = {h: {} for h in HORIZONS}
    for position, players in pos.items():
        left = [p for p in players if p.player_id not in taken]
        for h in HORIZONS:
            out[h][position] = max((horizon_points(p, h) for p in left), default=0.0)
    return out

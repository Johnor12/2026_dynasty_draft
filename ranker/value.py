"""Expected lineup points and replacement-level measurement.

The provider's cumulative projections are split into year 1 and years 2-3. A roster is
valued separately in both horizons, then the results are added. Within one horizon the
solver considers every legal positional composition of the ten starting slots.

For a composition that starts ``c`` players at a position, players are ordered by their
points when active. The first ``c`` receive their full projection. Every deeper player
receives his projection times the probability that fewer than ``c`` higher-ranked
teammates are available. Those probabilities come from the position-wide unavailable
rates in ``league.py`` and are calculated exactly with a small Bernoulli distribution.
The best waiver player at the position is inserted once as an always-available body, so
one free agent can fill one lineup job, never several simultaneous holes.

This is one objective with one set of units: expected lineup points. There is no role
threshold and no separate bench bonus. A better projection can retain every role a worse
projection could fill, which makes roster value monotone when a player improves, is
replaced by a better same-position player, or is simply added. Marginal-starter levels
are reported as league diagnostics; they are not mixed into roster utility.
"""

from __future__ import annotations

import math
from functools import lru_cache

from .league import (
    DEDICATED_SLOTS,
    POSITIONS,
    SLOT_CHAIN,
    STARTING_SLOTS,
    TEAMS,
    UNAVAILABLE_RATE,
)
from .pool import Player, by_position

# The two value horizons the provider's cumulative projections can distinguish:
# year 1, and years 2-3 as one block. Every level dict in the ranker — replacement,
# and wire — is keyed by horizon first, position second.
HORIZONS = ("yr1", "yr23")


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
    points = horizon_points(p, h)
    lo, hi = 0, len(seq)
    while lo < hi:
        mid = (lo + hi) // 2
        q = seq[mid]
        q_points = horizon_points(q, h)
        if q_points > points or (q_points == points and q.player_id < p.player_id):
            lo = mid + 1
        else:
            hi = mid
    out = seq.copy()
    out.insert(lo, p)
    return out


# --- expected lineup solver -------------------------------------------------------


def _starter_compositions() -> tuple[dict[str, int], ...]:
    """All position counts that can legally occupy the ten starting slots."""
    total = sum(STARTING_SLOTS.values())
    out: list[dict[str, int]] = []
    for qb in range(DEDICATED_SLOTS["QB"], DEDICATED_SLOTS["QB"] + 2):
        for rb in range(DEDICATED_SLOTS["RB"], DEDICATED_SLOTS["RB"] + 4):
            for wr in range(DEDICATED_SLOTS["WR"], DEDICATED_SLOTS["WR"] + 4):
                for te in range(DEDICATED_SLOTS["TE"], DEDICATED_SLOTS["TE"] + 4):
                    counts = {"QB": qb, "RB": rb, "WR": wr, "TE": te}
                    if sum(counts.values()) != total:
                        continue
                    qb_extra = qb - DEDICATED_SLOTS["QB"]
                    non_qb_extra = sum(
                        counts[pos] - DEDICATED_SLOTS[pos] for pos in ("RB", "WR", "TE")
                    )
                    if qb_extra <= STARTING_SLOTS["SF"] and non_qb_extra <= (
                        STARTING_SLOTS["FLEX"] + STARTING_SLOTS["SF"] - qb_extra
                    ):
                        out.append(counts)
    return tuple(out)


_STARTER_COMPOSITIONS = _starter_compositions()
_MAX_STARTERS = {
    pos: max(composition[pos] for composition in _STARTER_COMPOSITIONS)
    for pos in POSITIONS
}


@lru_cache(maxsize=32_768)
def _position_expected_values(
    projections: tuple[tuple[int, float], ...],
    wire_points: float,
    unavailable: float,
    max_starters: int,
) -> tuple[float, ...]:
    """All starter-count values for one depth chart; cached across candidate branches."""
    available = 1.0 - unavailable
    entries = [
        (points / available, points, available, player_id)
        for player_id, points in projections
    ]
    # One actual free agent, not an unlimited scalar that can fill every open slot.
    entries.append((wire_points, wire_points, 1.0, -1))
    entries.sort(key=lambda row: (-row[0], row[3]))

    out = [0.0]
    for starter_count in range(1, max_starters + 1):
        # dist[k] = P(exactly k higher bodies are active), with the final cell
        # collecting saturated states that cannot leave a job for a lower body.
        dist = [1.0] + [0.0] * starter_count
        total = 0.0
        for _, expected_points, active_probability, _ in entries:
            total += expected_points * sum(dist[:starter_count])
            next_dist = [0.0] * (starter_count + 1)
            for active_higher, probability in enumerate(dist):
                next_dist[active_higher] += probability * (1.0 - active_probability)
                next_dist[min(active_higher + 1, starter_count)] += (
                    probability * active_probability
                )
            dist = next_dist
        out.append(total)
    return tuple(out)


def position_expected_value(
    players: list[Player], wire_points: float, starter_count: int, h: str
) -> float:
    """Expected points from one position with a unique, always-available wire body.

    Player projections are unconditional. Dividing by availability orders players by
    their points when active; multiplying that active rate by their own availability
    returns the original projection. The running Bernoulli distribution supplies the
    probability that fewer than ``starter_count`` higher bodies are active, which is the
    probability this depth-chart entry is called on.
    """
    unavailable = UNAVAILABLE_RATE[players[0].position] if players else 0.0
    projections = tuple(sorted((p.player_id, horizon_points(p, h)) for p in players))
    return _position_expected_values(
        projections, wire_points, unavailable, starter_count
    )[starter_count]


def expected_lineup_value(
    roster: list[Player],
    wire: dict[str, float],
    h: str,
) -> float:
    """Highest expected lineup value across every legal starter composition."""
    by_pos = {pos: [] for pos in POSITIONS}
    for player in roster:
        by_pos[player.position].append(player)
    values = {}
    for pos in POSITIONS:
        position_projections = tuple(
            sorted((p.player_id, horizon_points(p, h)) for p in by_pos[pos])
        )
        values[pos] = _position_expected_values(
            position_projections, wire[pos], UNAVAILABLE_RATE[pos], _MAX_STARTERS[pos]
        )
    return max(
        sum(values[pos][counts[pos]] for pos in POSITIONS)
        for counts in _STARTER_COMPOSITIONS
    )


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
    wire: dict[str, dict[str, float]],
    extra: Player | None = None,
) -> float:
    """Expected optimal lineup points over both horizons."""
    return sum(
        expected_lineup_value(
            roster[h]
            if extra is None
            else insert_sorted(roster[h], extra, h),
            wire[h],
            h,
        )
        for h in HORIZONS
    )


def team_values_with_candidates(
    roster: list[Player],
    wire: dict[str, dict[str, float]],
    candidates: list[Player],
) -> tuple[float, dict[int, float]]:
    """Roster value and each one-player addition, sharing the unchanged positions.

    A candidate alters one position. Computing the other three depth charts once per
    horizon avoids rebuilding them for every personal-strategy option during rollouts.
    """
    by_pos = {pos: [] for pos in POSITIONS}
    for player in roster:
        by_pos[player.position].append(player)

    baseline = 0.0
    values = {candidate.player_id: 0.0 for candidate in candidates}
    for h in HORIZONS:
        point_rows = {
            pos: tuple(
                sorted(
                    (player.player_id, horizon_points(player, h))
                    for player in by_pos[pos]
                )
            )
            for pos in POSITIONS
        }
        base_position_values = {
            pos: _position_expected_values(
                point_rows[pos],
                wire[h][pos],
                UNAVAILABLE_RATE[pos],
                _MAX_STARTERS[pos],
            )
            for pos in POSITIONS
        }
        baseline += max(
            sum(base_position_values[pos][counts[pos]] for pos in POSITIONS)
            for counts in _STARTER_COMPOSITIONS
        )
        for candidate in candidates:
            pos = candidate.position
            candidate_rows = tuple(
                sorted(
                    point_rows[pos]
                    + ((candidate.player_id, horizon_points(candidate, h)),)
                )
            )
            candidate_position_values = _position_expected_values(
                candidate_rows,
                wire[h][pos],
                UNAVAILABLE_RATE[pos],
                _MAX_STARTERS[pos],
            )
            values[candidate.player_id] += max(
                sum(
                    (
                        candidate_position_values
                        if position == pos
                        else base_position_values[position]
                    )[counts[position]]
                    for position in POSITIONS
                )
                for counts in _STARTER_COMPOSITIONS
            )
    return baseline, values


def upside_points(p: Player) -> float:
    """Reporting diagnostic: the 3-year total re-projected at the years-2-3 pace.

    Equal to `points` for a perfectly flat scorer, above it for a backloaded player,
    below it for a declining veteran. It is a reporting diagnostic, not a pricing input:
    future growth already lives in the years-2-3 projection and lineup.
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
    conventional VBD baseline and the board's sort key. The latter contributes one unique,
    always-available body per position to expected lineup value. The best year-1 add and
    the best years-2-3 stash can be different players, so each horizon takes its own max.
    """
    out: dict[str, dict[str, float]] = {h: {} for h in HORIZONS}
    for position, players in pos.items():
        left = [p for p in players if p.player_id not in taken]
        for h in HORIZONS:
            out[h][position] = max((horizon_points(p, h) for p in left), default=0.0)
    return out

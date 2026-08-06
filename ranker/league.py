"""This league's shape (from README.md) and the strategy constants.

10 teams, 0.5 PPR + 0.5 TE premium, superflex. Starters are 1 QB / 2 RB / 3 WR / 1 TE /
2 W-R-T / 1 W-R-T-Q = 10. Then 15 bench and 4 rookie-only taxi spots = 29 draftable
roster spots = 29 rounds = 290 picks. Snake with a 3rd-round reversal. My slot is 1.02.
"""

from __future__ import annotations

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
# The taxi spots are rookie-only, so this is also the most veterans a roster can hold.
NON_TAXI_SLOTS = sum(STARTING_SLOTS.values()) + BENCH_SLOTS  # 25
ROSTER_SLOTS = NON_TAXI_SLOTS + TAXI_SLOTS  # 29
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

DEPTH_BASE = 0.30  # growth: a first backup keeps this fraction of his upside over the wire
POSITION_DEPTH_DECAY = 0.55  # ...and each further body at the same position much less
# Insurance: expected share of a season the first backup at a position inherits from the
# starters ahead of him — their byes plus games missed to injury. RB highest (two dedicated
# starters, worst attrition), QB lowest (one starter, durable). Judgment-call constants on
# league-wide averages, not per-player durability.
INSURANCE_BASE = {"QB": 0.08, "RB": 0.20, "WR": 0.12, "TE": 0.10}
SURVIVAL_SIGMA = 3.5  # softness of "will he last until my next pick"
# Candidates per position *per horizon ordering* considered for the next pick — the
# lists take the top of both the year-1 and the years-2-3 ordering, so this yields up
# to four distinct players per position.
LOOKAHEAD_PER_POS = 2
# An entirely unfilled dedicated starter group receives a 3x source-rank boost;
# the boost fades linearly as that position's dedicated starters are filled.
OPPONENT_BALANCE_STRENGTH = 2.0
# Multiplier around each opponent's fitted source adherence: 1 reproduces the observed
# mean log-rank loss before roster-balance adjustments, while 0 removes random variation.
NOISE = 1.0
# Cap on fixed-point iterations before a cycle must have closed. Per-horizon levels
# doubled the state (8 starter counts + 8 wire levels), so exact recurrence takes
# longer than the 24 the single-horizon state needed.
MAX_ITERS = 80
SIMS = 200
ROLLOUT_SIMS = 100  # full-draft playouts per candidate at my next pick (sim.rollout)
SEED = 20260804


# --- draft order ------------------------------------------------------------------


def draft_order(teams: int = TEAMS, rounds: int = ROUNDS) -> list[int]:
    """Slot (1-based) picking at each overall pick. Snake with a 3rd-round reversal.

    Round 1 forward, rounds 2 and 3 both reverse (that is the reversal), then the snake
    resumes: even rounds forward, odd rounds reverse. Pinned to the README's stated picks
    for slot 2 (1.02, 2.09, 3.09, 4.02, 5.09, 6.02, ..., 28.02, 29.09) in validate().
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

"""Solve replacement levels from the league shape produced by a simulated draft."""

from __future__ import annotations

import sys

from .board import Board
from .league import MAX_ITERS, POSITIONS, STARTING_SLOTS, TEAMS
from .opponents import OpponentStrategy
from .pool import Player, by_position
from .simulation import Draft
from .value import (
    HORIZONS,
    apportion,
    compute_vor,
    pos_by_horizon,
    replacement_from_draft,
    seed_replacement,
    wire_replacement,
)


def converge(
    players: list[Player],
    board: Board,
    report: bool,
    opponents: dict[int, OpponentStrategy],
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    dict[str, dict[str, int]],
    Draft,
    dict,
]:
    """Fixed point: replacement -> valuation -> draft -> starter counts -> replacement.

    All levels and counts are per horizon (value.HORIZONS): each iteration's draft is
    measured twice — once on year-1 points, once on years-2-3 points — and both sets of
    levels feed the next iteration's valuation.

    The map is piecewise-constant: starter counts are integers, so the replacement level
    jumps between adjacent players in the pool (RB21 and RB22 are 18 points apart) instead
    of moving continuously. That means gradient-style damping cannot settle it — it just
    orbits the discontinuity — while plain undamped iteration is eventually periodic.

    So iterate undamped, watch for a repeated state, and average the replacement levels
    over the cycle once one closes. Each iteration is a deterministic function of the
    previous one's (replacement, wire) pair, and the replacement levels are themselves a
    function of the starter counts, so the state key is the counts plus the wire levels.
    A cycle of length 1 is an exact fixed point; a longer cycle means the draft genuinely
    alternates between neighbouring league shapes (e.g. 20 vs 21 RBs starting) and the
    average across it is the honest answer. The cycle is reported rather than hidden.

    Two levels come out of each draft and both are iterated: the marginal-starter level
    that is the VOR baseline, and the wire level that prices bench depth. They are separate
    quantities and must not be collapsed into one (see `slot_replacement`).

    Every draft here starts from `board`, so on a live board the fixed point is over the
    rosters this league will actually finish with. The measurement is still league-wide and
    still against the whole pool: replacement is a property of the league's 100 starting
    slots, and the player who defines it may already be on somebody's roster.
    """
    pos = by_position(players)
    pos_h = pos_by_horizon(players)
    rep = seed_replacement(players)
    stream = {h: dict(rep[h]) for h in HORIZONS}  # no draft to read a wire off yet
    trace = [
        {
            "iteration": 0,
            "source": "slot_assignment",
            "replacement": {h: dict(rep[h]) for h in HORIZONS},
        }
    ]
    seen: dict[tuple[float, ...], int] = {}
    observations: list[dict[str, dict[str, float]]] = []
    wire_observations: list[dict[str, dict[str, float]]] = []
    counts_by_iter: list[dict[str, dict[str, int]]] = []
    cycle_start: int | None = None

    for it in range(1, MAX_ITERS + 1):
        vor = compute_vor(players, rep)
        draft = Draft(
            players, rep, vor, board, wire=stream,
            opponents=opponents,
        )
        draft.run()
        starter_rep, counts = replacement_from_draft(draft.rosters, pos_h)
        wire = wire_replacement(draft.taken, pos)
        observations.append(starter_rep)
        wire_observations.append(wire)
        counts_by_iter.append(counts)
        key = tuple(counts[h][k] for h in HORIZONS for k in POSITIONS) + tuple(
            wire[h][k] for h in HORIZONS for k in POSITIONS
        )
        trace.append(
            {
                "iteration": it,
                "source": "draft_simulation",
                "starters_by_position": {h: dict(counts[h]) for h in HORIZONS},
                "observed_replacement": {
                    h: {k: round(v, 1) for k, v in starter_rep[h].items()} for h in HORIZONS
                },
            }
        )
        if report:
            for h in HORIZONS:
                print(
                    f"  iter {it} {h}: starters={counts[h]} observed rep="
                    + ", ".join(f"{k} {starter_rep[h][k]:.0f}" for k in POSITIONS),
                    file=sys.stderr,
                )
        if key in seen:
            cycle_start = seen[key]
            break
        seen[key] = it
        rep = starter_rep
        stream = wire

    assert cycle_start is not None, f"no cycle within {MAX_ITERS} iterations"
    cycle = observations[cycle_start - 1 : -1] or observations[cycle_start - 1 :]
    wire_cycle = wire_observations[cycle_start - 1 : -1] or wire_observations[cycle_start - 1 :]
    cycle_counts = counts_by_iter[cycle_start - 1 : len(observations) - 1] or counts_by_iter[-1:]
    rep = {
        h: {k: sum(o[h][k] for o in cycle) / len(cycle) for k in POSITIONS} for h in HORIZONS
    }
    stream = {
        h: {k: sum(o[h][k] for o in wire_cycle) / len(wire_cycle) for k in POSITIONS}
        for h in HORIZONS
    }
    counts = {
        h: apportion(
            {k: sum(c[h][k] for c in cycle_counts) / len(cycle_counts) for k in POSITIONS},
            TEAMS * sum(STARTING_SLOTS.values()),
        )
        for h in HORIZONS
    }
    if report:
        print(
            f"  cycle of length {len(cycle)} closed at iteration {cycle_start}; "
            "averaging replacement across it",
            file=sys.stderr,
        )

    # Final deterministic draft at the settled levels, so sim_pick, my_decisions and the
    # reported replacement levels describe one and the same draft.
    vor = compute_vor(players, rep)
    draft = Draft(
        players, rep, vor, board, wire=stream,
        opponents=opponents,
    )
    draft.run()
    _, final_counts = replacement_from_draft(draft.rosters, pos_h)
    history = {
        "method": "undamped iteration to a limit cycle, averaged over the cycle",
        "cycle_length": len(cycle),
        "cycle_first_seen_at_iteration": cycle_start,
        "iterations_run": len(observations),
        "starters_in_final_draft": final_counts,
        # Spread of the cycle the levels were averaged over. A wide band means the league
        # shape genuinely wobbles between neighbouring configurations rather than settling.
        "cycle_replacement_range": {
            h: {
                k: [
                    round(min(o[h][k] for o in cycle), 1),
                    round(max(o[h][k] for o in cycle), 1),
                ]
                for k in POSITIONS
            }
            for h in HORIZONS
        },
        "trace": trace,
    }
    return rep, stream, counts, draft, history


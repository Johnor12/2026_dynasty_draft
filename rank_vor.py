#!/usr/bin/env python3
"""Value over replacement for this league, with an optimal-drafter draft simulation.

    uv run rank_vor.py                     # pool.json + draft.json -> rankings.json
    uv run rank_vor.py --report            # + board, convergence and recommendation on stderr
    uv run rank_vor.py --no-draft          # ignore the live board, rank the whole pool
    uv run rank_vor.py --selftest          # verify the lineup solver and the board loader

Scope is this league and nothing else; the league constants and strategy knobs live in
ranker/league.py. The value inputs are `points_3yr` and `points_1yr` from `pool.json` —
points in this league's scoring, split into two horizons: year 1 and years 2-3 (see
ranker/pool.py for why the provider's 3D value is deliberately unused). Lineups are
fielded per season, so each horizon is priced against its own replacement levels — a
69-point injury year cannot hide inside a healthy 3-year sum. The method, in one breath:
how many players start at each position is an *outcome* of how the league drafts, so
per-horizon replacement levels are found as a fixed point of an optimal-drafter draft
simulation (ranker/sim.py), roster value and the replacement measurement live in
ranker/value.py, and `draft.json` — the live board — is the simulation's starting state,
not a filter (ranker/board.py).

The output's headline `vor` sums the horizons: (year-1 points minus the year-1
marginal-starter level) + (years-2-3 points minus that level) — "how many points over
three years does this player add versus the best guy any team could have had at his
position without spending a starting-caliber pick, period by period" (`vor_yr1` and
`vor_yr23` carry the split). The `my_next_picks`
block is the direct answer to "who should I draft next" — unlike `vor`, it sees my roster
and the odds a player survives to my following pick, and its first pick is scored by
playing every candidate out to the end of the draft (ranker/sim.py `rollout`). Rank
columns are renumbered over the undrafted players actually emitted.

A caveat the data imposes, not the model: QB replacement lands around QB20-21 = ~820
points, which compresses elite QB value hard (Josh Allen is ~+225). That follows from the
provider projecting backup and rookie QBs at starter-grade volume. If those are not
credible, QB replacement is overstated and every elite QB is underrated here. The lever
is the projections, not the ranking method.

Python stdlib only. Deterministic: every tie breaks on player_id and the RNG is seeded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ranker.board import fresh_board, load_board
from ranker.league import MARKET_WEIGHT, NOISE, ROLLOUT_SIMS, SEED, SIMS
from ranker.output import build_payload, build_rankings, report_board, report_summary
from ranker.pool import load_pool
from ranker.selftest import selftest
from ranker.sim import (
    apply_option_redraw,
    apply_rollout,
    candidate_survival,
    converge,
    monte_carlo,
    option_redraw,
    rollout,
)
from ranker.validate import validate


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("input", nargs="?", default=Path("pool.json"), type=Path)
    ap.add_argument("-o", "--output", default=Path("rankings.json"), type=Path)
    ap.add_argument(
        "--draft",
        type=Path,
        default=None,
        help="live board to start from (default: draft.json if it is there)",
    )
    ap.add_argument(
        "--no-draft",
        action="store_true",
        help="ignore the live board: rank the whole pool from an empty draft",
    )
    ap.add_argument("--report", action="store_true", help="validation summary on stderr")
    ap.add_argument(
        "--selftest", action="store_true", help="check the solver and the board loader, then exit"
    )
    ap.add_argument("--sims", type=int, default=SIMS, help="noisy redraws for availability")
    ap.add_argument("--noise", type=float, default=NOISE, help="other teams' Gumbel scale")
    ap.add_argument(
        "--market-weight",
        type=float,
        default=MARKET_WEIGHT,
        help="how far the other nine teams follow the source ADP instead of their own "
        "board: 0 = all drafters fully optimal, 1 = pure best-available-by-ADP",
    )
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    if not args.input.exists():
        print(f"missing {args.input} - run pipeline.py first", file=sys.stderr)
        return 1

    players, pool_meta = load_pool(args.input)
    if args.selftest:
        return selftest(players)

    # The live board is the default starting state; an absent draft.json is not an error,
    # it is the preseason case. An explicitly named one that is missing is an error.
    board_problems: list[str] = []
    draft_path = args.draft or Path("draft.json")
    if args.no_draft:
        board = fresh_board()
    elif draft_path.exists():
        board, board_problems = load_board(
            json.loads(draft_path.read_text()), players, str(draft_path)
        )
    elif args.draft is not None:
        print(f"missing {draft_path} - run draft_pipeline/fetch_draft.py", file=sys.stderr)
        return 1
    else:
        print(
            f"no {draft_path}: ranking the whole pool from an empty board "
            "(run draft_pipeline/fetch_draft.py for the live one)",
            file=sys.stderr,
        )
        board = fresh_board()

    if args.report:
        print(
            f"pool: {len(players)} players "
            + ", ".join(f"{k} {v}" for k, v in pool_meta["by_position"].items())
            + f"; dropped {pool_meta['dropped_non_offense']} non-offense and "
            f"{len(pool_meta['dropped_zero_projection'])} zero-projection",
            file=sys.stderr,
        )
        report_board(board)
        print("converging replacement levels:", file=sys.stderr)

    rep, stream, counts, draft, history = converge(
        players, board, args.report, args.market_weight
    )

    candidates = (
        [c for _, _, c in draft.my_decisions.get(board.my_picks[0], [])]
        if board.my_picks
        else []
    )
    if args.report and candidates:
        print(
            f"next-pick options: {args.sims} short opponent redraws x "
            f"{len(candidates)} candidates",
            file=sys.stderr,
        )
    options = option_redraw(
        players, rep, board, stream, candidates,
        args.sims, args.noise, args.seed, args.market_weight,
    )
    draft = apply_option_redraw(
        draft, options, players, rep, board, stream, args.market_weight
    )
    candidates = (
        [c for _, _, c in draft.my_decisions.get(board.my_picks[0], [])]
        if board.my_picks
        else []
    )
    if args.report and candidates:
        print(
            f"rollout: {ROLLOUT_SIMS} full-draft playouts x {len(candidates)} candidates "
            f"at my next pick",
            file=sys.stderr,
        )
    rolled = rollout(
        players, rep, board, stream, candidates,
        ROLLOUT_SIMS, args.noise, args.seed, args.market_weight,
    )
    draft = apply_rollout(draft, rolled, players, rep, board, stream, args.market_weight)

    if args.report:
        print(
            f"monte carlo: {args.sims} noisy drafts (noise={args.noise}, "
            f"market_weight={args.market_weight})",
            file=sys.stderr,
        )
    picks, drafted = monte_carlo(
        players, rep, board, stream, args.sims, args.noise, args.seed, args.market_weight
    )
    if args.report and candidates:
        print(
            f"candidate survival: {args.sims} banned-me redraws x {len(candidates)} "
            f"candidates",
            file=sys.stderr,
        )
    survival = candidate_survival(
        players, rep, board, stream, candidates,
        args.sims, args.noise, args.seed, args.market_weight,
    )
    rows = build_rankings(players, rep, stream, draft, picks, drafted, args.sims, board)

    problems = board_problems + validate(rows, players, rep, counts, draft, board, history)
    payload = build_payload(
        players, pool_meta, board, rep, stream, counts, draft, history, rows, problems,
        args.sims, args.noise, args.seed, args.market_weight, options, rolled, survival,
    )
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    if args.report:
        report_summary(rows, rep, counts, draft, board, rolled, survival)
    if problems:
        print(f"\n{len(problems)} VALIDATION PROBLEM(S):", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
    scope = f"{len(rows)} undrafted of {len(players)}" if board.live else f"{len(rows)} players"
    print(f"wrote {args.output} ({scope})", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Offline checks for the states the live files cannot currently reach.

Three suites: the greedy lineup solver against brute force, source-based opponent
behavior, and the board loader against synthetic draft.json boards — a traded pick, a
selection outside the pool, a resumed partial board, and six malformed boards that must
be rejected.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import random
import sys

from .board import fresh_board, load_board
from .league import (
    DEDICATED_SLOTS,
    MY_SLOT,
    POSITIONS,
    ROUNDS,
    SEED,
    SLOT_ELIGIBLE,
    STARTING_SLOTS,
    TEAMS,
    TOTAL_PICKS,
    draft_order,
    pick_label,
)
from .opponents import OpponentStrategy, expected_log2_rank, rank_power
from .pool import Player
from .sim import Draft
from .value import (
    HORIZONS,
    compute_vor,
    horizon_points,
    lineup_surplus,
    seed_replacement,
    slot_replacement,
    sorted_by_horizon,
)


def synthetic_opponents(
    players: list[Player], board, first_order: list[Player] | None = None
) -> dict[int, OpponentStrategy]:
    """Complete external boards for simulation tests; no VOR value is stored in them."""
    default = first_order or players
    order = tuple(p.player_id for p in default)
    ranks = {player_id: rank for rank, player_id in enumerate(order, start=1)}
    return {
        slot: OpponentStrategy(
            slot=slot,
            roster_id=slot,
            username=f"team{slot}",
            source_id=f"test_source_{slot}",
            source_name=f"Test source {slot}",
            source_format="selftest",
            fit_score=100.0,
            confidence="strong",
            mean_log2_loss=0.5,
            rank_power=rank_power(0.5, len(players)),
            primary_players=len(players),
            ranks=ranks,
            order=order,
        )
        for slot in range(1, TEAMS + 1)
        if slot != board.my_slot
    }


def opponent_selftest(players: list[Player]) -> list[str]:
    """Opponent source order must diverge from and remain independent of my VOR board."""
    fails: list[str] = []
    board = fresh_board()
    rep = seed_replacement(players)
    vor = compute_vor(players, rep)
    # Put the lowest-VOR player first on every external board. An opponent must take him;
    # my optimizer must not, demonstrating that candidate generation is separated too.
    external = sorted(players, key=lambda p: (vor[p.player_id], p.player_id))
    opponents = synthetic_opponents(players, board, external)
    draft = Draft(players, rep, vor, board, opponents=opponents)
    opponent_take = draft.choose_opponent(0, 1)
    my_take = draft.choose(1, board.my_slot)
    if opponent_take != external[0]:
        fails.append("opponent ignored the top player on its inferred source board")
    if my_take == external[0]:
        fails.append("my VOR optimizer followed the opponent source board")

    for loss in (0.3, 1.5, 2.7):
        power = rank_power(loss, len(players))
        if abs(expected_log2_rank(power, len(players)) - loss) > 1e-6:
            fails.append(f"source-adherence calibration missed mean log2 loss {loss}")

    print(
        "  opponent strategies: provider order diverges from my VOR order and fitted "
        "rank noise reproduces observed adherence",
        file=sys.stderr,
    )
    return fails


def brute_force_surplus(
    roster: list[Player], slot_rep: dict[str, float], stream: dict[str, float], h: str
) -> float:
    """Exhaustive best assignment of a small roster to slots, in one horizon.

    Same objective as the greedy: filled slots earn (points - slot_rep), slots left
    open earn (stream - slot_rep). Benching everyone is a legal assignment, so the
    maximum can be negative on a bad roster against a picked-clean board.
    """
    slots = [s for s, n in STARTING_SLOTS.items() for _ in range(n)]
    empty = sum(stream[s] - slot_rep[s] for s in slots)
    best = -math.inf
    for combo in itertools.product(range(len(slots) + 1), repeat=len(roster)):
        used: set[int] = set()
        total = empty
        ok = True
        for p, s in zip(roster, combo):
            if s == len(slots):  # benched
                continue
            if s in used or p.position not in SLOT_ELIGIBLE[slots[s]]:
                ok = False
                break
            used.add(s)
            total += horizon_points(p, h) - stream[slots[s]]
        if ok:
            best = max(best, total)
    return best


def lineup_selftest(players: list[Player], trials: int = 300, seed: int = SEED) -> list[str]:
    """The greedy lineup solver must match brute force on random rosters, per horizon.

    Two stream scenarios per roster: a fresh board (stream = the slot levels, so an open
    slot costs nothing) and a picked-clean one (stream at 60% of the levels, so open
    slots go negative and below-replacement starters still beat streaming).
    """
    rng = random.Random(seed)
    rep = seed_replacement(players)
    slot_rep = slot_replacement(rep)
    fails: list[str] = []
    worst = 0.0
    for _ in range(trials):
        roster = rng.sample(players, rng.randint(1, 6))
        roster_sorted = sorted_by_horizon(roster)
        for h in HORIZONS:
            scenarios = (dict(slot_rep[h]), {s: 0.6 * v for s, v in slot_rep[h].items()})
            for stream in scenarios:
                greedy, _, _ = lineup_surplus(roster_sorted[h], slot_rep[h], stream, h)
                exact = brute_force_surplus(roster, slot_rep[h], stream, h)
                worst = max(worst, exact - greedy)
                if exact - greedy > 1e-6:
                    fails.append(
                        f"lineup solver ({h}, stream {stream['QB']:.0f}): "
                        + ", ".join(
                            f"{p.name}({p.position},{horizon_points(p, h):.0f})"
                            for p in roster
                        )
                        + f"  greedy={greedy:.1f} exact={exact:.1f}"
                    )
                    break
            if fails:
                break
        if fails:
            break
    print(
        f"  greedy lineup solver matched brute force on {trials} random rosters "
        f"x {len(HORIZONS)} horizons x 2 stream scenarios (max shortfall {worst:.2e})",
        file=sys.stderr,
    )
    return fails


def synthetic_draft(
    players: list[Player],
    made: int = 0,
    unrankable: dict[int, str] | None = None,
    trades: dict[int, int] | None = None,
) -> dict:
    """A `draft.json`-shaped board built offline, for the states the live file cannot reach.

    Today's live file has no traded picks and no selection outside the pool, so the two
    branches that handle them would go unexercised until the night they matter. Made picks
    take the pool in points order, which is a legal board and enough to check bookkeeping.
    `unrankable` maps a pick number to a position for a selection the pool does not carry;
    `trades` maps a pick number to the roster id that acquired it.
    """
    order = draft_order()
    slots = [
        {
            "draft_slot": s,
            "roster_id": 20 + s,  # deliberately not equal to the slot, as Sleeper's are not
            "user_id": None,
            "username": f"team{s}",
            "team_name": None,
            "is_mine": s == MY_SLOT,
        }
        for s in range(1, TEAMS + 1)
    ]
    roster_of_slot = {s["draft_slot"]: s["roster_id"] for s in slots}
    take = iter(players)
    picks: list[dict] = []
    for n, slot in enumerate(order, start=1):
        owner = (trades or {}).get(n, roster_of_slot[slot])
        pick = {
            "pick_no": n,
            "round": (n - 1) // TEAMS + 1,
            "pick_in_round": (n - 1) % TEAMS + 1,
            "draft_slot": slot,
            "roster_id": owner,
            "user_id": None,
            "username": None,
            "is_mine": owner == roster_of_slot[MY_SLOT],
            "status": "made" if n <= made else "pending",
            "sleeper_id": None,
            "name": None,
            "position": None,
            "team": None,
            "is_keeper": None,
        }
        if n <= made and (unrankable or {}).get(n):
            pick |= {
                "sleeper_id": f"not-in-pool-{n}",
                "name": f"Unrankable {n}",
                "position": unrankable[n],
                "team": "FA",
            }
        elif n <= made:
            p = next(take)
            pick |= {
                "sleeper_id": p.sleeper_id,
                "name": p.name,
                "position": p.position,
                "team": p.team,
            }
        picks.append(pick)

    pending = [p for p in picks if p["status"] == "pending"]
    mine = next((p for p in pending if p["is_mine"]), None)

    def summary(pick: dict | None) -> dict | None:
        if pick is None:
            return None
        return {
            "pick_no": pick["pick_no"],
            "round": pick["round"],
            "pick_in_round": pick["pick_in_round"],
            "draft_slot": pick["draft_slot"],
            "username": pick["username"],
            "slot": pick_label(pick["pick_no"]),
        }

    return {
        "source": "synthetic",
        "fetched_at": "2026-08-04T00:00:00+00:00",
        "draft_id": "synthetic",
        "league_name": "selftest",
        "status": "drafting",
        "format": {"type": "snake", "teams": TEAMS, "rounds": ROUNDS, "reversal_round": 3},
        "pick_count": TOTAL_PICKS,
        "picks_made": made,
        "picks_pending": TOTAL_PICKS - made,
        "on_the_clock": summary(pending[0] if pending else None),
        "me": {"username": "me", "draft_slot": MY_SLOT, "roster_id": roster_of_slot[MY_SLOT]},
        "my_next_pick": summary(mine),
        "slots": slots,
        "traded_picks": [{"round": (n - 1) // TEAMS + 1} for n in (trades or {})],
        "picks": picks,
    }


def board_selftest(players: list[Player]) -> list[str]:
    """The live-board loader, on states the real draft.json does not currently contain."""
    fails: list[str] = []

    def check(ok: bool, msg: str) -> None:
        if not ok:
            fails.append(f"board: {msg}")

    # An untouched live board must be the static snake, or the live path and the offline
    # path disagree about the league before a single pick is made.
    board, problems = load_board(synthetic_draft(players), players, "synthetic")
    fresh = fresh_board()
    check(not problems, f"empty synthetic board complained: {problems}")
    check(board.order == fresh.order, "empty live board's order != the static snake")
    check(board.pick_nos == fresh.pick_nos, "empty live board's pick numbers != 1..290")
    check(board.my_picks == fresh.my_picks, "empty live board's picks for me != the snake's")
    check(board.picks_left == fresh.picks_left, f"picks left {board.picks_left} != all {ROUNDS}")
    check(not board.taken and board.picks_made == 0, "empty live board has players drafted")

    # Made picks leave the pool and land on the team that made them.
    board, problems = load_board(synthetic_draft(players, made=13), players, "synthetic")
    check(not problems, f"13-pick board complained: {problems}")
    check(board.picks_made == 13 and len(board.taken) == 13, "13 made picks did not come through")
    check(board.pick_nos[:1] == [14], f"simulation resumes at {board.pick_nos[:1]}, want 14")
    check(board.my_picks[:1] == [19], f"my next pick is {board.my_picks[:1]}, want 19 (2.09)")
    check(
        [p.name for p in board.rosters[MY_SLOT - 1]] == [players[1].name],
        "pick 1.02 did not land on my roster",
    )
    check(board.picks_left[MY_SLOT - 1] == ROUNDS - 1, "my remaining picks did not drop by one")
    check(sum(board.picks_left) == TOTAL_PICKS - 13, "remaining picks do not sum to the board")
    # Picks 10 and 11 are both slot 10 — the turn at the end of round 1 into round 2.
    check(len(board.rosters[9]) == 2, "slot 10 did not get both sides of its turn")

    # A traded pick is exercised by the roster that acquired it, not by its column.
    board, problems = load_board(
        synthetic_draft(players, trades={5: 20 + MY_SLOT}), players, "synthetic"
    )
    check(board.order[4] == MY_SLOT, f"traded pick 5 is exercised by slot {board.order[4]}")
    check(
        board.my_picks[:3] == [2, 5, 19],
        f"my picks start {board.my_picks[:3]}, want my own 1.02, the traded 5, then 2.09",
    )
    check(
        board.picks_left[MY_SLOT - 1] == ROUNDS + 1 and board.picks_left[4] == ROUNDS - 1,
        "a traded pick did not move between the two teams' pick counts",
    )
    check(board.owed_size(MY_SLOT) == ROUNDS + 1, "the acquiring team's roster size did not grow")

    # A selection the pool cannot value fills a spot and answers its mandatory position.
    board, problems = load_board(
        synthetic_draft(players, made=1, unrankable={1: "QB"}), players, "synthetic"
    )
    check(not board.taken, "an unrankable pick took a pool player off the board")
    check(len(board.off_pool[0]) == 1, "an unrankable pick was not held as a roster spot")
    check(board.picks_left[0] == ROUNDS - 1, "an unrankable pick did not cost its team a pick")
    rep = seed_replacement(players)
    vor = compute_vor(players, rep)
    draft = Draft(players, rep, vor, board, opponents=synthetic_opponents(players, board))
    owed = sum(DEDICATED_SLOTS.values()) - 1  # the QB is answered, six mandatory spots left
    with_qb = draft.candidates([], picks_left=owed, off=board.off_pool[0])
    without = draft.candidates([], picks_left=owed, off=[])
    check(
        {c.position for c in with_qb} == {"RB", "WR", "TE"},
        "an unrankable QB did not satisfy the QB requirement",
    )
    check(
        {c.position for c in without} == set(POSITIONS),
        "the same team without him should still owe a QB",
    )

    # Resuming: every made pick survives, every pending pick is played exactly once.
    board, _ = load_board(synthetic_draft(players, made=57), players, "synthetic")
    opponents = synthetic_opponents(players, board)
    partial = Draft(players, rep, compute_vor(players, rep), board, opponents=opponents)
    partial.run(stop_before=5)
    check(
        set(partial.pick_of.values()) == set(board.pick_nos[:5]),
        "a short redraw did not stop immediately before its requested pick index",
    )
    draft = Draft(players, rep, compute_vor(players, rep), board, opponents=opponents)
    draft.run()
    check(
        len(draft.taken) == len(board.taken) + len(board.order),
        "the resumed draft did not take one new player per pending pick",
    )
    check(
        set(draft.pick_of.values()) == set(board.pick_nos),
        "the simulated picks are not exactly the board's pending picks",
    )
    check(not (set(draft.pick_of) & board.taken), "a player already drafted was drafted again")
    for slot in range(1, TEAMS + 1):
        made = board.rosters[slot - 1]
        got = draft.rosters[slot - 1]
        check(got[: len(made)] == made, f"slot {slot} lost one of its made picks")
        check(
            len(got) == len(made) + board.picks_left[slot - 1],
            f"slot {slot} finished with {len(got)} players, not what it owns",
        )

    # A board that disagrees with this script must say so, not be quietly absorbed. Every
    # one of these is a way a wrong draft.json could otherwise produce a plausible board.
    def complains(raw: dict, about: str, pool: list[Player] = players) -> None:
        _, problems = load_board(raw, pool, "synthetic")
        check(bool(problems), f"a board with {about} was accepted without complaint")

    raw = synthetic_draft(players, made=2)
    raw["format"]["teams"] = 12
    complains(raw, "12 teams")
    raw = synthetic_draft(players, made=3)
    raw["picks"][2] |= {
        "sleeper_id": raw["picks"][0]["sleeper_id"],
        "name": raw["picks"][0]["name"],
    }
    complains(raw, "the same player drafted twice")
    raw = synthetic_draft(players, made=0)
    raw["picks"][4] |= {"roster_id": None, "draft_slot": None}
    complains(raw, "a pick nobody owns")
    raw = synthetic_draft(players, made=3)
    raw["picks_made"] = 5
    complains(raw, "a header contradicting its own picks")
    raw = synthetic_draft(players, made=0)
    raw["me"]["draft_slot"] = 7
    complains(raw, "a different draft slot for me")
    complains(
        synthetic_draft(players, made=6),
        "a pool carrying no sleeper ids to join on",
        [dataclasses.replace(p, sleeper_id=None) for p in players],
    )

    print(
        "  live board: static snake reproduced, made picks retained, traded pick and "
        "unvalued pick handled, 233 pending picks resumed, 6 bad boards rejected",
        file=sys.stderr,
    )
    return fails


def selftest(players: list[Player]) -> int:
    print("selftest:", file=sys.stderr)
    fails = lineup_selftest(players) + opponent_selftest(players) + board_selftest(players)
    for f in fails:
        print(f"  FAIL {f}", file=sys.stderr)
    verdict = f"{len(fails)} failure(s)" if fails else "all checks passed"
    print(f"selftest: {verdict}", file=sys.stderr)
    return 1 if fails else 0

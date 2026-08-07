"""Stochastic availability, lookahead, and rollout planning.

The deterministic draft engine lives in `simulation.py`, and replacement convergence
lives in `convergence.py`. This module fans independent redraws and plan playouts across
worker processes, then applies the selected recommendation back to a deterministic draft.
"""

from __future__ import annotations

import math
import multiprocessing
import os
import random
from collections.abc import Sequence

from .board import Board
from .league import FIRST_PICK_PER_POS, LOOKAHEAD_PICKS
from .opponents import OpponentStrategy
from .pool import Player, by_position
from .simulation import Draft
from .value import compute_vor, sorted_by_horizon, team_value, wire_replacement


def broaden_first_pick(
    draft: Draft,
    players: list[Player],
    rep: dict[str, dict[str, float]],
    board: Board,
    stream: dict[str, dict[str, float]],
    opponents: dict[int, OpponentStrategy],
) -> Draft:
    """Replace only the live first decision's shortlist with the broader candidate pool."""
    if not board.my_picks:
        return draft
    pick_no = board.my_picks[0]
    pick_index = board.pick_nos.index(pick_no)
    state = Draft(
        players, rep, compute_vor(players, rep), board, wire=stream, opponents=opponents
    )
    state.run(stop_before=pick_index)
    draft.my_decisions[pick_no] = state.score_my_candidates(
        pick_index, per_pos=FIRST_PICK_PER_POS
    )
    return draft

# The noisy redraws and the rollout playouts are hundreds of independent, seeded draft
# simulations, so they fan out over a process pool (stdlib multiprocessing). Workers get
# the shared inputs once via the initializer; each task is identified by its seed index,
# so results are deterministic regardless of scheduling. The playout worker looks its
# forced candidate up by id in its *own* copy of the pool — Draft stamps `availability_index`
# onto the pool's Player objects, so a separately-pickled Player would carry a stale one.
_WORKER: dict = {}
_FOUR_PICK_BEAM = 16


def _init_worker(
    players, rep, board, stream, noise, seed, opponents, vor, i_my, plans=None
) -> None:
    _WORKER.update(
        players=players, rep=rep, board=board, stream=stream, noise=noise, seed=seed,
        opponents=opponents, vor=vor, i_my=i_my, plans=plans or {},
        by_id={p.player_id: p for p in players},
    )


def _worker_pool_size() -> int:
    return max(1, len(os.sched_getaffinity(0)))


def _target_map(plan: Sequence[int]) -> dict[int, Player]:
    """A player-id plan mapped onto the worker board's next held picks."""
    w = _WORKER
    indices = [i for i, slot in enumerate(w["board"].order) if slot == w["board"].my_slot]
    return {i: w["by_id"][player_id] for i, player_id in zip(indices, plan)}


def _final_roster_value(draft: Draft) -> float:
    """The common end-of-draft objective used by plan screening and noisy rollouts."""
    wire = wire_replacement(draft.taken, by_position(draft.players))
    return team_value(sorted_by_horizon(draft.rosters[draft.board.my_slot - 1]), wire)


def _plan_playout(plan: tuple[int, ...]) -> float:
    """One deterministic full-draft screen for a four-pick target plan."""
    w = _WORKER
    d = Draft(
        w["players"], w["rep"], w["vor"], w["board"], wire=w["stream"],
        opponents=w["opponents"], targets=_target_map(plan),
    )
    d.run()
    return _final_roster_value(d)


def _mc_draft(s: int) -> dict[int, tuple[int, bool]]:
    """One noisy redraw -> per player (pick taken at, taken by my slot?).

    The flag matters because my own simulated picks are this policy's behaviour, not
    market pressure: counting them as takes reported the model's own favourite stashes
    as scarce (a player the policy grabs early looked "gone by 5.03" when in most
    redraws *I* was the one taking him). They cannot just be dropped either — a redraw
    where I took him early observes no opponent demand after that pick — so downstream
    they are censoring times, not events (see build_rankings).
    """
    w = _WORKER
    d = Draft(
        w["players"], w["rep"], w["vor"], w["board"], wire=w["stream"], noise=w["noise"],
        rng=random.Random(w["seed"] + s),
        opponents=w["opponents"],
    )
    d.run()
    slot_of = dict(zip(d.pick_nos, d.order))
    return {pid: (pk, slot_of[pk] == d.my_slot) for pid, pk in d.pick_of.items()}


def _rollout_playout(task: tuple[int, int, int]) -> float:
    cand_id, plan_index, s = task
    w = _WORKER
    plan = w["plans"].get(cand_id, ((cand_id,),))[plan_index]
    d = Draft(
        w["players"], w["rep"], w["vor"], w["board"], wire=w["stream"], noise=w["noise"],
        rng=random.Random(f"rollout-{w['seed']}-{s}"),
        opponents=w["opponents"],
        targets=_target_map(plan), noise_from=w["i_my"] + 1,
    )
    d.run()
    return _final_roster_value(d)


def _best_option_value(draft: Draft) -> float:
    """Best marginal value available to my roster at the current draft state."""
    slot = draft.my_slot
    roster = draft.rosters[slot - 1]
    roster_sorted = sorted_by_horizon(roster)
    base = team_value(roster_sorted, draft.wire)
    off = draft.off_pool[slot - 1]
    values = [
        (
            team_value(roster_sorted, draft.wire, cand) - base
        )
        / draft.roster_depth_penalty(roster, off, cand.position)
        for cand in draft.candidates(
            roster,
            per_pos=1,
            picks_left=draft.picks_left[slot - 1],
            off=off,
        )
    ]
    assert values, "pool exhausted at my following pick"
    return max(values)


def _option_playout(task: tuple[int, int]) -> float:
    """Force one candidate and measure my best option at the following pick."""
    cand_id, s = task
    w = _WORKER
    i_my = w["i_my"]
    assert i_my is not None
    d = Draft(
        w["players"], w["rep"], w["vor"], w["board"], wire=w["stream"], noise=w["noise"],
        rng=random.Random(f"option-{w['seed']}-{s}"),
        opponents=w["opponents"],
        forced={i_my: w["by_id"][cand_id]}, noise_from=i_my + 1,
    )
    i_next = d.next_pick[i_my]
    assert i_next is not None
    d.run(stop_before=i_next)
    return _best_option_value(d)


def monte_carlo(
    players: list[Player],
    rep: dict[str, dict[str, float]],
    board: Board,
    stream: dict[str, dict[str, float]],
    sims: int,
    noise: float,
    seed: int,
    opponents: dict[int, OpponentStrategy],
) -> dict[int, list[tuple[int, bool]]]:
    """Noisy redraws -> per-player (pick, taken-by-me) observations (see `_mc_draft`).

    Each observation records whether my slot made the pick; consumers can count
    opponent takes without maintaining the same information in a second structure.
    `candidate_survival` is the assumption-free counterfactual, priced only for the
    players where the decision actually needs it.
    """
    vor = compute_vor(players, rep)
    picks: dict[int, list[tuple[int, bool]]] = {p.player_id: [] for p in players}
    with multiprocessing.Pool(
        _worker_pool_size(),
        initializer=_init_worker,
        initargs=(players, rep, board, stream, noise, seed, opponents, vor, None),
    ) as pool:
        for pick_of in pool.map(_mc_draft, range(sims)):
            for pid, (pick, mine) in pick_of.items():
                picks[pid].append((pick, mine))
    return picks


def _survival_draft(task: tuple[int, int]) -> int | None:
    cand_id, s = task
    w = _WORKER
    d = Draft(
        w["players"], w["rep"], w["vor"], w["board"], wire=w["stream"], noise=w["noise"],
        rng=random.Random(w["seed"] + s),
        opponents=w["opponents"], my_ban=cand_id,
    )
    return d.run(until_taken=cand_id)


def candidate_survival(
    players: list[Player],
    rep: dict[str, dict[str, float]],
    board: Board,
    stream: dict[str, dict[str, float]],
    candidates: list[Player],
    sims: int,
    noise: float,
    seed: int,
    opponents: dict[int, OpponentStrategy],
) -> dict[int, dict[int, float]]:
    """P(a next-pick candidate is still there at each of my picks) if I keep passing on him.

    "How long can I wait on him" cannot be read off `monte_carlo`: my slot drafts in
    those redraws, and for a player this policy likes it takes him early in most of them,
    which censors the opponents' demand exactly where the question matters. So each
    candidate gets his own redraws with my slot banned from ever taking him (`my_ban`) —
    the other nine teams play exactly as in `monte_carlo` — and availability at my pick
    is simply "no opponent had taken him yet". Priced for my next pick's candidates only:
    one banned-me redraw set per player is too expensive for the whole board.
    """
    if not board.my_picks or not candidates:
        return {}
    vor = compute_vor(players, rep)
    tasks = [(cand.player_id, s) for cand in candidates for s in range(sims)]
    with multiprocessing.Pool(
        _worker_pool_size(),
        initializer=_init_worker,
        initargs=(players, rep, board, stream, noise, seed, opponents, vor, None),
    ) as pool:
        flat = pool.map(_survival_draft, tasks)
    out: dict[int, dict[int, float]] = {}
    for i, cand in enumerate(candidates):
        taken = flat[i * sims : (i + 1) * sims]
        out[cand.player_id] = {
            pick: sum(1 for pk in taken if pk is None or pk >= pick) / sims
            for pick in board.my_picks
        }
    return out


def four_pick_lookahead(
    players: list[Player],
    rep: dict[str, dict[str, float]],
    board: Board,
    stream: dict[str, dict[str, float]],
    candidates: list[Player],
    survival_by_candidate: dict[int, dict[int, float]],
    opponents: dict[int, OpponentStrategy],
    lookahead_picks: int = LOOKAHEAD_PICKS,
) -> dict | None:
    """Choose one target plan per first-pick candidate across my next four held picks.

    Candidate survival supplies the market timing signal the old two-pick policy discarded.
    A small beam proposes sequences by survival-weighted marginal lineup gain; every prefix
    is also retained, so resuming the ordinary policy is an explicit option at turns two,
    three, and four. The best deterministic plan of each length advances; `rollout` applies
    opponent noise and chooses the winning plan for each first candidate.
    """
    if not board.my_picks or not candidates:
        return None
    pick_nos = board.my_picks[:lookahead_picks]
    first_index = board.pick_nos.index(pick_nos[0])
    vor = compute_vor(players, rep)

    # Rebuild the exact deterministic state where the pending decision occurs. The live
    # board can be several opponent picks away from my pick.
    state = Draft(players, rep, vor, board, wire=stream, opponents=opponents)
    state.run(stop_before=first_index)
    roster = state.rosters[board.my_slot - 1]
    off = state.off_pool[board.my_slot - 1]
    picks_left = state.picks_left[board.my_slot - 1]
    by_id = {p.player_id: p for p in candidates}
    candidate_ids = tuple(by_id)
    base = team_value(sorted_by_horizon(roster), state.wire)
    value_cache: dict[tuple[int, ...], float] = {(): base}

    def roster_value(plan: tuple[int, ...]) -> float:
        key = tuple(sorted(plan))
        value = value_cache.get(key)
        if value is None:
            value = team_value(
                sorted_by_horizon(roster + [by_id[player_id] for player_id in plan]),
                state.wire,
            )
            value_cache[key] = value
        return value

    def available_probability(player_id: int, pick_no: int) -> float:
        observations = survival_by_candidate[player_id]
        at_first = observations[pick_nos[0]]
        # Plans are conditional on the first candidate actually being on the board when
        # the decision is made, so later survival is conditioned on reaching that pick.
        return min(1.0, observations[pick_no] / at_first) if at_first else 0.0

    def legal_after(plan: tuple[int, ...], depth: int) -> list[int]:
        future_roster = roster + [by_id[player_id] for player_id in plan]
        remaining = [player_id for player_id in candidate_ids if player_id not in plan]
        for positions, rookies_only in state._eligibility_scenarios(
            future_roster, picks_left - depth, off
        ):
            legal = [
                player_id
                for player_id in remaining
                if by_id[player_id].position in positions
                and (not rookies_only or by_id[player_id].is_rookie)
            ]
            if legal:
                return legal
        return []

    proposed: dict[int, dict[tuple[int, ...], float]] = {}
    for first in candidates:
        first_plan = (first.player_id,)
        states = [(roster_value(first_plan) - base, first_plan)]
        plans = {first_plan: states[0][0]}
        for depth, pick_no in enumerate(pick_nos[1:], start=1):
            expanded: list[tuple[float, tuple[int, ...]]] = []
            for score, plan in states:
                before = roster_value(plan)
                future_roster = roster + [by_id[pid] for pid in plan]
                for player_id in legal_after(plan, depth):
                    next_plan = plan + (player_id,)
                    marginal = (roster_value(next_plan) - before) / state.roster_depth_penalty(
                        future_roster, off, by_id[player_id].position
                    )
                    next_score = score + available_probability(player_id, pick_no) * marginal
                    expanded.append((next_score, next_plan))
            if not expanded:
                break
            expanded.sort(key=lambda row: (-row[0], row[1]))
            states = expanded[:_FOUR_PICK_BEAM]
            plans.update({plan: score for score, plan in states})
        proposed[first.player_id] = plans

    all_plans = [plan for plans in proposed.values() for plan in plans]
    with multiprocessing.Pool(
        _worker_pool_size(),
        initializer=_init_worker,
        initargs=(players, rep, board, stream, 0.0, 0, opponents, vor, first_index),
    ) as pool:
        final_values = pool.map(_plan_playout, all_plans)

    screened: dict[int, dict] = {}
    cursor = 0
    for first in candidates:
        plans = proposed[first.player_id]
        rows = []
        for plan, heuristic_gain in plans.items():
            rows.append((final_values[cursor], heuristic_gain, plan))
            cursor += 1
        finalists = []
        for length in range(1, len(pick_nos) + 1):
            at_length = [row for row in rows if len(row[2]) == length]
            if not at_length:
                continue
            at_length.sort(key=lambda row: (-row[0], -row[1], row[2]))
            final_value, heuristic_gain, plan = at_length[0]
            finalists.append(
                {
                    "target_ids": list(plan),
                    "heuristic_gain": heuristic_gain,
                    "deterministic_ev": final_value,
                }
            )
        screened[first.player_id] = {
            "finalists": finalists,
        }
    return {
        "pick_no": pick_nos[0],
        "pick_nos": pick_nos,
        "depth": len(pick_nos),
        "candidate_count": len(candidates),
        "beam_width": _FOUR_PICK_BEAM,
        "plans": screened,
    }


def option_redraw(
    players: list[Player],
    rep: dict[str, dict[str, float]],
    board: Board,
    stream: dict[str, dict[str, float]],
    candidates: list[Player],
    sims: int,
    noise: float,
    seed: int,
    opponents: dict[int, OpponentStrategy],
) -> dict | None:
    """Empirical next-pick option value for each candidate at my first pending pick.

    Each branch plays deterministically up to my pick, forces its candidate, then applies
    opponent noise only until my following pick. The result is the marginal value of the
    best player actually left there. This deliberately replaces the global-rank survival
    shortcut for the decision in front of me: roster-aware opponents can attack a position
    long before a globally low-ranked player would appear to be at risk.
    """
    if not board.my_picks or not candidates:
        return None
    pick_no = board.my_picks[0]
    i_my = board.pick_nos.index(pick_no)
    i_next = next(
        (i for i in range(i_my + 1, len(board.order)) if board.order[i] == board.my_slot),
        None,
    )
    if i_next is None:
        return {
            "pick_no": pick_no,
            "sims": 0,
            "stats": {cand.player_id: {"ev": 0.0} for cand in candidates},
        }

    vor = compute_vor(players, rep)
    tasks = [(cand.player_id, s) for cand in candidates for s in range(sims)]
    with multiprocessing.Pool(
        _worker_pool_size(),
        initializer=_init_worker,
        initargs=(players, rep, board, stream, noise, seed, opponents, vor, i_my),
    ) as pool:
        flat = pool.map(_option_playout, tasks)
    return {
        "pick_no": pick_no,
        "sims": sims,
        "stats": {
            cand.player_id: {
                "ev": sum(flat[i * sims : (i + 1) * sims]) / sims,
            }
            for i, cand in enumerate(candidates)
        },
    }


def _replay_pick(
    draft: Draft,
    pick_no: int,
    take: Player,
    detail: list[tuple[float, float, Player]],
    players: list[Player],
    rep: dict[str, dict[str, float]],
    board: Board,
    stream: dict[str, dict[str, float]],
    opponents: dict[int, OpponentStrategy],
    targets: dict[int, Player] | None = None,
) -> Draft:
    """Re-play the deterministic draft when a re-scored recommendation changes."""
    actual_id = next((pid for pid, pk in draft.pick_of.items() if pk == pick_no), None)
    if actual_id == take.player_id and not targets:
        draft.my_decisions[pick_no] = detail
        return draft
    vor = compute_vor(players, rep)
    forced = Draft(
        players, rep, vor, board, wire=stream,
        opponents=opponents,
        forced=None if targets else {board.pick_nos.index(pick_no): take},
        targets=targets,
    )
    forced.run()
    forced.my_decisions[pick_no] = detail
    return forced


def apply_option_redraw(
    draft: Draft,
    redrawn: dict | None,
    players: list[Player],
    rep: dict[str, dict[str, float]],
    board: Board,
    stream: dict[str, dict[str, float]],
    opponents: dict[int, OpponentStrategy],
) -> Draft:
    """Replace the first pick's sigmoid lookahead with its branch-redraw estimates."""
    if redrawn is None:
        return draft
    pick_no = redrawn["pick_no"]
    detail = [
        (
            now,
            redrawn["stats"][cand.player_id]["ev"],
            cand,
        )
        for now, _, cand in draft.my_decisions[pick_no]
    ]
    detail.sort(key=lambda t: (-(t[0] + t[1]), t[2].player_id))
    return _replay_pick(
        draft, pick_no, detail[0][2], detail,
        players, rep, board, stream, opponents,
    )


def rollout(
    players: list[Player],
    rep: dict[str, dict[str, float]],
    board: Board,
    stream: dict[str, dict[str, float]],
    candidates: list[Player],
    sims: int,
    noise: float,
    seed: int,
    opponents: dict[int, OpponentStrategy],
    lookahead: dict | None = None,
) -> dict | None:
    """Full-horizon EV for each four-pick plan at my next pick.

    `four_pick_lookahead` supplies the best deterministic target plan of each length for
    every first-pick candidate. A target is exercised if he survives; otherwise that turn
    falls back to the ordinary two-pick policy. After the fourth held pick the ordinary
    policy resumes. Each finalist is played out `sims` times and priced by the final roster
    objective; the highest-EV length represents that first candidate.

    Everything up to my pick plays deterministically (`noise_from`), so all playouts of
    all candidates branch from the same board state — the one `my_decisions` drew its
    candidates from. Playout s uses the same seed for every candidate (common random
    numbers), so `edge` — a candidate's mean paired advantage over the base policy's
    choice — mostly cancels the opponents' noise, and `se` is the standard error of that
    paired difference. `take_id` only overrides the base choice when its edge clears
    2 standard errors; below that the ordering is Monte Carlo noise, not signal, and
    flip-flopping the recommendation between refreshes would be worse than keeping it.
    """
    if not board.my_picks or not candidates:
        return None
    pick_no = board.my_picks[0]
    i_my = board.pick_nos.index(pick_no)
    vor = compute_vor(players, rep)
    plans = {
        cand.player_id: tuple(
            tuple(finalist["target_ids"])
            for finalist in (lookahead or {}).get("plans", {}).get(
                cand.player_id, {"finalists": [{"target_ids": [cand.player_id]}]}
            )["finalists"]
        )
        for cand in candidates
    }
    tasks = [
        (cand.player_id, plan_index, s)
        for cand in candidates
        for plan_index in range(len(plans[cand.player_id]))
        for s in range(sims)
    ]
    with multiprocessing.Pool(
        _worker_pool_size(),
        initializer=_init_worker,
        initargs=(players, rep, board, stream, noise, seed, opponents, vor, i_my, plans),
    ) as pool:
        flat = pool.map(_rollout_playout, tasks)
    values: dict[int, list[float]] = {}
    selected_plans: dict[int, dict] = {}
    cursor = 0
    for cand in candidates:
        finalists = (lookahead or {}).get("plans", {}).get(cand.player_id, {}).get(
            "finalists", [{"target_ids": [cand.player_id]}]
        )
        choices = []
        for finalist in finalists:
            samples = flat[cursor : cursor + sims]
            cursor += sims
            choices.append((sum(samples) / sims, tuple(finalist["target_ids"]), samples, finalist))
        choices.sort(key=lambda row: (-row[0], len(row[1]), row[1]))
        _, _, values[cand.player_id], selected_plans[cand.player_id] = choices[0]

    base = candidates[0]  # my_decisions is sorted by the base policy's score, best first
    stats: dict[int, dict[str, float]] = {}
    for cand in candidates:
        diffs = [a - b for a, b in zip(values[cand.player_id], values[base.player_id])]
        edge = sum(diffs) / sims
        var = sum((x - edge) ** 2 for x in diffs) / (sims - 1) if sims > 1 else 0.0
        stats[cand.player_id] = {
            "ev": sum(values[cand.player_id]) / sims,
            "edge": edge,
            "se": math.sqrt(var / sims),
        }
    take_id = base.player_id
    for cand in candidates:
        s = stats[cand.player_id]
        if s["edge"] > 2 * s["se"] and s["edge"] > stats[take_id]["edge"]:
            take_id = cand.player_id
    return {
        "pick_no": pick_no,
        "pick_nos": (lookahead or {}).get("pick_nos", [pick_no]),
        "sims": sims,
        "take_id": take_id,
        "stats": stats,
        "plans": selected_plans,
    }


def apply_rollout(
    draft: Draft,
    rolled: dict | None,
    players: list[Player],
    rep: dict[str, dict[str, float]],
    board: Board,
    stream: dict[str, dict[str, float]],
    opponents: dict[int, OpponentStrategy],
) -> Draft:
    """Re-play the deterministic draft with the rollout's four-pick target plan.

    Without this, sim_pick and example_draft would show the two-pick policy's choice at my
    next pick while my_next_picks recommends another plan. One extra deterministic draft
    makes every reported block describe the path I am actually being told to play. The
    candidates' two-pick detail is transplanted unchanged: the deterministic prefix up to
    my pick is identical in both drafts, so the scores are too — only the selection
    differs, and everything after it re-plays around that choice.
    """
    if rolled is None:
        return draft
    pick_no = rolled["pick_no"]
    detail = draft.my_decisions[pick_no]
    take = next(c for _, _, c in detail if c.player_id == rolled["take_id"])
    plan_ids = rolled.get("plans", {}).get(take.player_id, {}).get(
        "target_ids", [take.player_id]
    )
    target_indices = [
        i for i, slot in enumerate(board.order) if slot == board.my_slot
    ][: len(plan_ids)]
    by_id = {p.player_id: p for p in players}
    targets = {i: by_id[player_id] for i, player_id in zip(target_indices, plan_ids)}
    return _replay_pick(
        draft, pick_no, take, detail,
        players, rep, board, stream, opponents, targets,
    )

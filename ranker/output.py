"""rankings.json and the stderr reports.

`vor` is the headline number and the sort key: the sum over the two horizons of that
horizon's points minus its converged replacement level (`vor_yr1` + `vor_yr23`).
`my_next_picks` is the direct answer to "who should
I draft next": the model's own choice at each of my next picks, which sees my roster and
the odds a candidate survives to my following pick — so it can disagree with `vor_rank`,
and when it does, it is the better answer.
"""

from __future__ import annotations

import math
import sys

from .board import Board
from .league import (
    BENCH_SLOTS,
    DEPTH_BASE,
    INSURANCE_BASE,
    POINTS_FIELD,
    POSITION_DEPTH_DECAY,
    POSITIONS,
    ROUNDS,
    SCHEME,
    STARTING_SLOTS,
    SURVIVAL_SIGMA,
    TAXI_SLOTS,
    TEAMS,
    TOTAL_PICKS,
    draft_order,
    pick_label,
    picks_for_slot,
)
from .opponents import OpponentStrategy
from .pool import Player
from .sim import Draft
from .value import (
    HORIZONS,
    compute_vor,
    horizon_points,
    pos_by_horizon,
    slot_replacement,
    upside_points,
)


def _sum_levels(levels: dict[str, dict[str, float]]) -> dict[str, float]:
    """Collapse per-horizon levels to their 3-year sum, for the headline columns."""
    return {k: sum(levels[h][k] for h in HORIZONS) for k in POSITIONS}


def build_rankings(
    players: list[Player],
    rep: dict[str, dict[str, float]],
    wire: dict[str, dict[str, float]],
    draft: Draft,
    picks: dict[int, list[int]],
    drafted: dict[int, int],
    sims: int,
    board: Board,
) -> list[dict]:
    """One row per *undrafted* player: the board is a list of decisions still to make.

    Drafted players are dropped rather than flagged — they cannot be picked, and leaving
    them in would put a name at rank 1 that is not available. Their points still shape the
    replacement levels every row is measured against, which happens in `converge`. All
    rank columns are renumbered over what is emitted, so they read as positions on the
    remaining board rather than as gapped survivors of the preseason one.

    Two simulations are reported per player, and they answer different questions:
      * `sim_pick` is from the single deterministic draft, no noise.
      * `sim_adp` / `p_drafted` / `p_available_at_my_picks` come from the noisy redraws
        (adherence-calibrated draws over their provider ranks) and measure the other nine
        teams' demand only. My own simulated picks are this policy's behaviour, not market
        pressure — counting them as takes reported the model's own favourite stashes as
        scarce — but a redraw where I took him early also observes no opponent demand
        afterwards, so availability is a Kaplan-Meier estimate: an opponent take is the
        event, my own take censors the redraw. Under deterministic play availability is
        0 or 1, which tells you nothing about risk, so the noise band is what makes the
        columns usable at the table. It is also where uncertainty in the inferred source
        policies belongs.
    """
    my_picks = board.my_picks
    vor = compute_vor(players, rep)
    rep_sum, wire_sum = _sum_levels(rep), _sum_levels(wire)
    available = board.available(players)
    ranked = sorted(available, key=lambda p: (-vor[p.player_id], p.player_id))
    opponent_rank = {
        p.player_id: i
        for i, p in enumerate(
            sorted(
                available,
                key=lambda p: (draft.opponent_consensus_rank[p.player_id], p.player_id),
            ),
            start=1,
        )
    }
    pos_rank: dict[str, int] = {pos: 0 for pos in POSITIONS}
    rows: list[dict] = []
    for i, p in enumerate(ranked, start=1):
        pos_rank[p.position] += 1
        seen = picks[p.player_id]  # (pick, taken_by_me) per redraw he was drafted in
        # P(no opponent has taken him before each of my picks), Kaplan-Meier: walking the
        # takes in pick order, an opponent take drops survival by 1/at_risk; my own take
        # censors the redraw (removes it from at_risk without an event) — after it that
        # redraw can no longer show opponent demand, and counting it as either scarcity
        # or availability was wrong in turn. Ties sort events before censors ((pick,
        # False) < (pick, True)), the conservative convention. KM assumes my take times
        # are independent of opponent demand — my slot drafts noiselessly, so roughly
        # true; `p_available_if_i_pass` in my_next_picks is the assumption-free
        # counterfactual where the decision needs one. Only uncertain entries are
        # emitted; a 0.0 or 1.0 carries no information for a draft board. `latest` is
        # the last pick still at even odds, not the first (which is trivially 1.02 for
        # everybody except whoever goes 1.01).
        avail = {}
        latest = None
        surv, at_risk, j = 1.0, sims, 0
        removals = sorted(seen)
        for pick in my_picks:
            while j < len(removals) and removals[j][0] < pick:
                if not removals[j][1]:
                    surv *= 1.0 - 1.0 / at_risk
                at_risk -= 1
                j += 1
            if 0.01 < surv < 0.99:
                avail[pick_label(pick)] = round(surv, 3)
            if surv >= 0.5:
                latest = pick_label(pick)
        sim_pick = draft.pick_of.get(p.player_id)
        opp = [pk for pk, mine in seen if not mine]
        mean = sum(opp) / len(opp) if opp else None
        sd = (
            math.sqrt(sum((x - mean) ** 2 for x in opp) / len(opp))
            if len(opp) > 1
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
                "points_1yr": p.points_1yr,
                # Diagnostic of growth shape: his 3-year total at his years-2-3 pace.
                # Equal to points_3yr for a flat scorer; the gap is the provider's
                # implied growth. See value.upside_points.
                "upside_points": round(upside_points(p), 1),
                # Sum of the per-horizon levels, so vor = points_3yr - replacement_points
                # stays verifiable from this row; the split is in vor_yr1 / vor_yr23.
                "replacement_points": round(rep_sum[p.position], 1),
                "vor": round(vor[p.player_id], 1),
                "vor_yr1": round(horizon_points(p, "yr1") - rep["yr1"][p.position], 1),
                "vor_yr23": round(horizon_points(p, "yr23") - rep["yr23"][p.position], 1),
                "vor_vs_wire": round(p.points - wire_sum[p.position], 1),
                "sim_pick": sim_pick,
                "sim_pick_label": pick_label(sim_pick) if sim_pick else None,
                "sim_adp": round(mean, 1) if mean is not None else None,
                "sim_adp_sd": round(sd, 1) if sd is not None else None,
                "p_drafted": round(drafted[p.player_id] / sims, 3),
                "latest_my_pick_likely_available": latest,
                "p_available_at_my_picks": avail,
                "provider_adp": p.provider_adp,
                "opponent_consensus_rank": opponent_rank[p.player_id],
                # Positive means my VOR board values him earlier than the average of the
                # nine slot-specific provider boards actually used in the simulations.
                "opponent_rank_delta": opponent_rank[p.player_id] - i,
            }
        )
    return rows


def my_next_picks(
    draft: Draft,
    board: Board,
    rollout: dict | None = None,
    survival: dict[int, dict[int, float]] | None = None,
    limit: int = 3,
) -> list[dict]:
    """The model's own decision at each of my next picks, from the deterministic draft.

    This is the question the whole script exists to answer, so it is surfaced rather than
    left implicit in `sim_pick`. The candidates carry the two parts of the decision score:
    what the player adds to my roster now, and the expected value of the best player still
    there at my following pick if I take him.

    At my first pending pick, `next_pick_ev` comes from short branch-specific opponent
    redraws rather than the global-rank survival shortcut used by the bulk draft policy.
    That pick additionally carries the full-horizon rollout (sim.rollout):
    `rollout_ev` is the mean final value of my whole roster if I take the candidate and
    the rest of the draft plays out, `rollout_edge` is his paired advantage over the base
    policy's choice, `rollout_se` its standard error. The `take` for that pick is the
    rollout's — it can overrule the two-pick score, but only when the edge is clearly
    above the playout noise.

    Its candidates also carry `p_available_if_i_pass` (sim.candidate_survival): across
    redraws where my slot is banned from ever taking him, the share where no opponent
    has taken him before each of my picks — the honest "how long can I wait on him".
    Unlike the Kaplan-Meier `p_available_at_my_picks`, it needs no independence
    assumption: the redraws actually play the pass-on-him counterfactual out.
    """
    out: list[dict] = []
    for pick_no in board.my_picks[:limit]:
        detail = draft.my_decisions.get(pick_no)
        if not detail:
            continue
        rolled = rollout if rollout and rollout["pick_no"] == pick_no else None
        take = detail[0][2]
        if rolled:
            take = next(c for _, _, c in detail if c.player_id == rolled["take_id"])
        candidates = []
        for now, later, c in detail:
            row = {
                "player_id": c.player_id,
                "name": c.name,
                "position": c.position,
                "value_now": round(now, 1),
                "next_pick_ev": round(later, 1),
                "score": round(now + later, 1),
            }
            if rolled:
                s = rolled["stats"][c.player_id]
                row["rollout_ev"] = round(s["ev"], 1)
                row["rollout_edge"] = round(s["edge"], 1)
                row["rollout_se"] = round(s["se"], 1)
            if survival and pick_no == board.my_picks[0] and c.player_id in survival:
                # Same emission rule as p_available_at_my_picks: certainties are noise.
                row["p_available_if_i_pass"] = {
                    pick_label(pk): round(p, 3)
                    for pk, p in survival[c.player_id].items()
                    if 0.01 < p < 0.99
                }
            candidates.append(row)
        out.append(
            {
                "pick": pick_label(pick_no),
                "overall": pick_no,
                "take_id": take.player_id,
                "take": f"{take.name} ({take.position})",
                "candidates": candidates,
            }
        )
    return out


def _team_names(board: Board) -> dict[int, str | None]:
    return {
        s["draft_slot"]: (s.get("team_name") or s.get("username"))
        for s in ((board.live or {}).get("slots") or [])
    }


def draft_block(board: Board) -> dict | None:
    """How the output describes the board it started from. None when there was no live one."""
    if not board.live:
        return None
    names = _team_names(board)
    block = {k: v for k, v in board.live.items() if k != "slots"}
    block["my_remaining_picks"] = [pick_label(n) for n in board.my_picks]
    block["note"] = (
        "The simulation starts here: made picks are already on their teams' rosters and out "
        "of the pool, and only the pending picks are played out, in this order — so a traded "
        "pick is exercised by the roster that acquired it. `rankings` covers the undrafted "
        "players only."
    )
    block["off_pool_note"] = (
        "Made picks with no match in pool.json by sleeper_id: kickers, IDP and anyone past "
        "the pool's rank cut. They fill a roster spot and satisfy a mandatory position, so "
        "the team owes one fewer pick, but they are never started and never valued — there "
        "is no projection to value them with."
    )
    block["rosters"] = []
    for slot in range(1, TEAMS + 1):
        made, off = board.rosters[slot - 1], board.off_pool[slot - 1]
        counts = {pos: 0 for pos in POSITIONS}
        for p in made:
            counts[p.position] += 1
        for o in off:
            if o.get("position") in counts:
                counts[o["position"]] += 1
        block["rosters"].append(
            {
                "draft_slot": slot,
                "team": names.get(slot),
                "is_mine": slot == board.my_slot,
                "picks_made": len(made) + len(off),
                "picks_left": board.picks_left[slot - 1],
                "positions": {pos: n for pos, n in counts.items() if n},
                "players": [f"{p.name} ({p.position})" for p in made],
                "off_pool": [f"{o['name']} ({o['position']})" for o in off],
            }
        )
    return block


def example_rosters(draft: Draft, board: Board) -> list[dict]:
    """Every team's final roster from the deterministic draft, for eyeballing smells.

    Rosters read in pick order: the live board's made picks first (the board does not keep
    their pick numbers), then the simulated picks with the pick they were taken at. Each
    entry carries the projection fields the dashboard needs; rankings only contains
    undrafted players, so looking rostered players up there leaves every live pick blank.
    """
    names = _team_names(board)
    out: list[dict] = []
    for slot in range(1, TEAMS + 1):
        roster, off = draft.rosters[slot - 1], draft.off_pool[slot - 1]
        counts = {pos: 0 for pos in POSITIONS}
        for p in roster:
            counts[p.position] += 1
        for o in off:
            if o.get("position") in counts:
                counts[o["position"]] += 1
        picks: list[dict] = []
        for p in roster:
            pick_no = draft.pick_of.get(p.player_id)
            future_per_year = (p.points - p.points_1yr) / 2
            picks.append(
                {
                    "pick": pick_label(pick_no) if pick_no else None,
                    "is_made": pick_no is None,
                    "player_id": p.player_id,
                    "name": p.name,
                    "position": p.position,
                    "nfl_team": p.team,
                    "age": p.age,
                    "is_rookie": p.is_rookie,
                    "points_1yr": p.points_1yr,
                    "points_3yr": p.points,
                    "future_points_per_year": round(future_per_year, 1),
                    "growth_per_year": round(future_per_year - p.points_1yr, 1),
                    "off_pool": False,
                }
            )
        picks += [
            {
                "pick": o["pick"],
                "is_made": True,
                "player_id": None,
                "name": o["name"],
                "position": o["position"],
                "nfl_team": None,
                "age": None,
                "is_rookie": None,
                "points_1yr": None,
                "points_3yr": None,
                "future_points_per_year": None,
                "growth_per_year": None,
                "off_pool": True,
            }
            for o in off
        ]
        out.append(
            {
                "draft_slot": slot,
                "team": names.get(slot),
                "is_mine": slot == board.my_slot,
                "players": len(roster) + len(off),
                "rookies": sum(p.is_rookie for p in roster),
                "positions": counts,
                "picks": picks,
            }
        )
    return out


def build_payload(
    players: list[Player],
    pool_meta: dict,
    board: Board,
    rep: dict[str, dict[str, float]],
    stream: dict[str, dict[str, float]],
    counts: dict[str, dict[str, int]],
    draft: Draft,
    history: dict,
    rows: list[dict],
    problems: list[str],
    sims: int,
    noise: float,
    seed: int,
    opponents: dict[int, OpponentStrategy],
    option_redraw: dict | None = None,
    rollout: dict | None = None,
    survival: dict[int, dict[int, float]] | None = None,
) -> dict:
    pos_h = pos_by_horizon(players)
    return {
        "generated_from": pool_meta["source_file"],
        "scoring_scheme": SCHEME,
        "value_input": f"pool.json {POINTS_FIELD} + points_1yr ({SCHEME})",
        "value_note": (
            "Projected points in this league's scheme, split into two horizons — year 1 "
            "(points_1yr) and years 2-3 (points_3yr - points_1yr) — because lineups are "
            "fielded per season: the starting lineup is solved per horizon against that "
            "horizon's replacement levels, so an injury-wrecked year 1 cannot hide inside "
            "a healthy 3-year sum. Draftsharks' 3D value is deliberately unused: it is a "
            "provider-scaled ordinal, not points, so it cannot be differenced against a "
            "replacement level."
        ),
        "league": {
            "teams": TEAMS,
            "starting_slots": STARTING_SLOTS,
            "bench_slots": BENCH_SLOTS,
            "taxi_slots": TAXI_SLOTS,
            "rounds": ROUNDS,
            "total_picks": TOTAL_PICKS,
            "draft_type": "snake with 3rd-round reversal",
            "my_slot": board.my_slot,
            "my_picks": [pick_label(p) for p in picks_for_slot(board.my_slot, draft_order())],
        },
        "pool": pool_meta,
        "draft": draft_block(board),
        "example_draft": {
            "note": (
                "Full final rosters from the single deterministic draft at the converged "
                "levels — the same draft sim_pick and my_next_picks report. Made picks are "
                "facts from the live board; every pending pick is the model drafting. Read "
                "it for smells: position hoarding, a position left to the last rounds, "
                "fewer than 4 rookies to fill the taxi spots."
            ),
            "rosters": example_rosters(draft, board),
        },
        "my_next_picks": {
            "note": (
                "The model's own choice at each of my next picks, from the deterministic "
                "draft at the converged levels. value_now is the candidate's marginal value "
                "to my roster (lineup surplus + bench depth); next_pick_ev is E[value of the "
                "best player still there at my following pick] if I take him now. The pick "
                "maximizes their sum, so it can disagree with vor_rank — it sees my roster "
                "and who will keep, which vor does not. At my first pending pick, each "
                "next_pick_ev forces that candidate, redraws only the intervening opponents, "
                "and measures the best marginal option actually left at my following turn; "
                "later displayed picks retain the fast global-rank approximation used by "
                "the bulk draft policy. My first pending pick is additionally scored over "
                "the whole remaining draft (rollout_ev/rollout_edge/rollout_se): "
                "each candidate is forced there and the rest of the draft is played out, my "
                "future picks by the two-pick policy, the other teams noisy. Its take stands "
                "when a candidate's full-horizon edge over the two-pick choice clears twice "
                "its standard error, and when it overrules, the deterministic draft is "
                "re-played with that pick forced — so sim_pick, example_draft and the later "
                "picks here all describe the recommended path. Its candidates also carry "
                "p_available_if_i_pass — P(no opponent has taken him before each of my "
                "picks) across redraws where my slot never takes him: the honest 'how "
                "long can I wait', with the pass-on-him counterfactual actually played "
                "out rather than estimated (compare p_available_at_my_picks, which is "
                "Kaplan-Meier from redraws where I do take him)."
            ),
            "option_sims": option_redraw["sims"] if option_redraw else None,
            "rollout_sims": rollout["sims"] if rollout else None,
            "picks": my_next_picks(draft, board, rollout, survival),
        },
        "rankings_note": (
            "Undrafted players only, ranked over each other. `vor` is a points quantity "
            "and does not depend on who is gone; the rank columns are renumbered over "
            "the rows emitted here."
            if board.live
            else "The whole pool, from an empty board: no live draft was read."
        ),
        "opponent_model": {
            "who": "the other nine teams; my slot alone uses VOR and roster value",
            "how": (
                "Each opponent orders legal available players by the provider board most "
                "associated with its completed picks in data_source_matches.json. Opponent "
                "valuation never reads projections, replacement levels, team value, or VOR. "
                "The deterministic draft takes the source's top legal player; Monte Carlo "
                "draws around that order using the manager's observed source adherence."
            ),
            "adherence": (
                "The investigator's mean_log2_loss is converted to a power distribution "
                "over source rank among legal available players. At --noise 1, its expected "
                "log2 choice rank matches that manager's observed loss; 0 forces strict "
                "source order. fit_score and confidence identify association strength, "
                "while mean_log2_loss determines adherence."
            ),
            "coverage": (
                "A provider's normalized players come first. Any pool player it does not "
                "rank is appended in DraftSharks ADP order so all 290 picks remain possible; "
                "the fallback is still an external opponent board, never VOR."
            ),
            "delta": (
                "opponent_consensus_rank averages the nine managers' complete source "
                "orders, counting a source once per associated manager, then re-ranks the "
                "available pool. opponent_rank_delta is opponent_consensus_rank - vor_rank; "
                "positive identifies players my VOR process values earlier than the modeled "
                "field. Monte Carlo availability determines whether that gap is exploitable."
            ),
            "divergence": {
                "distinct_sources": len({s.source_id for s in opponents.values()}),
                "mean_absolute_rank_delta": round(
                    sum(abs(row["opponent_rank_delta"]) for row in rows) / len(rows), 1
                ),
                "max_absolute_rank_delta": max(
                    abs(row["opponent_rank_delta"]) for row in rows
                ),
            },
            "strategies": [opponents[slot].public() for slot in sorted(opponents)],
        },
        "replacement": {
            "definition": (
                "Per horizon (yr1 = year 1, yr23 = years 2-3): that horizon's points of "
                "the (S+1)-th best player at a position by those points, where S is how "
                "many players at that position hold a starting slot across all 10 teams "
                "in the simulated draft when lineups are set on that horizon. This is the "
                "marginal starter of that period. `levels` is the horizon sum, so "
                "vor = points_3yr - levels[pos]; the split is in levels_by_horizon."
            ),
            "levels": {k: round(v, 1) for k, v in _sum_levels(rep).items()},
            "levels_by_horizon": {
                h: {k: round(v, 1) for k, v in rep[h].items()} for h in HORIZONS
            },
            "starters_by_position": counts,
            "marginal_starter": {
                h: {
                    k: {
                        "rank": f"{k}{counts[h][k] + 1}",
                        "player": pos_h[h][k][min(counts[h][k], len(pos_h[h][k]) - 1)].name,
                    }
                    for k in POSITIONS
                }
                for h in HORIZONS
            },
            "wire_levels": {k: round(v, 1) for k, v in _sum_levels(stream).items()},
            "wire_levels_by_horizon": {
                h: {k: round(v, 1) for k, v in stream[h].items()} for h in HORIZONS
            },
            "wire_note": (
                "The best player at each position left undrafted — the post-draft free "
                "agent baseline, per horizon (the best year-1 add and the best years-2-3 "
                "stash can differ). It is what `vor_vs_wire` measures against (as the "
                "horizon sum), and inside the simulation it prices bench depth. Far below "
                "the marginal-starter level because 290 of the pool's 350 players get "
                "rostered."
            ),
            "slot_levels": {
                h: {k: round(v, 1) for k, v in lv.items()}
                for h, lv in slot_replacement(rep).items()
            },
            "slot_levels_note": (
                "What an empty starting slot is priced at inside the simulation: the "
                "marginal-starter level of the best position the slot accepts, so a flex "
                "slot is max(RB, WR, TE) and a player is worth strictly less there than in "
                "his dedicated slot — that is where positional scarcity comes from. "
                "Marginal-starter, not the wire, because skipping a position early means "
                "ending the draft with a late-round starter there, not with a free agent "
                "(see slot_replacement in ranker/value.py)."
            ),
            "convergence": history,
        },
        "strategy": {
            "objective": (
                "sum over horizons (year 1, years 2-3) of starting-lineup points above "
                "that horizon's replacement + decayed bench value (growth + insurance)"
            ),
            "depth_base": DEPTH_BASE,
            "position_depth_decay": POSITION_DEPTH_DECAY,
            "insurance_base": INSURANCE_BASE,
            "depth_note": (
                "Bench value decays with how many players that team already has at the "
                "same position, not with a running bench index — a fifth QB cannot start "
                "in a two-QB-slot league however large his VOR looks. Each body benched "
                "in a horizon is priced on that horizon's excess over its wire: in year 1 "
                "insurance only — insurance_base, his position's expected share of starter "
                "games missed to byes and injury; a player who cannot play this season "
                "cannot cover it — and in years 2-3 growth (depth_base, the chance he "
                "grows past a starter) plus the same insurance. So a backloaded rookie "
                "and a startable veteran are both worth bench picks for different reasons, "
                "each in the horizon where he actually produces."
            ),
            "lookahead": "value now + E[best value still available at my next pick]",
            "survival_sigma": SURVIVAL_SIGMA,
            "note": (
                "Two-pick rollout with an independence approximation across candidates; "
                "a strong greedy, not equilibrium play. The first pending pick replaces "
                "that approximation with short opponent redraws and gets a full-horizon "
                "check on top — see my_next_picks."
            ),
        },
        "monte_carlo": {
            "sims": sims,
            "noise": noise,
            "seed": seed,
            "note": (
                "Source-adherence Gumbel draws on the other 9 teams only, to turn 0/1 "
                "availability under deterministic source order into a usable probability "
                "band. At noise=1 each manager's distribution is calibrated to its observed "
                "mean log-rank loss; noise=0 is strict source order. sim_pick is from the "
                "noiseless draft; sim_adp, p_drafted and p_available_at_my_picks are from "
                "these redraws and measure the other nine teams' demand only — my own "
                "simulated picks are the VOR policy under evaluation, not opponent demand. "
                "p_available_at_my_picks is a Kaplan-Meier estimate (an opponent take is "
                "the event, my own take censors the redraw); "
                "my_next_picks.p_available_if_i_pass is the assumption-free counterfactual "
                "for the candidates that matter. sim_adp and p_drafted are over observed "
                "opponent takes, so both read shallow for a player this policy usually "
                "grabs first."
            ),
        },
        "validation": {"problems": problems, "ok": not problems},
        "count": len(rows),
        "rankings": rows,
    }


# --- stderr reports ---------------------------------------------------------------


def report_board(board: Board) -> None:
    """The starting state, on stderr: what is gone, who holds it, what is still coming."""
    if not board.live:
        print(
            f"board: no live draft; all {len(board.order)} picks simulated from the static "
            "snake",
            file=sys.stderr,
        )
        return
    live = board.live
    print(
        f"board: {live['picks_made']}/{TOTAL_PICKS} picks made, {live['picks_pending']} "
        f"pending ({live['matched_to_pool']} made picks joined to the pool, "
        f"{len(live['off_pool_picks'])} outside it); {live['status']}, "
        f"fetched {live['fetched_at']}",
        file=sys.stderr,
    )
    if live["traded_picks"]:
        print(
            f"  {live['traded_picks']} traded pick(s), applied by the draft pipeline",
            file=sys.stderr,
        )
    for o in live["off_pool_picks"]:
        print(
            f"  {o['pick']} slot {o['slot']}: {o['name']} ({o['position']}) is not in the "
            "pool - held as a filled roster spot with no value",
            file=sys.stderr,
        )
    names = _team_names(board)
    for slot in range(1, TEAMS + 1):
        made, off = board.rosters[slot - 1], board.off_pool[slot - 1]
        if not made and not off:
            continue
        who = names.get(slot) or f"slot {slot}"
        held = [f"{p.name} ({p.position})" for p in made]
        held += [f"{o['name']} ({o['position']}, unvalued)" for o in off]
        print(
            f"  {'*' if slot == board.my_slot else ' '} slot {slot:>2} {who[:20]:<20}"
            f" {board.picks_left[slot - 1]:>2} picks left: " + ", ".join(held),
            file=sys.stderr,
        )
    clock = live.get("on_the_clock") or {}
    mine = live.get("next_pick_of_mine") or {}
    print(
        f"  on the clock {clock.get('slot')} ({clock.get('username')}); my next "
        f"{mine.get('slot')} ({mine.get('picks_away')} away), "
        f"{len(board.my_picks)} of my picks remain",
        file=sys.stderr,
    )


def report_summary(
    rows: list[dict],
    rep: dict[str, dict[str, float]],
    counts: dict[str, dict[str, int]],
    draft: Draft,
    board: Board,
    rollout: dict | None = None,
    survival: dict[int, dict[int, float]] | None = None,
) -> None:
    """Top of the board, the recommendation, and the levels, on stderr."""
    top = rows[:12]
    if top:
        width = max(len(r["name"]) for r in top)
        print(
            "\ntop of the board" + (f", {len(rows)} undrafted:" if board.live else ":"),
            file=sys.stderr,
        )
    for r in top:
        print(
            f"  {r['vor_rank']:>3}. {r['name']:<{width}}  {r['position']}"
            f"{r['positional_vor_rank']:<3} vor {r['vor']:>7.1f}"
            f"  pts {r['points_3yr']:>5}  sim {r['sim_pick_label'] or '--':>6}"
            f"  provider adp {r['provider_adp'] or float('nan'):>5}",
            file=sys.stderr,
        )
    recs = my_next_picks(draft, board, rollout)
    if recs:
        print("\nmy next picks (value now + E[left at my following pick]):", file=sys.stderr)
        for rec in recs:
            cands = ", ".join(
                f"{c['name']} {c['score']:.0f} ({c['value_now']:.0f}+{c['next_pick_ev']:.0f})"
                for c in rec["candidates"]
            )
            print(f"  {rec['pick']}: take {rec['take']}  |  {cands}", file=sys.stderr)
        first = recs[0]
        if "rollout_ev" in first["candidates"][0]:
            cands = ", ".join(
                f"{c['name']} edge {c['rollout_edge']:+.0f}±{c['rollout_se']:.0f}"
                for c in first["candidates"]
            )
            print(f"  {first['pick']} full-horizon rollout: {cands}", file=sys.stderr)
        if survival:
            # Last of my picks each candidate survives to at even odds if I keep passing
            # on him ('--' = likely gone before my current pick comes back around).
            parts = []
            for _, _, c in draft.my_decisions.get(board.my_picks[0], []):
                sv = survival.get(c.player_id)
                if not sv:
                    continue
                last = None
                for pk in board.my_picks:
                    if sv[pk] < 0.5:
                        break
                    last = pk
                parts.append(f"{c.name} {pick_label(last) if last else '--'}")
            print(
                f"  {first['pick']} last even-odds pick if I pass: " + ", ".join(parts),
                file=sys.stderr,
            )
    for h in HORIZONS:
        print(
            f"\nreplacement {h}: "
            + ", ".join(f"{k}{counts[h][k] + 1} = {rep[h][k]:.0f}" for k in POSITIONS),
            file=sys.stderr,
        )

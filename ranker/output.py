"""Assemble the documented rankings.json payload."""

from __future__ import annotations

from .board import Board
from .league import (
    BENCH_SLOTS,
    DEPTH_BASE,
    FIRST_PICK_PER_POS,
    INSURANCE_BASE,
    LOOKAHEAD_PICKS,
    POINTS_FIELD,
    POSITION_DEPTH_DECAY,
    POSITIONS,
    ROSTER_DEPTH_PENALTY,
    ROSTER_DEPTH_TARGETS,
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
from .rankings import _sum_levels, my_next_picks
from .simulation import Draft
from .value import HORIZONS, pos_by_horizon, slot_replacement


def team_names(board: Board) -> dict[int, str | None]:
    return {
        s["draft_slot"]: (s.get("team_name") or s.get("username"))
        for s in ((board.live or {}).get("slots") or [])
    }


def draft_block(board: Board) -> dict | None:
    """How the output describes the board it started from. None when there was no live one."""
    if not board.live:
        return None
    names = team_names(board)
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
    names = team_names(board)
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
                "to my roster (lineup surplus + bench depth), divided by the soft depth "
                "penalty when the position is already deep; next_pick_ev is E[value of the "
                "best player still there at my following pick], with the same adjustment, "
                "if I take him now. The pick "
                "maximizes their sum, so it can disagree with vor_rank — it sees my roster "
                "and who will keep, which vor does not. At my first pending pick, each "
                "next_pick_ev forces that candidate, redraws only the intervening opponents, "
                "and measures the best marginal option actually left at my following turn; "
                "later displayed picks retain the fast global-rank approximation used by "
                "the bulk draft policy. My first pending pick searches target plans across "
                "my next four held picks; every shorter prefix is eligible, so the ordinary "
                "policy can resume at any turn. The best deterministic plan of each length "
                "is then scored over the whole remaining draft, and the best noisy EV "
                "represents that first candidate "
                "(rollout_ev/rollout_edge/rollout_se), with a planned target used only if he "
                "survives and the ordinary policy used otherwise. Its take stands "
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
            "lookahead_picks": LOOKAHEAD_PICKS,
            "first_pick_candidates_per_position_horizon": FIRST_PICK_PER_POS,
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
                "A position's source rank receives a soft boost in proportion to its "
                "unfilled dedicated starters and a compounding penalty beyond its "
                "comfortable depth; the deterministic draft takes the best adjusted "
                "rank, and Monte Carlo draws around that preference."
            ),
            "depth_preference": {
                "targets": ROSTER_DEPTH_TARGETS,
                "penalty_per_extra_player": ROSTER_DEPTH_PENALTY,
                "note": (
                    "Targets sum to 25, leaving four roster spots to spill into the best "
                    "remaining positions. This adjusts source rank and is not a draft limit."
                ),
            },
            "adherence": (
                "The investigator's mean_log2_loss is converted to a power distribution "
                "over source rank among legal available players, before the roster-balance "
                "adjustment. --noise 1 uses that fitted randomness; 0 removes randomness "
                "but retains the balance adjustment. fit_score and confidence identify "
                "association strength, while mean_log2_loss determines adherence."
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
            "roster_depth_preference": {
                "targets": ROSTER_DEPTH_TARGETS,
                "penalty_per_extra_player": ROSTER_DEPTH_PENALTY,
                "note": (
                    "The simulated pick score divides marginal gain by this compounding "
                    "penalty. It shapes choices but does not change final roster value or "
                    "act as a positional limit. Validation flags a deterministic roster "
                    "more than five players beyond any target."
                ),
            },
            "lookahead": (
                "first pending decision: four held picks with survival-aware target plans; "
                "bulk policy: value now + E[best value at the following pick]"
            ),
            "survival_sigma": SURVIVAL_SIGMA,
            "note": (
                "The bulk policy is a two-pick greedy with an independence approximation. "
                "The first pending pick uses banned-me availability redraws to build plans "
                "across four held picks, then noisy full-draft rollouts choose among them — "
                "see my_next_picks."
            ),
        },
        "monte_carlo": {
            "sims": sims,
            "noise": noise,
            "seed": seed,
            "note": (
                "Balance-adjusted source-rank Gumbel draws on the other 9 teams only, to "
                "turn 0/1 availability under the deterministic preference into a usable "
                "probability band. At noise=1 the source-rank component is calibrated to "
                "each manager's observed mean log-rank loss before the balance adjustment; "
                "noise=0 removes random variation but retains that adjustment. sim_pick is "
                "from the noiseless draft; sim_adp, p_drafted and "
                "p_available_at_my_picks are from these redraws and measure the other nine "
                "teams' demand only — my own "
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

"""rankings.json and the stderr reports.

`vor` is the headline number and the sort key: 3-year points minus the converged
replacement level for that position. `my_next_picks` is the direct answer to "who should
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
from .pool import Player, by_position
from .sim import Draft
from .value import compute_vor, slot_replacement, upside_points


def build_rankings(
    players: list[Player],
    rep: dict[str, float],
    wire: dict[str, float],
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
    three rank columns are renumbered over what is emitted, so they read as positions on
    the remaining board rather than as gapped survivors of the preseason one.

    Two simulations are reported per player, and they answer different questions:
      * `sim_pick` is from the single deterministic draft, no noise.
      * `sim_adp` / `p_available_at_my_picks` come from the noisy redraws (Gumbel noise on
        the other teams' scores). Under deterministic play availability is 0 or 1, which
        tells you nothing about risk, so the noise band is what makes the columns usable
        at the table. It is also where the uncertainty in the ADP signal belongs.
    """
    my_picks = board.my_picks
    vor = compute_vor(players, rep)
    available = board.available(players)
    ranked = sorted(available, key=lambda p: (-vor[p.player_id], p.player_id))
    market_rank = {
        p.player_id: i
        for i, p in enumerate(
            sorted(
                available,
                key=lambda p: (
                    p.provider_adp if p.provider_adp is not None else math.inf,
                    p.player_id,
                ),
            ),
            start=1,
        )
    }
    pos_rank: dict[str, int] = {pos: 0 for pos in POSITIONS}
    rows: list[dict] = []
    for i, p in enumerate(ranked, start=1):
        pos_rank[p.position] += 1
        seen = picks[p.player_id]
        # Availability at each of my picks, across the noisy redraws. Only the uncertain
        # ones are emitted; a 0.0 or 1.0 entry carries no information for a draft board.
        # `latest` is the actionable one — how long I can wait on him — so it is the last
        # pick still at even odds, not the first (which is trivially 1.02 for everybody
        # except whoever goes 1.01).
        avail = {}
        latest = None
        for pick in my_picks:
            # `>=`, not `>`: a player taken *at* one of my picks was taken by me, so he was
            # on the board when I got there. Using `>` reported the board's top TE as
            # unavailable at 1.02 in exactly the sims where he was my 1.02 pick.
            prob = (sum(1 for s in seen if s >= pick) + (sims - len(seen))) / sims
            if 0.01 < prob < 0.99:
                avail[pick_label(pick)] = round(prob, 3)
            if prob >= 0.5:
                latest = pick_label(pick)
        sim_pick = draft.pick_of.get(p.player_id)
        mean = sum(seen) / len(seen) if seen else None
        sd = (
            math.sqrt(sum((x - mean) ** 2 for x in seen) / len(seen))
            if seen and len(seen) > 1
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
                # The bench-pricing quantity: his 3-year total at his years-2-3 pace.
                # Equal to points_3yr for a flat scorer; the gap is the provider's
                # implied growth. See value.upside_points.
                "upside_points": round(upside_points(p), 1),
                "replacement_points": round(rep[p.position], 1),
                "vor": round(vor[p.player_id], 1),
                "vor_vs_wire": round(p.points - wire[p.position], 1),
                "sim_pick": sim_pick,
                "sim_pick_label": pick_label(sim_pick) if sim_pick else None,
                "sim_adp": round(mean, 1) if mean is not None else None,
                "sim_adp_sd": round(sd, 1) if sd is not None else None,
                "p_drafted": round(drafted[p.player_id] / sims, 3),
                "latest_my_pick_likely_available": latest,
                "p_available_at_my_picks": avail,
                "provider_adp": p.provider_adp,
                "market_rank": market_rank[p.player_id],
                # Divergence from a fuzzy proxy, not a measured mispricing. The source ADP
                # is a 12-team no-TE-premium board, so a large delta at TE mostly reflects
                # what the signal cannot see rather than what opponents will do.
                "adp_rank_delta": market_rank[p.player_id] - i,
            }
        )
    return rows


def my_next_picks(
    draft: Draft, board: Board, rollout: dict | None = None, limit: int = 3
) -> list[dict]:
    """The model's own decision at each of my next picks, from the deterministic draft.

    This is the question the whole script exists to answer, so it is surfaced rather than
    left implicit in `sim_pick`. The candidates carry the two parts of the decision score:
    what the player adds to my roster now, and the expected value of the best player still
    there at my following pick if I take him.

    My first pending pick additionally carries the full-horizon rollout (sim.rollout):
    `rollout_ev` is the mean final value of my whole roster if I take the candidate and
    the rest of the draft plays out, `rollout_edge` is his paired advantage over the base
    policy's choice, `rollout_se` its standard error. The `take` for that pick is the
    rollout's — it can overrule the two-pick score, but only when the edge is clearly
    above the playout noise.
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
            candidates.append(row)
        out.append(
            {
                "pick": pick_label(pick_no),
                "overall": pick_no,
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

    Rosters read in pick order: the live board's made picks first (labelled 'made' — the
    board does not keep their pick numbers), then the simulated picks with the pick they
    were taken at. Rookies are flagged because the 4 taxi spots mean every team must end
    the draft with at least 4 of them.
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
        picks = []
        for p in roster:
            pick_no = draft.pick_of.get(p.player_id)
            label = pick_label(pick_no) if pick_no else "made"
            picks.append(f"{label} {p.name} ({p.position}{', R' if p.is_rookie else ''})")
        picks += [f"{o['pick']} {o['name']} ({o['position']}, off-pool)" for o in off]
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
    rep: dict[str, float],
    stream: dict[str, float],
    counts: dict[str, int],
    draft: Draft,
    history: dict,
    rows: list[dict],
    problems: list[str],
    sims: int,
    noise: float,
    seed: int,
    market_weight: float,
    rollout: dict | None = None,
) -> dict:
    pos = by_position(players)
    return {
        "generated_from": pool_meta["source_file"],
        "scoring_scheme": SCHEME,
        "value_input": f"pool.json {POINTS_FIELD} ({SCHEME})",
        "value_note": (
            "3-year projected points in this league's scheme. Draftsharks' 3D value is "
            "deliberately unused: it is a provider-scaled ordinal, not points, so it "
            "cannot be differenced against a replacement level."
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
                "and who will keep, which vor does not. My first pending pick is additionally "
                "scored over the whole remaining draft (rollout_ev/rollout_edge/rollout_se): "
                "each candidate is forced there and the rest of the draft is played out, my "
                "future picks by the two-pick policy, the other teams noisy. Its take stands "
                "when a candidate's full-horizon edge over the two-pick choice clears twice "
                "its standard error, and when it overrules, the deterministic draft is "
                "re-played with that pick forced — so sim_pick, example_draft and the later "
                "picks here all describe the recommended path."
            ),
            "rollout_sims": rollout["sims"] if rollout else None,
            "picks": my_next_picks(draft, board, rollout),
        },
        "rankings_note": (
            "Undrafted players only, ranked over each other. `vor` is a points quantity "
            "and does not depend on who is gone; the rank columns are renumbered over "
            "the rows emitted here."
            if board.live
            else "The whole pool, from an empty board: no live draft was read."
        ),
        "market_model": {
            "market_weight": market_weight,
            "who": "the other nine teams; my slot always drafts on this board",
            "how": (
                "Each opposing pick scores candidates as (1 - w) * its own optimal value "
                "+ w * the market's implied value, where the market's implied value comes "
                "from pouring the ADP *ordering* into the shape of the VOR distribution "
                "(rank-based, so the 12-team pick numbering cancels). Blended at the "
                "decision rather than inside the valuation, so opponents still fill "
                "roster needs — they just rank players closer to consensus."
            ),
            "why": (
                "ADP never touches VOR, and it is not treated as the right answer. It is "
                "the best available *estimate of how this draft will actually behave*, "
                "which is what decides who is still on the board at my pick. It is a "
                "fuzzy estimate, not a biased one: a true 10-team TE-premium superflex ADP "
                "does not exist, so the closest thing on hand is a 12-team ADP with no TE "
                "premium. Set --market-weight 0 to recover the all-drafters-optimal "
                "assumption."
            ),
            "what_this_is_not": (
                "This is NOT an assumption that the other nine managers will misprice "
                "tight ends. They know their own scoring settings and will price the "
                "premium themselves; we simply cannot observe how. So the TE gap between "
                "this board and `market_rank` is measurement error in our signal, not a "
                "detected edge, and it should not be read as 'elite TEs will fall to me'. "
                "Where the signal is weakest is where --noise, not confidence, belongs."
            ),
            "adp_note": (
                "`provider_adp` is passthrough reference only — an overall pick number "
                "in the source's 12-team draft. Its values run continuously into the "
                "thousands rather than stopping at a clean unranked sentinel, which is "
                "why only its rank is ever used. "
                "`market_rank` is that rank; `adp_rank_delta` is market_rank - vor_rank, "
                "a divergence between this board and a fuzzy proxy — positive means the "
                "proxy ranks him later than this board does, which is a prompt to check "
                "why, not a free win."
            ),
        },
        "replacement": {
            "definition": (
                "Points of the (S+1)-th best player at a position, where S is how many "
                "players at that position hold a starting slot across all 10 teams in "
                "the simulated draft. This is the marginal starter, i.e. the worst "
                "player still good enough to start somewhere in this league."
            ),
            "levels": {k: round(v, 1) for k, v in rep.items()},
            "starters_by_position": counts,
            "marginal_starter": {
                k: {
                    "rank": f"{k}{counts[k] + 1}",
                    "player": pos[k][min(counts[k], len(pos[k]) - 1)].name,
                }
                for k in POSITIONS
            },
            "wire_levels": {k: round(v, 1) for k, v in stream.items()},
            "wire_note": (
                "The best player at each position left undrafted — the post-draft free "
                "agent baseline. It is what `vor_vs_wire` measures against, and inside the "
                "simulation it prices bench depth. Far below the marginal-starter level "
                "because 290 of the pool's 350 players get rostered."
            ),
            "slot_levels": {k: round(v, 1) for k, v in slot_replacement(rep).items()},
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
                "starting-lineup points above replacement + decayed bench value "
                "(growth + insurance)"
            ),
            "depth_base": DEPTH_BASE,
            "position_depth_decay": POSITION_DEPTH_DECAY,
            "insurance_base": INSURANCE_BASE,
            "depth_note": (
                "Bench value decays with how many players that team already has at the "
                "same position, not with a running bench index — a fifth QB cannot start "
                "in a two-QB-slot league however large his VOR looks. Each body is priced "
                "on two jobs over the wire: growth — `upside_points`, his 3-year total at "
                "his years-2-3 pace, weighted depth_base — and insurance — his full 3-year "
                "sum including year 1, weighted by insurance_base, his position's expected "
                "share of starter games missed to byes and injury. So a backloaded rookie "
                "and a startable veteran are both worth bench picks for different reasons. "
                "Starters are still priced on points_3yr."
            ),
            "lookahead": "value now + E[best value still available at my next pick]",
            "survival_sigma": SURVIVAL_SIGMA,
            "note": (
                "Two-pick rollout with an independence approximation across candidates; "
                "a strong greedy, not equilibrium play. My next pick's candidates get a "
                "full-horizon check on top — see my_next_picks."
            ),
        },
        "monte_carlo": {
            "sims": sims,
            "noise": noise,
            "seed": seed,
            "note": (
                "Gumbel noise on the other 9 teams only, to turn 0/1 availability under "
                "deterministic play into a usable probability band. sim_pick is from the "
                "noiseless draft; sim_adp and p_available_at_my_picks are from these."
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
    rep: dict[str, float],
    counts: dict[str, int],
    draft: Draft,
    board: Board,
    rollout: dict | None = None,
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
    print(
        "\nreplacement: " + ", ".join(f"{k}{counts[k] + 1} = {rep[k]:.0f}" for k in POSITIONS),
        file=sys.stderr,
    )

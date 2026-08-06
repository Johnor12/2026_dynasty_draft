"""The optimal-drafter draft simulation and the replacement fixed point.

Naive VOR picks a replacement rank per position out of thin air ("QB10, because 10 teams
start one QB"). That is wrong here in both directions: the superflex slot means well over
10 QBs start league-wide, and the two W-R-T slots mean the RB/WR/TE split is decided by
who is actually good, not by the slot names. The number of players at each position who
end up starting is an *outcome* of how the league drafts, so it has to be simulated, not
assumed. That creates a circularity, which `converge` resolves as a fixed point:

    replacement levels -> what a player is worth to a team -> how the draft goes
                       -> how many of each position start -> replacement levels

Picks are not myopic. Each team scores a candidate as

    value now  +  E[value of the best player still there at my next pick]

using survival probabilities from the number of intervening picks versus the candidate's
rank in the order players are actually coming off the board. This is what makes the
drafters "optimal" in the sense that matters for a board: they take the player who will
not be there later, not merely the highest number on the screen. It is a two-pick rollout
with an independence approximation across candidates, not equilibrium play — the honest
name is a strong greedy, and the two-pick horizon is the part most likely to understate
how early a truly scarce position gets attacked. That weakness is patched where it
matters most — the one decision actually in front of me: `rollout` re-scores my next
pick's candidates over the whole remaining draft, with this greedy as the base policy.

The other nine teams are pulled toward the source ADP (`market_weight`; 0 recovers
all-drafters-fully-optimal). ADP never enters VOR. It is used only as the best available
*estimate of how this draft will actually run*, since that is what decides who is on the
board at my pick — see `market_value` for why it is rank-mapped and what it is not.
"""

from __future__ import annotations

import math
import multiprocessing
import os
import random
import sys
from collections.abc import Sequence

from .board import Board
from .league import (
    DEDICATED_SLOTS,
    DEPTH_BASE,
    INSURANCE_BASE,
    LOOKAHEAD_PER_POS,
    MAX_ITERS,
    NON_TAXI_SLOTS,
    POSITIONS,
    SLOT_ELIGIBLE,
    STARTING_SLOTS,
    SURVIVAL_SIGMA,
    TEAMS,
)
from .pool import Player, by_position
from .value import (
    HORIZONS,
    compute_vor,
    apportion,
    horizon_points,
    pos_by_horizon,
    insert_sorted,
    replacement_from_draft,
    seed_replacement,
    slot_replacement,
    sorted_by_horizon,
    team_value,
    wire_replacement,
)

# --- availability model -----------------------------------------------------------


class Fenwick:
    """Availability counter over the VOR-sorted pool, for O(log n) 'how many better'."""

    __slots__ = ("n", "tree")

    def __init__(self, n: int) -> None:
        self.n = n
        self.tree = [0] * (n + 1)

    def add(self, i: int, delta: int) -> None:
        i += 1
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i

    def prefix(self, i: int) -> int:
        """Count of set entries strictly before index i."""
        total = 0
        while i > 0:
            total += self.tree[i]
            i -= i & -i
        return total


def survival(better_available: int, gap: int) -> float:
    """P(a player is still there `gap` picks later), given how many rank above him.

    If drafters followed the VOR order exactly he lasts iff more than `gap` players rank
    above him. Softened into a logistic because they follow their own roster needs, not a
    single global list.
    """
    if gap <= 0:
        return 1.0
    return 1.0 / (1.0 + math.exp((gap - better_available) / SURVIVAL_SIGMA))


# --- draft simulation -------------------------------------------------------------


class Draft:
    """One simulated draft from a Board's starting state.

    `noise` > 0 perturbs the other teams' scores only. Everything the board supplies is
    copied, never mutated, so the same Board seeds every iteration of the fixed point and
    all `--sims` noisy redraws.
    """

    def __init__(
        self,
        players: list[Player],
        rep: dict[str, dict[str, float]],
        vor: dict[int, float],
        board: Board,
        wire: dict[str, dict[str, float]] | None = None,
        noise: float = 0.0,
        rng: random.Random | None = None,
        market_vor: dict[int, float] | None = None,
        market_weight: float = 0.0,
        forced: dict[int, Player] | None = None,
        noise_from: int = 0,
        my_ban: int | None = None,
    ) -> None:
        self.players = players
        self.rep = rep
        self.slot_rep = slot_replacement(rep)
        self.vor = vor
        # A bench body's jobs, per horizon, each measured against that horizon's wire and
        # floored at zero (see value.team_value for why the floor is load-bearing). Year 1
        # is insurance only — a player who cannot play this season cannot cover starter
        # games missed this season. Years 2-3 add growth (DEPTH_BASE, the chance he grows
        # past a starter) to the same insurance weight, both on the years-2-3 excess.
        # Before any draft exists to read a wire off, fall back to the VOR baseline.
        self.wire = rep if wire is None else wire
        self.depth_value = {
            "yr1": {
                p.player_id: INSURANCE_BASE[p.position]
                * max(horizon_points(p, "yr1") - self.wire["yr1"][p.position], 0.0)
                for p in players
            },
            "yr23": {
                p.player_id: (DEPTH_BASE + INSURANCE_BASE[p.position])
                * max(horizon_points(p, "yr23") - self.wire["yr23"][p.position], 0.0)
                for p in players
            },
        }
        self.board = board
        self.order = board.order
        self.pick_nos = board.pick_nos
        self.my_slot = board.my_slot
        self.noise = noise
        self.rng = rng
        self.market_vor = market_vor or {}
        self.market_weight = market_weight if market_vor else 0.0
        # The nine other teams are market-influenced, so the order players actually come
        # off the board is the blended one — and that is what the survival model has to
        # follow, for me as much as for them. Predicting opponents correctly is the whole
        # point: it is what lets my picks exploit a TE the market is sleeping on.
        w = self.market_weight
        # The market mapping is built over the players still available, so an already
        # drafted player has no entry and falls back to his own VOR. He is off the board
        # and can never be a candidate; he only needs a place in the sort below.
        perceived = {
            p.player_id: (1 - w) * vor[p.player_id]
            + w * self.market_vor.get(p.player_id, vor[p.player_id])
            if w
            else vor[p.player_id]
            for p in players
        }
        self.perceived = perceived
        # pool sorted by perceived value desc drives the survival model; per-position,
        # per-horizon lists sorted by that horizon's points drive candidate generation
        # (team value is monotone in each horizon component, so within a position only a
        # player on the yr1/yr23 Pareto frontier can be the best pick — the heads of the
        # two orderings are its extremes; interior frontier players are approximated away).
        self.vor_sorted = sorted(players, key=lambda p: (-perceived[p.player_id], p.player_id))
        for i, p in enumerate(self.vor_sorted):
            p.vor_index = i
        # Already-drafted players stay in the sorted structures (their rank is still a fact
        # about the pool) but carry no availability bit, so `better_available` counts only
        # players who can actually be taken ahead of a candidate.
        self.taken: set[int] = set(board.taken)
        self.avail_bits = Fenwick(len(players))
        for p in players:
            if p.player_id not in self.taken:
                self.avail_bits.add(p.vor_index, 1)
        self.pos_lists = pos_by_horizon(players)
        self.heads = {h: {pos: 0 for pos in POSITIONS} for h in HORIZONS}
        self.rosters: list[list[Player]] = [list(r) for r in board.rosters]
        self.off_pool = board.off_pool  # read-only here; nothing is ever added
        self.pick_of: dict[int, int] = {}
        self.streams = self.stream_levels()  # refreshed at every pick in `choose`
        # My picks' scored candidates as (value_now, next_pick_ev, player), best first.
        # Recorded on deterministic runs only — this is what "who should I draft next"
        # reads off the final draft.
        self.my_decisions: dict[int, list[tuple[float, float, Player]]] = {}
        self.next_pick = self._next_pick_table()
        self.picks_left = list(board.picks_left)
        # Rollout hooks (see `rollout`): picks dictated by the caller instead of chosen,
        # and the first pick index where the other teams' noise applies — everything
        # before it plays deterministically, so every playout branches from one state.
        self.forced = forced or {}
        self.noise_from = noise_from
        # My slot never drafts this player (candidate_survival's counterfactual redraws).
        self.my_ban = my_ban

    def _next_pick_table(self) -> list[int | None]:
        """For each pick index, the slot's following pick index (None if it is their last)."""
        nxt: list[int | None] = [None] * len(self.order)
        last_seen: dict[int, int] = {}
        for i in range(len(self.order) - 1, -1, -1):
            slot = self.order[i]
            nxt[i] = last_seen.get(slot)
            last_seen[slot] = i
        return nxt

    # -- availability helpers

    def _advance(self, h: str, pos: str) -> None:
        lst = self.pos_lists[h][pos]
        i = self.heads[h][pos]
        while i < len(lst) and lst[i].player_id in self.taken:
            i += 1
        self.heads[h][pos] = i

    def candidates(
        self,
        roster: list[Player],
        per_pos: int = 1,
        picks_left: int | None = None,
        off: Sequence[dict] = (),
    ) -> list[Player]:
        """Best available at each position *by each horizon*, honouring taxi and
        roster-legality limits.

        With per-horizon pricing there is no single within-position ordering: the best
        year-1 body (a flat veteran) and the best years-2-3 body (a backloaded rookie, an
        injury-recovery veteran) can be different players, and either can be the right
        pick depending on what the roster is missing. So each position offers the head of
        both horizon lists, deduplicated — without this, a startable veteran sitting below
        a backloaded player on one list could never be drafted while that player remained.

        A lineup needs 1 QB, 2 RB, 3 WR and 1 TE from positions that nothing else can
        cover, so a manager cannot spend every pick on the best name available and end up
        without a quarterback. Once the picks remaining are exactly the unfilled mandatory
        spots, candidates narrow to the positions still owed. This is the honest way to stop
        a team punting a position — the earlier attempt distorted the value of an empty slot
        instead, which broke the board.

        The 4 taxi spots are rookie-only, so at most 25 of a team's 29 players can be
        veterans: once a team holds 25 non-rookies, only rookies are eligible. The cap is
        on veterans, not on total bodies — a team that took its rookies early is free to
        spend its last picks on veterans.

        `off` is the team's already-drafted players the pool cannot value. They count here
        and only here: they occupy a roster spot and they answer a mandatory position, so a
        team that spent a live pick on an unranked quarterback is not made to draft another.
        draft.json does not say whether such a player is a rookie, so they count as
        veterans — conservative, it can only force a rookie one pick early.
        """
        eligible = POSITIONS
        if picks_left is not None:
            have = {pos: 0 for pos in POSITIONS}
            for p in roster:
                have[p.position] += 1
            for o in off:
                if o.get("position") in have:
                    have[o["position"]] += 1
            owed = {pos: max(0, DEDICATED_SLOTS[pos] - have[pos]) for pos in POSITIONS}
            if picks_left <= sum(owed.values()):
                eligible = tuple(pos for pos in POSITIONS if owed[pos]) or POSITIONS
        vets_capped = sum(1 for p in roster if not p.is_rookie) + len(off) >= NON_TAXI_SLOTS

        # Both restrictions can be simultaneously unsatisfiable — a team at the veteran cap
        # still owed a QB when no rookie QB is left, which used to assert out as "pool
        # exhausted". Every one of the 29 picks is mandatory, so relax in order of what a
        # manager would actually give up: the taxi plan first (they can carry the veteran
        # on the bench and cut elsewhere), then the positional requirement.
        for positions, rookies_only in ((eligible, vets_capped), (eligible, False), (POSITIONS, False)):
            out: list[Player] = []
            seen_ids: set[int] = set()
            for pos in positions:
                for h in HORIZONS:
                    self._advance(h, pos)
                    found = 0
                    for p in self.pos_lists[h][pos][self.heads[h][pos] :]:
                        if p.player_id in self.taken:
                            continue
                        if rookies_only and not p.is_rookie:
                            continue
                        if p.player_id not in seen_ids:
                            seen_ids.add(p.player_id)
                            out.append(p)
                        found += 1
                        if found == per_pos:
                            break
            if out:
                return out
        return []

    def better_available(self, p: Player) -> int:
        return self.avail_bits.prefix(p.vor_index)

    # -- valuation

    def stream_levels(self) -> dict[str, dict[str, float]]:
        """What each starting slot yields unfilled, per horizon, on the current board.

        The best available body a slot accepts, capped at the slot's replacement level:
        skipping a slot early costs nothing (a starter-grade body will still be there —
        the documented empty-slot counterfactual), but once the last bodies above a
        horizon's level are drafted the stream falls with the board, so an unfilled
        year-1 slot is charged what filling it late would actually yield. At the end of
        the draft this is the wire. Recomputed at every pick (`choose`) and after a
        finished draft (`rollout`), because it is a fact about availability, not levels.
        """
        out: dict[str, dict[str, float]] = {}
        for h in HORIZONS:
            best: dict[str, float] = {}
            for pos in POSITIONS:
                self._advance(h, pos)
                lst, i = self.pos_lists[h][pos], self.heads[h][pos]
                best[pos] = horizon_points(lst[i], h) if i < len(lst) else 0.0
            out[h] = {
                slot: min(lv, max(best[pos] for pos in SLOT_ELIGIBLE[slot]))
                for slot, lv in self.slot_rep[h].items()
            }
        return out

    def expected_streams(self, gap: int) -> dict[str, dict[str, float]]:
        """Stream levels as they are expected to stand `gap` picks from now.

        `stream_levels` reads the board as it is; a lookahead that reuses it assumes the
        board holds still, which hides exactly the moment that matters — the last
        startable body at a position leaving between this team's picks. So the lookahead
        prices next-pick rosters against the expected best-available at each position
        after `gap` picks, from the same survival model the candidates use: the best
        available survives, or the next one is the best, and so on down the list. Players
        the current pick might itself remove are still counted as available — in the
        branch where one is drafted here he fills the very slot being streamed, so the
        optimism is confined to flex slots, whose stream another position usually sets
        anyway.
        """
        if gap <= 0:
            return self.streams
        out: dict[str, dict[str, float]] = {}
        for h in HORIZONS:
            best: dict[str, float] = {}
            for pos in POSITIONS:
                self._advance(h, pos)
                expected = 0.0
                mass = 1.0
                for p in self.pos_lists[h][pos][self.heads[h][pos] :]:
                    if p.player_id in self.taken:
                        continue
                    surv = survival(self.better_available(p), gap)
                    expected += mass * surv * horizon_points(p, h)
                    mass *= 1.0 - surv
                    if mass < 1e-4:
                        break
                best[pos] = expected
            out[h] = {
                slot: min(lv, max(best[pos] for pos in SLOT_ELIGIBLE[slot]))
                for slot, lv in self.slot_rep[h].items()
            }
        return out

    def lookahead(
        self,
        roster: list[Player],
        roster_sorted: dict[str, list[Player]],
        taking: Player,
        streams_next: dict[str, dict[str, float]],
        gap: int,
        left: int,
        off: Sequence[dict] = (),
    ) -> float:
        """E[value of the best player still available at this team's next pick].

        Everything here is valued at `streams_next` — the stream levels expected to hold
        `gap` picks from now (`expected_streams`) — so a roster that leaves a slot
        unfilled while the last startable bodies drain away is charged for it at the
        moment the decision is being made, not one pick too late. Order statistic over
        the plausible next-pick candidates: the best surviving one is
        candidate i if i survives and everyone better does not. Candidates are treated as
        independent, which slightly understates the chance that a whole position gets
        cleared out between picks.
        """
        future_roster = roster + [taking]
        future_sorted = {
            h: insert_sorted(roster_sorted[h], taking, h) for h in HORIZONS
        }
        base = team_value(future_sorted, self.slot_rep, self.depth_value, streams_next)
        scored: list[tuple[float, float]] = []
        for cand in self.candidates(
            future_roster, per_pos=LOOKAHEAD_PER_POS, picks_left=left, off=off
        ):
            if cand.player_id == taking.player_id:
                continue
            gain = (
                team_value(future_sorted, self.slot_rep, self.depth_value, streams_next, cand)
                - base
            )
            scored.append((gain, survival(self.better_available(cand), gap)))
        scored.sort(key=lambda t: -t[0])
        expected = 0.0
        mass = 1.0
        for gain, surv in scored:
            expected += mass * surv * gain
            mass *= 1.0 - surv
            if mass < 1e-4:
                break
        return expected

    def choose(self, pick_index: int, slot: int) -> Player:
        roster = self.rosters[slot - 1]
        off = self.off_pool[slot - 1]
        # Availability moved since the last pick, so the price of an unfilled slot did too.
        self.streams = self.stream_levels()
        roster_sorted = sorted_by_horizon(roster)
        base = team_value(roster_sorted, self.slot_rep, self.depth_value, self.streams)
        nxt = self.next_pick[pick_index]
        gap = None if nxt is None else nxt - pick_index - 1
        streams_next = None if gap is None else self.expected_streams(gap)
        # I draft on my own board; the other nine are pulled toward the market's ordering.
        # Blended at the decision, not inside the valuation, so their roster logic stays
        # intact — they still fill needs, they just rank players closer to consensus. At
        # market_weight 1.0 they are pure best-available-by-ADP drafters.
        w = self.market_weight if slot != self.my_slot else 0.0
        left = self.picks_left[slot - 1]
        cands = self.candidates(roster, per_pos=1, picks_left=left, off=off)
        if self.my_ban is not None and slot == self.my_slot:
            # The ban yields to roster legality: if he is my only legal candidate, take him.
            cands = [c for c in cands if c.player_id != self.my_ban] or cands
        scored: list[tuple[float, Player]] = []
        detail: list[tuple[float, float, Player]] = []
        for cand in cands:
            now = (
                team_value(roster_sorted, self.slot_rep, self.depth_value, self.streams, cand)
                - base
            )
            later = (
                0.0
                if gap is None
                else self.lookahead(roster, roster_sorted, cand, streams_next, gap, left - 1, off)
            )
            score = now + later
            if w:
                score = (1 - w) * score + w * self.market_vor[cand.player_id]
            scored.append((score, cand))
            detail.append((now, later, cand))
        assert scored, "pool exhausted"

        if slot == self.my_slot and self.rng is None:
            self.my_decisions[self.pick_nos[pick_index]] = sorted(
                detail, key=lambda t: (-(t[0] + t[1]), t[2].player_id)
            )

        if self.noise and self.rng is not None and slot != self.my_slot and pick_index >= self.noise_from:
            # Gumbel noise -> the other nine teams follow a softmax over their own scores
            # instead of a strict argmax, which is what turns 0/1 availability under
            # deterministic play into a usable probability band. Scaled to the spread
            # between this pick's candidates, so it expresses "the ordering among
            # near-equals is uncertain" rather than a fixed number of points; a pick with
            # one clear best option stays nearly deterministic.
            spread = max(s for s, _ in scored) - min(s for s, _ in scored)
            scale = self.noise * max(spread, 1.0)
            scored = [
                (s - scale * math.log(-math.log(self.rng.random())), c) for s, c in scored
            ]

        return max(scored, key=lambda t: (t[0], -t[1].player_id))[1]

    def run(self) -> None:
        """Play out the pending picks. `pick_of` is in real overall pick numbers."""
        for i, slot in enumerate(self.order):
            pick = self.forced.get(i)
            if pick is None:
                pick = self.choose(i, slot)
            else:
                assert pick.player_id not in self.taken, f"forced pick {pick.name} already taken"
            self.taken.add(pick.player_id)
            self.avail_bits.add(pick.vor_index, -1)
            self.rosters[slot - 1].append(pick)
            self.picks_left[slot - 1] -= 1
            self.pick_of[pick.player_id] = self.pick_nos[i]


def market_value(players: list[Player], vor: dict[int, float]) -> dict[int, float]:
    """Quantile-map the market's ADP ordering onto the VOR scale, over what is available.

    Called with the undrafted players only, so both the ordering and the scale it is poured
    into describe the board the opposing teams are actually looking at. On an untouched
    board that is the whole pool.

    The source ADP cannot be used as a value directly — it is a pick number, and per README
    it is a *12-team, no-TE-premium* superflex ADP, so neither its scale nor its scoring
    matches this league. What it does carry is a credible *ordering* of how the field ranks
    players. So take that ordering and pour it into the shape of the VOR distribution: the
    player the market likes best is assigned the top VOR value, the second-best the second,
    and so on. Rank-based, so the 12-team pick numbering cancels out and the raw ADP values
    (which run continuously into the thousands rather than stopping at a clean unranked
    sentinel) never need interpreting as real picks.

    This is the best available estimate of how the draft will actually run, not a claim that
    the field is wrong. The other managers know their own scoring; a true 10-team TE-premium
    superflex ADP simply does not exist to be had. So where this ordering diverges from the
    board — TEs above all — treat it as a gap in the signal rather than an opponent error,
    and let --noise carry that uncertainty. What it must never do is feed back into VOR
    itself; it only shapes how the opponents behave.
    """
    by_adp = sorted(
        players,
        key=lambda p: (p.provider_adp if p.provider_adp is not None else math.inf, p.player_id),
    )
    scale = sorted((vor[p.player_id] for p in players), reverse=True)
    return {p.player_id: scale[i] for i, p in enumerate(by_adp)}


def converge(
    players: list[Player],
    board: Board,
    report: bool,
    market_weight: float = 0.0,
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
    available = board.available(players)
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
        mkt = market_value(available, vor) if market_weight else None
        draft = Draft(
            players, rep, vor, board, wire=stream,
            market_vor=mkt, market_weight=market_weight,
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
    mkt = market_value(available, vor) if market_weight else None
    draft = Draft(
        players, rep, vor, board, wire=stream,
        market_vor=mkt, market_weight=market_weight,
    )
    draft.run()
    _, final_counts = replacement_from_draft(draft.rosters, pos_h)
    history = {
        "market_weight": market_weight,
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


# The noisy redraws and the rollout playouts are hundreds of independent, seeded draft
# simulations, so they fan out over a process pool (stdlib multiprocessing). Workers get
# the shared inputs once via the initializer; each task is identified by its seed index,
# so results are deterministic regardless of scheduling. The playout worker looks its
# forced candidate up by id in its *own* copy of the pool — Draft stamps `vor_index`
# onto the pool's Player objects, so a separately-pickled Player would carry a stale one.
_WORKER: dict = {}


def _init_worker(players, rep, board, stream, noise, seed, mkt, market_weight, vor, i_my) -> None:
    _WORKER.update(
        players=players, rep=rep, board=board, stream=stream, noise=noise, seed=seed,
        mkt=mkt, market_weight=market_weight, vor=vor, i_my=i_my,
        by_id={p.player_id: p for p in players},
    )


def _worker_pool_size() -> int:
    return max(1, len(os.sched_getaffinity(0)))


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
        market_vor=w["mkt"], market_weight=w["market_weight"],
    )
    d.run()
    slot_of = dict(zip(d.pick_nos, d.order))
    return {pid: (pk, slot_of[pk] == d.my_slot) for pid, pk in d.pick_of.items()}


def _rollout_playout(task: tuple[int, int]) -> float:
    cand_id, s = task
    w = _WORKER
    d = Draft(
        w["players"], w["rep"], w["vor"], w["board"], wire=w["stream"], noise=w["noise"],
        rng=random.Random(f"rollout-{w['seed']}-{s}"),
        market_vor=w["mkt"], market_weight=w["market_weight"],
        forced={w["i_my"]: w["by_id"][cand_id]}, noise_from=w["i_my"] + 1,
    )
    d.run()
    # End-of-draft streams are the wire: a final roster's unfilled year-1 slots are
    # charged what signing off the leftover pool would actually yield.
    return team_value(
        sorted_by_horizon(d.rosters[w["board"].my_slot - 1]),
        d.slot_rep,
        d.depth_value,
        d.stream_levels(),
    )


def monte_carlo(
    players: list[Player],
    rep: dict[str, dict[str, float]],
    board: Board,
    stream: dict[str, dict[str, float]],
    sims: int,
    noise: float,
    seed: int,
    market_weight: float = 0.0,
) -> tuple[dict[int, list[tuple[int, bool]]], dict[int, int]]:
    """Noisy redraws -> per-player (pick, taken-by-me) observations (see `_mc_draft`).

    Returns the observations and, per player, the count of redraws in which an
    *opponent* took him — my own takes are censoring, not demand.
    `candidate_survival` is the assumption-free counterfactual, priced only for the
    players where the decision actually needs it.
    """
    vor = compute_vor(players, rep)
    mkt = market_value(board.available(players), vor) if market_weight else None
    picks: dict[int, list[tuple[int, bool]]] = {p.player_id: [] for p in players}
    drafted: dict[int, int] = {p.player_id: 0 for p in players}
    with multiprocessing.Pool(
        _worker_pool_size(),
        initializer=_init_worker,
        initargs=(players, rep, board, stream, noise, seed, mkt, market_weight, vor, None),
    ) as pool:
        for pick_of in pool.map(_mc_draft, range(sims)):
            for pid, (pick, mine) in pick_of.items():
                picks[pid].append((pick, mine))
                if not mine:
                    drafted[pid] += 1
    return picks, drafted


def _survival_draft(task: tuple[int, int]) -> int | None:
    cand_id, s = task
    w = _WORKER
    d = Draft(
        w["players"], w["rep"], w["vor"], w["board"], wire=w["stream"], noise=w["noise"],
        rng=random.Random(w["seed"] + s),
        market_vor=w["mkt"], market_weight=w["market_weight"], my_ban=cand_id,
    )
    d.run()
    return d.pick_of.get(cand_id)


def candidate_survival(
    players: list[Player],
    rep: dict[str, dict[str, float]],
    board: Board,
    stream: dict[str, dict[str, float]],
    candidates: list[Player],
    sims: int,
    noise: float,
    seed: int,
    market_weight: float,
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
    mkt = market_value(board.available(players), vor) if market_weight else None
    tasks = [(cand.player_id, s) for cand in candidates for s in range(sims)]
    with multiprocessing.Pool(
        _worker_pool_size(),
        initializer=_init_worker,
        initargs=(players, rep, board, stream, noise, seed, mkt, market_weight, vor, None),
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


def rollout(
    players: list[Player],
    rep: dict[str, dict[str, float]],
    board: Board,
    stream: dict[str, dict[str, float]],
    candidates: list[Player],
    sims: int,
    noise: float,
    seed: int,
    market_weight: float,
) -> dict | None:
    """Full-horizon EV for each candidate at my next pick, by playing the draft out.

    The two-pick score in `choose` is the base policy; this evaluates the one decision in
    front of me over the whole remaining draft instead: force the candidate at my next
    pick, play everything after it `sims` times — the other teams noisy as in
    `monte_carlo`, my own future picks by the base policy — and average my final roster's
    value. A rollout of a policy is at least as good as the policy in expectation, and the
    full horizon is exactly where the two-pick score is weakest: it cannot see a
    positional run that empties a position between my later picks.

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
    mkt = market_value(board.available(players), vor) if market_weight else None
    tasks = [(cand.player_id, s) for cand in candidates for s in range(sims)]
    with multiprocessing.Pool(
        _worker_pool_size(),
        initializer=_init_worker,
        initargs=(players, rep, board, stream, noise, seed, mkt, market_weight, vor, i_my),
    ) as pool:
        flat = pool.map(_rollout_playout, tasks)
    values = {
        cand.player_id: flat[i * sims : (i + 1) * sims] for i, cand in enumerate(candidates)
    }

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
    return {"pick_no": pick_no, "sims": sims, "take_id": take_id, "stats": stats}


def apply_rollout(
    draft: Draft,
    rolled: dict | None,
    players: list[Player],
    rep: dict[str, dict[str, float]],
    board: Board,
    stream: dict[str, dict[str, float]],
    market_weight: float,
) -> Draft:
    """Re-play the deterministic draft with the rollout's pick forced, when it overrules.

    Without this, sim_pick and example_draft would show the two-pick policy's choice at my
    next pick while my_next_picks recommends someone else. One extra deterministic draft
    makes every reported block describe the path I am actually being told to play. The
    candidates' two-pick detail is transplanted unchanged: the deterministic prefix up to
    my pick is identical in both drafts, so the scores are too — only the selection
    differs, and everything after it re-plays around that choice.
    """
    if rolled is None:
        return draft
    detail = draft.my_decisions[rolled["pick_no"]]
    if rolled["take_id"] == detail[0][2].player_id:
        return draft
    take = next(c for _, _, c in detail if c.player_id == rolled["take_id"])
    vor = compute_vor(players, rep)
    mkt = market_value(board.available(players), vor) if market_weight else None
    forced = Draft(
        players, rep, vor, board, wire=stream,
        market_vor=mkt, market_weight=market_weight,
        forced={board.pick_nos.index(rolled["pick_no"]): take},
    )
    forced.run()
    forced.my_decisions[rolled["pick_no"]] = detail
    return forced

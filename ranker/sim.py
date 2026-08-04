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
how early a truly scarce position gets attacked.

The other nine teams are pulled toward the source ADP (`market_weight`; 0 recovers
all-drafters-fully-optimal). ADP never enters VOR. It is used only as the best available
*estimate of how this draft will actually run*, since that is what decides who is on the
board at my pick — see `market_value` for why it is rank-mapped and what it is not.
"""

from __future__ import annotations

import math
import random
import sys
from collections.abc import Sequence

from .board import Board
from .league import (
    DEDICATED_SLOTS,
    LOOKAHEAD_PER_POS,
    MAX_ITERS,
    NON_TAXI_SLOTS,
    POSITIONS,
    STARTING_SLOTS,
    SURVIVAL_SIGMA,
    TEAMS,
)
from .pool import Player, by_position
from .value import (
    compute_vor,
    apportion,
    replacement_from_draft,
    seed_replacement,
    slot_replacement,
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
        rep: dict[str, float],
        vor: dict[int, float],
        board: Board,
        wire: dict[str, float] | None = None,
        noise: float = 0.0,
        rng: random.Random | None = None,
        market_vor: dict[int, float] | None = None,
        market_weight: float = 0.0,
    ) -> None:
        self.players = players
        self.rep = rep
        self.slot_rep = slot_replacement(rep)
        self.vor = vor
        # Bench depth is measured against the wire, floored at zero. Before any draft exists
        # to read a wire off, fall back to the VOR baseline.
        self.wire = rep if wire is None else wire
        self.depth_value = {
            p.player_id: max(p.points - self.wire[p.position], 0.0) for p in players
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
        # pool sorted by perceived value desc drives the survival model; per-position lists
        # sorted by points drive candidate generation (within a position, more points is
        # always at least as valuable, so only the head of each list can be the best pick).
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
        self.pos_lists = {
            pos: sorted(v, key=lambda p: (-p.points, p.player_id))
            for pos, v in by_position(players).items()
        }
        self.heads = {pos: 0 for pos in POSITIONS}
        self.rosters: list[list[Player]] = [list(r) for r in board.rosters]
        self.off_pool = board.off_pool  # read-only here; nothing is ever added
        self.pick_of: dict[int, int] = {}
        # My picks' scored candidates as (value_now, next_pick_ev, player), best first.
        # Recorded on deterministic runs only — this is what "who should I draft next"
        # reads off the final draft.
        self.my_decisions: dict[int, list[tuple[float, float, Player]]] = {}
        self.next_pick = self._next_pick_table()
        self.picks_left = list(board.picks_left)

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

    def _advance(self, pos: str) -> None:
        lst = self.pos_lists[pos]
        i = self.heads[pos]
        while i < len(lst) and lst[i].player_id in self.taken:
            i += 1
        self.heads[pos] = i

    def candidates(
        self,
        roster: list[Player],
        per_pos: int = 1,
        picks_left: int | None = None,
        off: Sequence[dict] = (),
    ) -> list[Player]:
        """Best available at each position, honouring taxi and roster-legality limits.

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
            for pos in positions:
                self._advance(pos)
                found = 0
                for p in self.pos_lists[pos][self.heads[pos] :]:
                    if p.player_id in self.taken:
                        continue
                    if rookies_only and not p.is_rookie:
                        continue
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

    def marginal(self, roster: list[Player], p: Player, base: float) -> float:
        return team_value(roster + [p], self.slot_rep, self.depth_value) - base

    def lookahead(
        self,
        roster: list[Player],
        taking: Player,
        gap: int,
        left: int,
        off: Sequence[dict] = (),
    ) -> float:
        """E[value of the best player still available at this team's next pick].

        Order statistic over the plausible next-pick candidates: the best surviving one is
        candidate i if i survives and everyone better does not. Candidates are treated as
        independent, which slightly understates the chance that a whole position gets
        cleared out between picks.
        """
        future_roster = roster + [taking]
        base = team_value(future_roster, self.slot_rep, self.depth_value)
        scored: list[tuple[float, float]] = []
        for cand in self.candidates(
            future_roster, per_pos=LOOKAHEAD_PER_POS, picks_left=left, off=off
        ):
            if cand.player_id == taking.player_id:
                continue
            gain = self.marginal(future_roster, cand, base)
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
        base = team_value(roster, self.slot_rep, self.depth_value)
        nxt = self.next_pick[pick_index]
        gap = None if nxt is None else nxt - pick_index - 1
        # I draft on my own board; the other nine are pulled toward the market's ordering.
        # Blended at the decision, not inside the valuation, so their roster logic stays
        # intact — they still fill needs, they just rank players closer to consensus. At
        # market_weight 1.0 they are pure best-available-by-ADP drafters.
        w = self.market_weight if slot != self.my_slot else 0.0
        left = self.picks_left[slot - 1]
        scored: list[tuple[float, Player]] = []
        detail: list[tuple[float, float, Player]] = []
        for cand in self.candidates(roster, per_pos=1, picks_left=left, off=off):
            now = self.marginal(roster, cand, base)
            later = 0.0 if gap is None else self.lookahead(roster, cand, gap, left - 1, off)
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

        if self.noise and self.rng is not None and slot != self.my_slot:
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
            pick = self.choose(i, slot)
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
) -> tuple[dict[str, float], dict[str, float], dict[str, int], Draft, dict]:
    """Fixed point: replacement -> valuation -> draft -> starter counts -> replacement.

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
    available = board.available(players)
    rep = seed_replacement(players)
    stream = dict(rep)  # no draft to read a wire off yet
    trace = [{"iteration": 0, "source": "slot_assignment", "replacement": dict(rep)}]
    seen: dict[tuple[float, ...], int] = {}
    observations: list[dict[str, float]] = []
    wire_observations: list[dict[str, float]] = []
    counts_by_iter: list[dict[str, int]] = []
    cycle_start: int | None = None

    for it in range(1, MAX_ITERS + 1):
        vor = compute_vor(players, rep)
        mkt = market_value(available, vor) if market_weight else None
        draft = Draft(
            players, rep, vor, board, wire=stream,
            market_vor=mkt, market_weight=market_weight,
        )
        draft.run()
        starter_rep, counts = replacement_from_draft(draft.rosters, pos)
        wire = wire_replacement(draft.taken, pos)
        observations.append(starter_rep)
        wire_observations.append(wire)
        counts_by_iter.append(counts)
        key = tuple(counts[k] for k in POSITIONS) + tuple(wire[k] for k in POSITIONS)
        trace.append(
            {
                "iteration": it,
                "source": "draft_simulation",
                "starters_by_position": dict(counts),
                "observed_replacement": {k: round(v, 1) for k, v in starter_rep.items()},
            }
        )
        if report:
            print(
                f"  iter {it}: starters={counts} observed rep="
                + ", ".join(f"{k} {starter_rep[k]:.0f}" for k in POSITIONS),
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
    rep = {k: sum(o[k] for o in cycle) / len(cycle) for k in POSITIONS}
    stream = {k: sum(o[k] for o in wire_cycle) / len(wire_cycle) for k in POSITIONS}
    counts = apportion(
        {k: sum(c[k] for c in cycle_counts) / len(cycle_counts) for k in POSITIONS},
        TEAMS * sum(STARTING_SLOTS.values()),
    )
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
    _, final_counts = replacement_from_draft(draft.rosters, pos)
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
            k: [round(min(o[k] for o in cycle), 1), round(max(o[k] for o in cycle), 1)]
            for k in POSITIONS
        },
        "trace": trace,
    }
    return rep, stream, counts, draft, history


def monte_carlo(
    players: list[Player],
    rep: dict[str, float],
    board: Board,
    stream: dict[str, float],
    sims: int,
    noise: float,
    seed: int,
    market_weight: float = 0.0,
) -> tuple[dict[int, list[int]], dict[int, int]]:
    """Noisy redraws -> pick distribution per player. Returns picks seen and draft counts."""
    vor = compute_vor(players, rep)
    mkt = market_value(board.available(players), vor) if market_weight else None
    picks: dict[int, list[int]] = {p.player_id: [] for p in players}
    drafted: dict[int, int] = {p.player_id: 0 for p in players}
    for s in range(sims):
        rng = random.Random(seed + s)
        d = Draft(
            players, rep, vor, board, wire=stream, noise=noise, rng=rng,
            market_vor=mkt, market_weight=market_weight,
        )
        d.run()
        for pid, pick in d.pick_of.items():
            picks[pid].append(pick)
            drafted[pid] += 1
    return picks, drafted

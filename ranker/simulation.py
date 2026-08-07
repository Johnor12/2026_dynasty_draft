"""Deterministic draft state and pick policies.

`Draft` resumes an immutable live board, applies roster legality, values my slot, and
uses each opponent's inferred provider order. Fixed-point replacement convergence is in
`convergence.py`; stochastic redraws and deeper plans are in `planning.py`.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from .board import Board
from .league import (
    DEDICATED_SLOTS,
    DEPTH_BASE,
    INSURANCE_BASE,
    LOOKAHEAD_PER_POS,
    NON_TAXI_SLOTS,
    OPPONENT_BALANCE_STRENGTH,
    POSITIONS,
    SLOT_ELIGIBLE,
    SURVIVAL_SIGMA,
)
from .opponents import OpponentStrategy
from .pool import Player
from .value import (
    HORIZONS,
    horizon_points,
    pos_by_horizon,
    insert_sorted,
    slot_replacement,
    sorted_by_horizon,
    team_value,
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
        opponents: dict[int, OpponentStrategy] | None = None,
        forced: dict[int, Player] | None = None,
        targets: dict[int, Player] | None = None,
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
        self.opponents = opponents or {}
        missing = set(self.order) - {self.my_slot} - set(self.opponents)
        if missing:
            raise ValueError(f"no source strategy for opponent slot(s) {sorted(missing)}")
        self.by_id = {p.player_id: p for p in players}
        # The fast survival approximation for my later picks needs one league-wide order.
        # Average the nine source ranks, counting repeated sources once per manager. This
        # is only an approximation of the slot-specific policies; the decision in front
        # of me is priced by branch-specific Monte Carlo redraws instead.
        consensus_score = {
            p.player_id: sum(s.ranks[p.player_id] for s in self.opponents.values())
            / len(self.opponents)
            for p in players
        }
        # pool sorted by opponent consensus drives the survival model; per-position,
        # per-horizon lists sorted by that horizon's points drive candidate generation
        # (team value is monotone in each horizon component, so within a position only a
        # player on the yr1/yr23 Pareto frontier can be the best pick — the heads of the
        # two orderings are its extremes; interior frontier players are approximated away).
        self.opponent_sorted = sorted(
            players, key=lambda p: (consensus_score[p.player_id], p.player_id)
        )
        self.opponent_consensus_rank = {
            p.player_id: i for i, p in enumerate(self.opponent_sorted, start=1)
        }
        for i, p in enumerate(self.opponent_sorted):
            p.availability_index = i
        # Already-drafted players stay in the sorted structures (their rank is still a fact
        # about the pool) but carry no availability bit, so `better_available` counts only
        # players who can actually be taken ahead of a candidate.
        self.taken: set[int] = set(board.taken)
        self.avail_bits = Fenwick(len(players))
        for p in players:
            if p.player_id not in self.taken:
                self.avail_bits.add(p.availability_index, 1)
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
        # A target yields to an opponent taking him first, then the normal policy chooses.
        # Unlike `forced`, this is a draft plan rather than an assertion about availability.
        self.targets = targets or {}
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
        scenarios = self._eligibility_scenarios(roster, picks_left, off)
        for positions, rookies_only in scenarios:
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

    def _eligibility_scenarios(
        self,
        roster: list[Player],
        picks_left: int | None,
        off: Sequence[dict],
    ) -> tuple[tuple[tuple[str, ...], bool], ...]:
        """Roster-legality filters shared by my board and the opponent boards."""
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

        # Every pick is mandatory, so relax an impossible taxi plan before an impossible
        # dedicated-position plan.
        return ((eligible, vets_capped), (eligible, False), (POSITIONS, False))

    def opponent_candidates(self, slot: int) -> list[Player]:
        """All legal players, ordered only by this opponent's inferred source board."""
        strategy = self.opponents[slot]
        roster = self.rosters[slot - 1]
        off = self.off_pool[slot - 1]
        for positions, rookies_only in self._eligibility_scenarios(
            roster, self.picks_left[slot - 1], off
        ):
            candidates = [
                self.by_id[player_id]
                for player_id in strategy.order
                if player_id not in self.taken
                and self.by_id[player_id].position in positions
                and (not rookies_only or self.by_id[player_id].is_rookie)
            ]
            if candidates:
                return candidates
        return []

    def target_is_legal(self, target: Player, slot: int) -> bool:
        """Whether a surviving plan target belongs to the first viable legal scenario."""
        if target.player_id in self.taken:
            return False
        roster = self.rosters[slot - 1]
        off = self.off_pool[slot - 1]
        for positions, rookies_only in self._eligibility_scenarios(
            roster, self.picks_left[slot - 1], off
        ):
            exists = any(
                p.player_id not in self.taken
                and p.position in positions
                and (not rookies_only or p.is_rookie)
                for p in self.players
            )
            if exists:
                return target.position in positions and (
                    not rookies_only or target.is_rookie
                )
        return False

    def opponent_balance_boosts(self, slot: int) -> dict[str, float]:
        """Source-rank boosts for this roster's unfilled dedicated starters."""
        have = {position: 0 for position in POSITIONS}
        for player in self.rosters[slot - 1]:
            have[player.position] += 1
        for row in self.off_pool[slot - 1]:
            if row.get("position") in have:
                have[row["position"]] += 1
        return {
            position: 1.0
            + OPPONENT_BALANCE_STRENGTH
            * max(required - have[position], 0)
            / required
            for position, required in DEDICATED_SLOTS.items()
        }

    def better_available(self, p: Player) -> int:
        return self.avail_bits.prefix(p.availability_index)

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
        value_cache: dict[tuple[int, ...], float],
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
        base_key = (taking.player_id,)
        base = value_cache.get(base_key)
        if base is None:
            base = team_value(future_sorted, self.slot_rep, self.depth_value, streams_next)
            value_cache[base_key] = base
        scored: list[tuple[float, float]] = []
        for cand in self.candidates(
            future_roster, per_pos=LOOKAHEAD_PER_POS, picks_left=left, off=off
        ):
            if cand.player_id == taking.player_id:
                continue
            a, b = taking.player_id, cand.player_id
            pair_key = (a, b) if a < b else (b, a)
            pair_value = value_cache.get(pair_key)
            if pair_value is None:
                pair_value = team_value(
                    future_sorted, self.slot_rep, self.depth_value, streams_next, cand
                )
                value_cache[pair_key] = pair_value
            gain = pair_value - base
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

    def score_my_candidates(
        self, pick_index: int, per_pos: int = 1
    ) -> list[tuple[float, float, Player]]:
        """Roster and next-pick value for my legal candidates at one draft state."""
        slot = self.my_slot
        roster = self.rosters[slot - 1]
        off = self.off_pool[slot - 1]
        # Availability moved since the last pick, so the price of an unfilled slot did too.
        self.streams = self.stream_levels()
        roster_sorted = sorted_by_horizon(roster)
        base = team_value(roster_sorted, self.slot_rep, self.depth_value, self.streams)
        nxt = self.next_pick[pick_index]
        gap = None if nxt is None else nxt - pick_index - 1
        streams_next = None if gap is None else self.expected_streams(gap)
        # Taking A then B values the same roster as B then A.
        lookahead_values: dict[tuple[int, ...], float] = {}
        left = self.picks_left[slot - 1]
        cands = self.candidates(roster, per_pos=per_pos, picks_left=left, off=off)
        target = self.targets.get(pick_index)
        if target is not None and self.target_is_legal(target, slot):
            # Planned later targets can be interior players that the bulk candidate shortcut
            # would omit. Score them too so the reported decision remains inspectable.
            if target.player_id not in {c.player_id for c in cands}:
                cands.append(target)
        if self.my_ban is not None and slot == self.my_slot:
            # The ban yields to roster legality: if he is my only legal candidate, take him.
            cands = [c for c in cands if c.player_id != self.my_ban] or cands
        detail: list[tuple[float, float, Player]] = []
        for cand in cands:
            now = (
                team_value(roster_sorted, self.slot_rep, self.depth_value, self.streams, cand)
                - base
            )
            later = (
                0.0
                if gap is None
                else self.lookahead(
                    roster, roster_sorted, cand, streams_next, lookahead_values,
                    gap, left - 1, off,
                )
            )
            detail.append((now, later, cand))
        assert detail, "pool exhausted"
        return sorted(detail, key=lambda t: (-(t[0] + t[1]), t[2].player_id))

    def choose(self, pick_index: int, slot: int) -> Player:
        if slot != self.my_slot:
            return self.choose_opponent(pick_index, slot)

        detail = self.score_my_candidates(pick_index)

        if slot == self.my_slot and self.rng is None:
            self.my_decisions[self.pick_nos[pick_index]] = detail

        return detail[0][2]

    def choose_opponent(self, pick_index: int, slot: int) -> Player:
        """Choose from a balance-adjusted source board, without consulting my valuation."""
        candidates = self.opponent_candidates(slot)
        assert candidates, "pool exhausted"
        boosts = self.opponent_balance_boosts(slot)
        ranked = [
            (rank / boosts[player.position], rank, player)
            for rank, player in enumerate(candidates, start=1)
        ]
        if not self.noise or self.rng is None or pick_index < self.noise_from:
            return min(ranked, key=lambda row: (row[0], row[1], row[2].player_id))[2]

        # The investigator measures log2(rank among available) for each real pick.
        # rank_power calibrates source adherence; the balance-adjusted rank then gives a
        # nearby player at an unfilled starter position better odds without guaranteeing
        # the pick. `--noise` scales random variation around that preference.
        power = self.opponents[slot].rank_power
        return max(
            (
                -power * math.log(preference_rank)
                - self.noise * math.log(-math.log(self.rng.random())),
                -source_rank,
                -player.player_id,
                player,
            )
            for preference_rank, source_rank, player in ranked
        )[3]

    def run(
        self,
        until_taken: int | None = None,
        stop_before: int | None = None,
    ) -> int | None:
        """Play pending picks, optionally stopping at a player or before a pick index."""
        for i, slot in enumerate(self.order):
            if i == stop_before:
                break
            pick = self.forced.get(i)
            if pick is None:
                target = self.targets.get(i)
                if target is not None and self.target_is_legal(target, slot):
                    # Deterministic replays still score the ordinary choice for reporting;
                    # noisy rollout workers only need to exercise the planned target.
                    if self.rng is None and slot == self.my_slot:
                        self.choose(i, slot)
                    pick = target
                else:
                    pick = self.choose(i, slot)
            else:
                assert pick.player_id not in self.taken, f"forced pick {pick.name} already taken"
            self.taken.add(pick.player_id)
            self.avail_bits.add(pick.availability_index, -1)
            self.rosters[slot - 1].append(pick)
            self.picks_left[slot - 1] -= 1
            self.pick_of[pick.player_id] = self.pick_nos[i]
            if pick.player_id == until_taken:
                return self.pick_nos[i]
        return None

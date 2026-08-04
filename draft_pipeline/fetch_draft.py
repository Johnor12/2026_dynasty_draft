#!/usr/bin/env python3
"""Query the live Sleeper draft and write the board: every pick, made and pending.

This is the whole draft pipeline — one stage, so there is no orchestrator to wrap it.
It reads nothing on disk and writes one file:

    Sleeper's draft API  ->  draft.json

    python3 draft_pipeline/fetch_draft.py

``pool_pipeline/`` is the repo's other pipeline (provider html -> ``pool.json``) and is
entirely separate: local, offline, deterministic, three ordered stages, re-run when the
projections change. This one is network-only, has no inputs, and is re-run on demand
during a live draft. They share no code and never touch each other's files; they meet
only at ``sleeper_id`` (see below), which is why keeping them apart costs nothing.

**It is on demand and deliberately uncached.** ``pool_pipeline/fetch_sleeper.py``
refuses to re-download inside 24h because a roster of NFL players does not change that
fast and the dump is 14 MB. This is the opposite case: the draft changes with every
pick, the four responses total a few hundred KB, and a stale answer is worse than no
answer, so every run re-asks. Sleeper's only stated limit is 1000 calls a minute; four
calls per run is not close to it. For the same reason the raw responses are not cached
the way ``sleeper_players.json`` is — re-fetching is cheaper than storing, so the
narrowing happens here rather than downstream.

Four endpoints, because the picks alone don't say who owns what:

    /draft/<id>                 teams, rounds, snake type, reversal round, draft order
    /draft/<id>/picks           the selections made so far
    /draft/<id>/traded_picks    who owns a pick that changed hands
    /league/<id>/users          display names, so the file is readable

WHAT THE OUTPUT IS
------------------
One ``picks`` array of all ``teams x rounds`` entries, indexed by ``pick_no``, each with
the same shape and a ``status`` of ``made`` or ``pending``. Made picks carry the player;
pending ones carry ``null`` and exist so the file answers "who picks next" and "when is
my next pick" — the whole reason to look at a live draft while ranking.

Every entry carries ``sleeper_id``, which is the join key back to ``pool.json``'s
``sleeper_id`` (put there by ``pool_pipeline/match_sleeper.py``). Nothing here is joined
to the pool: that is the consumer's business, and this file stays a record of what
Sleeper said.
Sleeper's own ``name``/``position``/``team`` come along so the file can be read by eye,
but they are informational — the pool has its own copy from the projections provider,
and ``sleeper_id`` is what to key on.

WHERE PENDING PICKS COME FROM
-----------------------------
Sleeper reports the slot, roster and user for a pick only once it has been made, so the
rest of the board is derived from the draft's geometry. A plain snake alternates — odd
rounds run slot 1..N, even rounds N..1 — and a *reversal round* repeats the previous
round's order instead of flipping back, which inverts the parity from that round on.
This league reverses at round 3, so the order runs:

    R1 forward, R2 reverse, R3 reverse, R4 forward, R5 reverse, R6 forward, ...

which puts my slot-2 picks at 1.02, 2.09, 3.09, 4.02, 5.09, 6.02 ... 28.02, 29.09 —
exactly what README.md documents. Traded picks are applied on top, from
``/traded_picks``, so a pending pick is attributed to whoever owns it now.

**The derivation is checked against reality on every run.** For each made pick the
derived slot and owning roster are compared with the ones Sleeper reported, and any
disagreement is warned about on stderr and recorded in the output's
``board_derivation`` block. That check strengthens with every pick and is what keeps a
wrong reversal rule from quietly misattributing the pending half of the board.

Early in a draft it can only exercise the rounds already played, though — on pick 4 of
290 that is one round of 29, and nothing at all of the trade logic, since a board with no
traded picks cannot test them. So ``--selftest`` covers the rest offline: the slot order
of each supported format, this league's slot-2 sequence against the one README.md states,
a traded pick landing with its acquirer, and the negative control that an *un*applied
trade is caught by the live check above.

Usage:
    python3 fetch_draft.py                      # -> draft.json
    python3 fetch_draft.py --report             # + validation summary, incl. the pool join
    python3 fetch_draft.py --selftest           # check the board geometry offline, then exit
    python3 fetch_draft.py --draft-id 123 -o other.json
    python3 fetch_draft.py --me someusername    # whose picks get is_mine
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import paths

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Our draft. The trailing number of the league's draft URL is the draft id:
#: https://sleeper.com/draft/nfl/1388293618208374784
DRAFT_ID = "1388293618208374784"

#: Whose picks get ``is_mine``. A Sleeper username (their ``display_name``), matched
#: case-insensitively; ``--me`` overrides. A username rather than a draft slot because
#: the slot is a fact about this draft that the draft itself already reports.
MY_USERNAME = "johnor"

API = "https://api.sleeper.app/v1"

TIMEOUT_SECONDS = 30

#: Formats a board can be laid out for. An auction has no pick order to derive.
SUPPORTED_TYPES = ("snake", "linear")

#: Round -> my pick's position in that round, as README.md states it (1.02, 2.9, 3.9,
#: 4.2, 5.9, 6.2 ... 28.2, 29.9). Checked by ``--report`` only, and only for the
#: configured owner: the live cross-check against made picks is the real guard, but it
#: cannot test a round nobody has drafted yet, and on pick 4 of 290 that is 28 of 29
#: rounds. Nothing reads this — it is a documented expectation, not an input.
README_MY_PICK_IN_ROUND = {1: 2, 2: 9, 3: 9, 4: 2, 5: 9, 6: 2, 28: 2, 29: 9}

FIELD_DEFINITIONS = {
    "pick_no": "Overall pick number, 1..pick_count. Unique, gap-free, and the array order.",
    "round": "Round number, 1..rounds.",
    "pick_in_round": (
        "Position within the round, 1..teams. Differs from draft_slot in a reversed "
        "round — in round 2 of a 10-team snake, pick_in_round 1 is draft_slot 10."
    ),
    "draft_slot": "The board column this pick belongs to, 1..teams.",
    "roster_id": (
        "The Sleeper roster that receives the player — the current owner, which is not "
        "the slot's original owner if the pick was traded."
    ),
    "user_id": "The Sleeper user that owns the pick. Null if the draft order is unpublished.",
    "username": "That user's Sleeper display name. Null if the league user list was unreadable.",
    "is_mine": "True for the configured owner's picks (see `me` in the header).",
    "status": "'made' — Sleeper has recorded a selection — or 'pending'.",
    "sleeper_id": (
        "Sleeper player id of the selection, as a string; null while pending. This is "
        "the join key back to pool.json's sleeper_id."
    ),
    "name": (
        "Sleeper's name for the selection, for reading the file by eye. Informational: "
        "the pool carries the projection provider's own name, and joins go by sleeper_id."
    ),
    "position": "Sleeper's position for the selection. Informational, as above.",
    "team": "Sleeper's NFL team for the selection. Informational, as above.",
    "is_keeper": "True when Sleeper flagged the pick as a keeper. False for a normal pick.",
}


# ---------------------------------------------------------------------------
# Board geometry
# ---------------------------------------------------------------------------


class Board:
    """Which slot, roster and user owns each pick number.

    Everything here comes from the draft object: ``settings`` for the shape,
    ``draft_order`` (user -> slot) and ``slot_to_roster_id`` for who sits where, and
    ``/traded_picks`` for the picks that have since changed hands.
    """

    def __init__(self, draft: dict, traded: list[dict] | None = None):
        settings = draft.get("settings") or {}
        self.type = draft.get("type") or "snake"
        self.teams = int(settings.get("teams") or 0)
        self.rounds = int(settings.get("rounds") or 0)
        self.reversal_round = int(settings.get("reversal_round") or 0)
        self.pick_count = self.teams * self.rounds

        self.slot_to_roster = {
            int(slot): int(roster)
            for slot, roster in (draft.get("slot_to_roster_id") or {}).items()
        }
        # draft_order maps user -> slot; a board is read the other way round.
        self.slot_to_user = {
            int(slot): user for user, slot in (draft.get("draft_order") or {}).items()
        }
        self.roster_to_user = {
            roster: self.slot_to_user.get(slot) for slot, roster in self.slot_to_roster.items()
        }
        self.traded = self._ownership(traded or [], str(draft.get("season") or ""))

    @staticmethod
    def _ownership(traded: list[dict], season: str) -> dict[tuple[int, int], int]:
        """(round, original roster) -> roster that owns that pick now.

        A traded-pick entry names the pick by its *original* owner's ``roster_id``,
        which is exactly how a slot maps onto the board, and ``owner_id`` is who holds
        it now. Entries for another season belong to a different draft in the league.
        """
        moved: dict[tuple[int, int], int] = {}
        for entry in traded:
            if season and str(entry.get("season") or season) != season:
                continue
            try:
                key = (int(entry["round"]), int(entry["roster_id"]))
                moved[key] = int(entry["owner_id"])
            except (KeyError, TypeError, ValueError):
                continue
        return moved

    def problems(self) -> list[str]:
        """Reasons a board cannot be laid out at all."""
        issues = []
        if self.type not in SUPPORTED_TYPES:
            issues.append(f"draft type {self.type!r} has no pick order to derive")
        if self.teams < 1 or self.rounds < 1:
            issues.append(f"settings give {self.teams} teams x {self.rounds} rounds")
        return issues

    def is_reversed(self, round_no: int) -> bool:
        """Does this round run slot teams..1 rather than 1..teams?

        A plain snake alternates, odd rounds forward. A reversal round repeats the
        previous round's order instead of flipping back, so from that round on the
        parity is inverted — reversal_round 3 gives forward, reverse, reverse,
        forward, reverse, forward, ...
        """
        if self.type == "linear":
            return False
        reversed_ = round_no % 2 == 0
        if self.reversal_round and round_no >= self.reversal_round:
            reversed_ = not reversed_
        return reversed_

    def locate(self, pick_no: int) -> tuple[int, int, int]:
        """(round, pick_in_round, draft_slot) for an overall pick number."""
        round_no = (pick_no - 1) // self.teams + 1
        pick_in_round = (pick_no - 1) % self.teams + 1
        slot = (
            self.teams + 1 - pick_in_round if self.is_reversed(round_no) else pick_in_round
        )
        return round_no, pick_in_round, slot

    def owner_roster(self, round_no: int, slot: int) -> int | None:
        """The roster holding this slot's pick in this round, trades applied."""
        original = self.slot_to_roster.get(slot)
        if original is None:
            return None
        return self.traded.get((round_no, original), original)


def round_pick(round_no: int, pick_in_round: int) -> str:
    """``1.02`` — how a draft slot is spoken about, and how README.md writes it."""
    return f"{round_no}.{pick_in_round:02d}"


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def get_json(url: str, timeout: int = TIMEOUT_SECONDS):
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def fetch(draft_id: str, api: str = API, timeout: int = TIMEOUT_SECONDS) -> dict:
    """The draft, its picks, its traded picks, and the league's users.

    The first three are load-bearing and a failure is fatal — half a board is worse
    than none, and a missing traded_picks would silently misattribute pending picks.
    The user list only supplies display names, so it degrades to a warning.
    """
    draft = get_json(f"{api}/draft/{draft_id}", timeout)
    if not isinstance(draft, dict) or not draft.get("draft_id"):
        raise ValueError(f"no draft {draft_id} at {api} — check the id in the draft URL")

    picks = get_json(f"{api}/draft/{draft_id}/picks", timeout) or []
    traded = get_json(f"{api}/draft/{draft_id}/traded_picks", timeout) or []
    if not isinstance(picks, list) or not isinstance(traded, list):
        raise ValueError("picks/traded_picks did not come back as lists")

    users, warning = [], None
    league_id = draft.get("league_id")
    if league_id:
        try:
            users = get_json(f"{api}/league/{league_id}/users", timeout) or []
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            warning = f"could not read league users ({exc}) — names will be null"
    else:
        warning = "draft has no league_id — names will be null"

    return {"draft": draft, "picks": picks, "traded": traded, "users": users, "warning": warning}


# ---------------------------------------------------------------------------
# Board assembly
# ---------------------------------------------------------------------------


def index_users(users: list[dict]) -> dict[str, dict]:
    """user id -> display name and team name. Both are free text a human typed, so
    they are stripped; several of this league's team names carry a trailing space."""
    return {
        str(user["user_id"]): {
            "username": (user.get("display_name") or "").strip() or None,
            "team_name": ((user.get("metadata") or {}).get("team_name") or "").strip() or None,
        }
        for user in users
        if isinstance(user, dict) and user.get("user_id")
    }


def resolve_me(username: str, board: Board, by_user: dict[str, dict]) -> dict:
    """Find the configured owner's user id, slot and roster, or say why not."""
    wanted = (username or "").strip().lower()
    match = next(
        (uid for uid, user in by_user.items() if (user["username"] or "").lower() == wanted),
        None,
    )
    if match is None:
        return {"username": username, "user_id": None, "draft_slot": None, "roster_id": None}
    slot = next((s for s, uid in board.slot_to_user.items() if uid == match), None)
    return {
        "username": by_user[match]["username"],
        "user_id": match,
        "draft_slot": slot,
        "roster_id": board.slot_to_roster.get(slot) if slot else None,
    }


def pick_rows(board: Board, picks: list[dict], by_user: dict[str, dict], my_user: str | None):
    """Every pick 1..pick_count, made ones from Sleeper and the rest derived.

    Returns (rows, checks) where checks records how the derivation compared with what
    Sleeper reported for the picks that have been made.
    """
    made = {}
    for pick in picks:
        try:
            made[int(pick["pick_no"])] = pick
        except (KeyError, TypeError, ValueError):
            continue

    mismatches: list[dict] = []
    rows: list[dict] = []

    for pick_no in range(1, board.pick_count + 1):
        round_no, pick_in_round, slot = board.locate(pick_no)
        derived_roster = board.owner_roster(round_no, slot)
        pick = made.get(pick_no)

        if pick is None:
            roster_id = derived_roster
            user_id = board.roster_to_user.get(roster_id) if roster_id else None
            player = {
                "status": "pending",
                "sleeper_id": None,
                "name": None,
                "position": None,
                "team": None,
                "is_keeper": None,
            }
        else:
            # Sleeper reported these, so they win; the derived pair is the thing
            # under test, and any gap is surfaced rather than silently preferred.
            reported_slot = pick.get("draft_slot")
            reported_roster = pick.get("roster_id")
            if (reported_slot is not None and int(reported_slot) != slot) or (
                reported_roster is not None and int(reported_roster) != derived_roster
            ):
                mismatches.append(
                    {
                        "pick_no": pick_no,
                        "reported": {"draft_slot": reported_slot, "roster_id": reported_roster},
                        "derived": {"draft_slot": slot, "roster_id": derived_roster},
                    }
                )
            slot = int(reported_slot) if reported_slot is not None else slot
            roster_id = int(reported_roster) if reported_roster is not None else derived_roster
            # picked_by is empty when the pick was made for the team, not by them.
            user_id = pick.get("picked_by") or board.roster_to_user.get(roster_id)
            meta = pick.get("metadata") or {}
            player = {
                "status": "made",
                "sleeper_id": str(pick["player_id"]) if pick.get("player_id") else None,
                "name": " ".join(
                    part for part in (meta.get("first_name"), meta.get("last_name")) if part
                )
                or None,
                "position": meta.get("position") or None,
                "team": meta.get("team") or None,
                "is_keeper": bool(pick.get("is_keeper")),
            }

        rows.append(
            {
                "pick_no": pick_no,
                "round": round_no,
                "pick_in_round": pick_in_round,
                "draft_slot": slot,
                "roster_id": roster_id,
                "user_id": user_id,
                "username": (by_user.get(str(user_id)) or {}).get("username"),
                "is_mine": bool(my_user) and str(user_id) == str(my_user),
                **player,
            }
        )

    checks = {
        "made_picks_checked": len(made),
        "slot_and_roster_agree": len(made) - len(mismatches),
        "mismatches": mismatches,
        "rounds_exercised": sorted({board.locate(n)[0] for n in made}),
    }
    return rows, checks


def pick_number_problems(picks: list[dict], board: Board) -> tuple[list[str], list[str]]:
    """Check the reported pick numbers. Returns (fatal, notable).

    Fatal means the array indexed by ``pick_no`` would silently lose a pick: a number
    outside the board, or two picks claiming the same one. A *gap* is not fatal —
    those pick numbers simply stay pending, and a keeper draft can legitimately have
    selections recorded before the picks in front of them — so it is only reported.
    """
    numbers = [pick.get("pick_no") for pick in picks]
    fatal, notable = [], []

    bad = [n for n in numbers if not isinstance(n, int) or not 1 <= n <= board.pick_count]
    if bad:
        fatal.append(f"{len(bad)} pick_no outside 1..{board.pick_count}: {bad[:5]}")
    duplicates = [n for n, count in collections.Counter(numbers).items() if count > 1]
    if duplicates:
        fatal.append(f"duplicate pick_no: {duplicates[:5]}")

    clean = sorted(n for n in numbers if isinstance(n, int))
    if clean and clean != list(range(1, len(clean) + 1)):
        missing = sorted(set(range(1, max(clean) + 1)) - set(clean))
        notable.append(
            f"picks are not a gap-free prefix — {len(missing)} earlier pick(s) unmade "
            f"behind pick {max(clean)}, first {missing[:5]}"
        )
    return fatal, notable


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


def iso(epoch_ms) -> str | None:
    """Sleeper's millisecond timestamps, as UTC ISO-8601."""
    if not epoch_ms:
        return None
    try:
        moment = dt.datetime.fromtimestamp(int(epoch_ms) / 1000, dt.timezone.utc)
    except (TypeError, ValueError, OSError, OverflowError):
        return None
    return moment.isoformat(timespec="seconds")


def build_document(fetched: dict, board: Board, rows: list[dict], checks: dict, me: dict) -> dict:
    draft = fetched["draft"]
    by_user = index_users(fetched["users"])
    on_clock = next((row for row in rows if row["status"] == "pending"), None)
    mine_next = next(
        (row for row in rows if row["status"] == "pending" and row["is_mine"]), None
    )
    made = [row for row in rows if row["status"] == "made"]

    def summarize(row: dict | None) -> dict | None:
        if row is None:
            return None
        summary = {
            key: row[key]
            for key in ("pick_no", "round", "pick_in_round", "draft_slot", "user_id", "username")
        }
        summary["slot"] = round_pick(row["round"], row["pick_in_round"])
        return summary

    next_mine = summarize(mine_next)
    if next_mine and on_clock:
        next_mine["picks_away"] = mine_next["pick_no"] - on_clock["pick_no"]

    return {
        "source": f"{API}/draft/{draft['draft_id']}",
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "draft_id": str(draft["draft_id"]),
        "league_id": draft.get("league_id"),
        "league_name": (draft.get("metadata") or {}).get("name"),
        "season": draft.get("season"),
        "status": draft.get("status"),
        "started_at": iso(draft.get("start_time")),
        "last_picked_at": iso(draft.get("last_picked")),
        "format": {
            "type": board.type,
            "teams": board.teams,
            "rounds": board.rounds,
            "reversal_round": board.reversal_round or None,
            "scoring_type": (draft.get("metadata") or {}).get("scoring_type"),
        },
        "pick_count": board.pick_count,
        "picks_made": len(made),
        "picks_pending": board.pick_count - len(made),
        "on_the_clock": summarize(on_clock),
        "me": me,
        "my_next_pick": next_mine,
        "board_derivation": {
            "method": (
                f"{board.type}"
                + (f", order reverses at round {board.reversal_round}" if board.reversal_round else "")
                + "; traded picks applied"
            ),
            "checked_against_made_picks": checks["made_picks_checked"],
            "slot_and_roster_agree": checks["slot_and_roster_agree"],
            "rounds_exercised": checks["rounds_exercised"],
            "mismatches": checks["mismatches"],
        },
        "slots": [
            {
                "draft_slot": slot,
                "roster_id": board.slot_to_roster.get(slot),
                "user_id": user,
                "username": (by_user.get(str(user)) or {}).get("username"),
                "team_name": (by_user.get(str(user)) or {}).get("team_name"),
                # Both sides can be None — an unpublished draft order must not read
                # as every slot being mine.
                "is_mine": bool(me.get("user_id")) and str(user) == str(me["user_id"]),
            }
            for slot in range(1, board.teams + 1)
            for user in [board.slot_to_user.get(slot)]
        ],
        "traded_picks": fetched["traded"],
        "fields": FIELD_DEFINITIONS,
        "picks": rows,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def readme_check(rows: list[dict], me: dict) -> tuple[list[str], list[str]]:
    """Compare my derived picks with the sequence README.md documents."""
    mine = {row["round"]: row["pick_in_round"] for row in rows if row["is_mine"]}
    agree, disagree = [], []
    for round_no, expected in sorted(README_MY_PICK_IN_ROUND.items()):
        got = mine.get(round_no)
        line = f"{round_pick(round_no, expected)} expected, got " + (
            round_pick(round_no, got) if got else "no pick"
        )
        (agree if got == expected else disagree).append(line)
    return agree, disagree


def pool_join(rows: list[dict], pool_path: Path) -> dict | None:
    """How the made picks land in pool.json — the join this file exists to enable."""
    if not pool_path.is_file():
        return None
    try:
        with pool_path.open(encoding="utf-8") as handle:
            players = json.load(handle).get("players") or []
    except (OSError, json.JSONDecodeError, AttributeError):
        return None
    by_id = {str(p["sleeper_id"]): p for p in players if p.get("sleeper_id")}
    made = [row for row in rows if row["status"] == "made" and row["sleeper_id"]]
    hits = [(row, by_id[row["sleeper_id"]]) for row in made if row["sleeper_id"] in by_id]
    return {
        "pool_size": len(players),
        "pool_with_id": len(by_id),
        "made": len(made),
        "matched": len(hits),
        "outside_pool": [row for row in made if row["sleeper_id"] not in by_id],
        "top_50_gone": sum(1 for _, player in hits if player.get("rank", 0) <= 50),
        # A position mismatch on a joined id would mean match_sleeper.py joined the
        # wrong player — the one failure mode a name-based join can hide.
        "disagreements": [
            (row, player) for row, player in hits if row["position"] != player["position"]
        ],
    }


def report(document: dict, rows: list[dict], board: Board, pool_path: Path) -> None:
    out = sys.stderr
    fmt = document["format"]

    print(
        f"\ndraft {document['draft_id']} — {document.get('league_name')} "
        f"{document.get('season')}, status {document['status']}",
        file=out,
    )
    print(
        f"  {fmt['type']}, {fmt['teams']} teams x {fmt['rounds']} rounds = "
        f"{document['pick_count']} picks"
        + (f", reversal at round {fmt['reversal_round']}" if fmt["reversal_round"] else "")
        + f"; last pick {document['last_picked_at']}",
        file=out,
    )

    print("\norder", file=out)
    for round_no in range(1, min(board.rounds, 6) + 1):
        first = board.locate((round_no - 1) * board.teams + 1)[2]
        last = board.locate(round_no * board.teams)[2]
        print(
            f"  round {round_no:>2}  slots {first} -> {last}  "
            f"({'reversed' if board.is_reversed(round_no) else 'forward'})",
            file=out,
        )
    if board.rounds > 6:
        print(f"  ... through round {board.rounds}", file=out)

    check = document["board_derivation"]
    print("\nderivation vs what Sleeper reported", file=out)
    print(
        f"  slot and roster agree on {check['slot_and_roster_agree']}/"
        f"{check['checked_against_made_picks']} made picks "
        f"(rounds exercised: {check['rounds_exercised'] or 'none yet'})",
        file=out,
    )
    for bad in check["mismatches"][:10]:
        print(
            f"  ^ pick {bad['pick_no']}: reported {bad['reported']} vs derived {bad['derived']}",
            file=out,
        )

    me = document["me"]
    agree, disagree = readme_check(rows, me)
    print(
        f"\nmy picks — {me['username']}, slot {me['draft_slot']}, roster {me['roster_id']}",
        file=out,
    )
    print(
        f"  README's documented rounds match: {len(agree)}/{len(agree) + len(disagree)}",
        file=out,
    )
    for line in disagree:
        print(f"  ^ MISMATCH {line}", file=out)
    mine = [row for row in rows if row["is_mine"]]
    print(
        "  all: "
        + ", ".join(round_pick(row["round"], row["pick_in_round"]) for row in mine[:10])
        + (f", ... ({len(mine)} total)" if len(mine) > 10 else ""),
        file=out,
    )

    made = [row for row in rows if row["status"] == "made"]
    print(f"\npicks made ({len(made)})", file=out)
    for row in made[-12:]:
        print(
            f"  {round_pick(row['round'], row['pick_in_round']):>6} "
            f"#{row['pick_no']:<4} {(row['username'] or row['user_id'] or '?'):<16} "
            f"{(row['name'] or '?'):<24} {row['position'] or '?':<3} {row['team'] or '?':<4}"
            f"{'  <- mine' if row['is_mine'] else ''}",
            file=out,
        )
    if made:
        by_position = collections.Counter(row["position"] for row in made)
        print(
            "  by position: "
            + ", ".join(f"{pos} {n}" for pos, n in by_position.most_common()),
            file=out,
        )
    if document["on_the_clock"]:
        clock = document["on_the_clock"]
        print(
            f"  on the clock: #{clock['pick_no']} ({clock['slot']}) "
            f"{clock['username'] or clock['user_id']}",
            file=out,
        )
    if document["my_next_pick"]:
        mine_next = document["my_next_pick"]
        print(
            f"  my next: #{mine_next['pick_no']} ({mine_next['slot']}), "
            f"{mine_next.get('picks_away')} picks away",
            file=out,
        )

    print("\nintegrity", file=out)
    numbers = [row["pick_no"] for row in rows]
    print(
        f"  pick_no is 1..{len(rows)} gap-free: {numbers == list(range(1, len(rows) + 1))}",
        file=out,
    )
    per_slot = collections.Counter(row["draft_slot"] for row in rows)
    print(
        f"  every slot appears {board.rounds}x: "
        f"{set(per_slot.values()) == {board.rounds} and len(per_slot) == board.teams}",
        file=out,
    )
    drafted = [row["sleeper_id"] for row in made if row["sleeper_id"]]
    repeats = [i for i, n in collections.Counter(drafted).items() if n > 1]
    print(
        f"  no player drafted twice: {not repeats}"
        + (f" <- {repeats}" if repeats else "")
        + f"; every made pick has a player: {len(drafted) == len(made)}",
        file=out,
    )
    unowned = [row["pick_no"] for row in rows if row["user_id"] is None]
    print(
        f"  every pick has an owner: {not unowned}"
        + (f" <- {len(unowned)} without one, first {unowned[:5]}" if unowned else ""),
        file=out,
    )

    join = pool_join(rows, pool_path)
    print(f"\npool join ({paths.display(pool_path)})", file=out)
    if join is None:
        print("  pool.json not readable — skipped", file=out)
    else:
        print(
            f"  {join['matched']}/{join['made']} made picks are in the pool "
            f"({join['pool_with_id']}/{join['pool_size']} pool players carry a sleeper_id)",
            file=out,
        )
        print(f"  pool top 50 already gone: {join['top_50_gone']}", file=out)
        for row in join["outside_pool"][:15]:
            print(
                f"  ^ outside the pool: #{row['pick_no']} {row['name']} "
                f"{row['position']} {row['team']} (sleeper_id {row['sleeper_id']})",
                file=out,
            )
        for row, player in join["disagreements"][:10]:
            print(
                f"  ^ position disagrees on sleeper_id {row['sleeper_id']}: "
                f"sleeper {row['name']} {row['position']} vs pool {player['name']} "
                f"{player['position']}",
                file=out,
            )
    print(file=out)


# ---------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------


def selftest() -> int:
    """Check the board geometry offline, against cases a live fetch cannot reach.

    The live cross-check in ``pick_rows`` is the real guard, but it can only test rounds
    that have been drafted and trades that have been made. These cases cover the rest:
    every supported format's slot order, this league's own pick sequence as README.md
    states it, and the trade logic in both directions.
    """
    failures: list[str] = []
    checked = 0

    def check(label: str, got, want) -> None:
        nonlocal checked
        checked += 1
        if got != want:
            failures.append(f"{label}\n    got  {got}\n    want {want}")

    def fake(teams=4, rounds=3, reversal=3, type_="snake", order=True) -> dict:
        # Slot n -> roster 100+n, so a slot/roster mix-up cannot accidentally pass.
        return {
            "draft_id": "X", "season": "2026", "type": type_,
            "settings": {"teams": teams, "rounds": rounds, "reversal_round": reversal},
            "slot_to_roster_id": {str(n): 100 + n for n in range(1, teams + 1)},
            "draft_order": {f"u{n}": n for n in range(1, teams + 1)} if order else None,
        }

    def order_of(draft: dict) -> list[list[int]]:
        board = Board(draft)
        return [
            [board.locate(n)[2] for n in range((r - 1) * board.teams + 1, r * board.teams + 1)]
            for r in range(1, board.rounds + 1)
        ]

    # Slot order per format. A reversal round repeats the round before it, so from
    # there on the parity is inverted — which is the whole subtlety.
    check("snake, reversal at 3", order_of(fake(rounds=6)),
          [[1, 2, 3, 4], [4, 3, 2, 1], [4, 3, 2, 1], [1, 2, 3, 4], [4, 3, 2, 1], [1, 2, 3, 4]])
    check("snake, reversal at 2", order_of(fake(rounds=4, reversal=2)),
          [[1, 2, 3, 4], [1, 2, 3, 4], [4, 3, 2, 1], [1, 2, 3, 4]])
    check("snake, no reversal", order_of(fake(rounds=4, reversal=0)),
          [[1, 2, 3, 4], [4, 3, 2, 1], [1, 2, 3, 4], [4, 3, 2, 1]])
    check("linear", order_of(fake(rounds=2, type_="linear")), [[1, 2, 3, 4], [1, 2, 3, 4]])
    check("auction has no board", Board(fake(type_="auction")).problems(),
          ["draft type 'auction' has no pick order to derive"])
    check("0 teams is refused", bool(Board(fake(teams=0)).problems()), True)

    # This league, against the sequence README.md documents for slot 2.
    real = Board({"type": "snake", "settings": {"teams": 10, "rounds": 29, "reversal_round": 3}})
    mine = {r: p for n in range(1, 291) for r, p, s in [real.locate(n)] if s == 2}
    check("slot 2 matches README", {r: mine.get(r) for r in README_MY_PICK_IN_ROUND},
          README_MY_PICK_IN_ROUND)
    check("slot 2 picks once per round", len(mine), 29)
    check("every slot picks once per round",
          sorted(collections.Counter(real.locate(n)[2] for n in range(1, 291)).values()),
          [29] * 10)

    # Traded picks: slot 1's round-2 pick, originally roster 101, is now roster 103's.
    traded = [{"season": "2026", "round": 2, "roster_id": 101, "owner_id": 103}]
    board = Board(fake(), traded)
    check("traded pick goes to the acquirer", board.owner_roster(2, 1), 103)
    check("the same slot's other rounds are untouched",
          (board.owner_roster(1, 1), board.owner_roster(3, 1)), (101, 101))
    check("another team's round 2 is untouched", board.owner_roster(2, 2), 102)
    check("another season's trade is ignored",
          Board(fake(), [{**traded[0], "season": "2027"}]).owner_roster(2, 1), 101)
    check("a malformed trade is skipped", Board(fake(), [{"round": None}]).traded, {})

    users = {f"u{n}": {"username": f"name{n}", "team_name": None} for n in range(1, 5)}
    rows, _ = pick_rows(board, [], users, "u1")
    # Round 2 is reversed, so slot 1 picks last in it: pick 8 of 4 x 3.
    check("a pending traded pick is attributed to the acquirer",
          {k: rows[7][k] for k in ("draft_slot", "roster_id", "user_id", "is_mine")},
          {"draft_slot": 1, "roster_id": 103, "user_id": "u3", "is_mine": False})
    check("an untraded pick is still mine", rows[0]["is_mine"], True)

    # Once that traded pick is made, Sleeper's report must agree with the derivation...
    made = [{"pick_no": 8, "draft_slot": 1, "roster_id": 103, "picked_by": "u3",
             "player_id": "999", "is_keeper": None,
             "metadata": {"first_name": "A", "last_name": "B", "position": "WR", "team": "SF"}}]
    rows, checks = pick_rows(board, made, users, "u1")
    check("a made traded pick agrees with the derivation",
          (checks["slot_and_roster_agree"], checks["mismatches"]), (1, []))
    check("a made pick carries its player",
          {k: rows[7][k] for k in ("status", "sleeper_id", "name", "is_keeper")},
          {"status": "made", "sleeper_id": "999", "name": "A B", "is_keeper": False})
    # ...and the negative control: the same pick with the trade *not* applied is exactly
    # what the live check has to catch, or it is not checking anything.
    _, missed = pick_rows(Board(fake()), made, users, "u1")
    check("an unapplied trade is caught",
          (missed["slot_and_roster_agree"], len(missed["mismatches"])), (0, 1))

    # picked_by is empty when a pick was made for the team rather than by them.
    rows, _ = pick_rows(Board(fake()), [{"pick_no": 1, "draft_slot": 1, "roster_id": 101,
                                         "picked_by": "", "player_id": 42, "metadata": {}}],
                        users, "u1")
    check("an autopick still finds its owner", (rows[0]["user_id"], rows[0]["is_mine"]),
          ("u1", True))
    check("a player id is stringified", rows[0]["sleeper_id"], "42")

    # An unpublished draft order must leave picks unowned, not owned by everyone.
    rows, _ = pick_rows(Board(fake(order=False)), [], users, None)
    check("no draft order leaves owners null",
          ({row["user_id"] for row in rows}, {row["is_mine"] for row in rows}),
          ({None}, {False}))

    # Pick numbers that would corrupt an array indexed by pick_no — versus a gap, which
    # is legitimate in a keeper draft and must not be fatal.
    board = Board(fake())
    check("a pick_no off the board is fatal",
          bool(pick_number_problems([{"pick_no": 13}], board)[0]), True)
    check("a duplicate pick_no is fatal",
          bool(pick_number_problems([{"pick_no": 1}, {"pick_no": 1}], board)[0]), True)
    check("a gap is reported, not fatal",
          [bool(part) for part in pick_number_problems([{"pick_no": 1}, {"pick_no": 3}], board)],
          [False, True])
    check("a clean prefix is silent",
          pick_number_problems([{"pick_no": 1}, {"pick_no": 2}], board), ([], []))

    check("me resolves case-insensitively", resolve_me("NAME2", board, users),
          {"username": "name2", "user_id": "u2", "draft_slot": 2, "roster_id": 102})
    check("an unknown me resolves to nothing", resolve_me("nobody", board, users)["user_id"], None)

    for failure in failures:
        print(f"  MISMATCH {failure}", file=sys.stderr)
    print(
        f"selftest {'ok' if not failures else 'FAILED'}: {checked - len(failures)}/{checked} "
        "board-geometry checks passed (formats, this league's order, trades, autopicks)",
        file=sys.stderr,
    )
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-o", "--output", default=paths.DRAFT, type=Path)
    ap.add_argument("--draft-id", default=DRAFT_ID, help=f"default: {DRAFT_ID}")
    ap.add_argument(
        "--me",
        default=MY_USERNAME,
        help=f"Sleeper username whose picks get is_mine (default {MY_USERNAME})",
    )
    ap.add_argument("--api", default=API, help=f"default: {API}")
    ap.add_argument("--pool", default=paths.POOL, type=Path, help="--report checks the join here")
    ap.add_argument("--timeout", type=int, default=TIMEOUT_SECONDS)
    ap.add_argument("--report", action="store_true", help="print a validation summary to stderr")
    ap.add_argument(
        "--selftest",
        action="store_true",
        help="check the board geometry offline (no network), then exit",
    )
    ap.add_argument("--indent", type=int, default=2, help="JSON indent; 0 for compact")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    print(f"GET {args.api}/draft/{args.draft_id} (+picks, traded_picks, league users)", file=sys.stderr)
    try:
        fetched = fetch(args.draft_id, args.api, args.timeout)
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"error: request failed: {exc}", file=sys.stderr)
        return 1
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if fetched["warning"]:
        print(f"warning: {fetched['warning']}", file=sys.stderr)

    try:
        board = Board(fetched["draft"], fetched["traded"])
        fatal, notable = pick_number_problems(fetched["picks"], board)
        fatal = board.problems() + fatal
    except (TypeError, ValueError) as exc:
        print(f"error: draft {args.draft_id} is not shaped as expected: {exc}", file=sys.stderr)
        return 1
    if fatal:
        for problem in fatal:
            print(f"error: {problem}", file=sys.stderr)
        return 1
    for note in notable:
        print(f"warning: {note}", file=sys.stderr)
    if not board.slot_to_user:
        print("warning: draft_order is empty — pick owners will be null", file=sys.stderr)

    by_user = index_users(fetched["users"])
    me = resolve_me(args.me, board, by_user)
    if me["user_id"] is None:
        print(
            f"warning: no league user named {args.me!r} — no pick will be marked is_mine",
            file=sys.stderr,
        )
    elif me["draft_slot"] is None:
        print(f"warning: {me['username']} has no slot in this draft's order", file=sys.stderr)

    rows, checks = pick_rows(board, fetched["picks"], by_user, me["user_id"])
    if checks["mismatches"]:
        print(
            f"warning: the derived pick order disagrees with Sleeper on "
            f"{len(checks['mismatches'])} of {checks['made_picks_checked']} made picks — "
            "pending picks may be attributed to the wrong team; run --report",
            file=sys.stderr,
        )

    document = build_document(fetched, board, rows, checks, me)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=args.indent or None, ensure_ascii=False)
        handle.write("\n")

    clock = document["on_the_clock"]
    mine_next = document["my_next_pick"]
    print(
        f"draft: {document['picks_made']}/{document['pick_count']} picks made, "
        f"status {document['status']}"
        + (
            f"; on the clock #{clock['pick_no']} ({clock['slot']}) "
            f"{clock['username'] or clock['user_id']}"
            if clock
            else "; complete"
        )
        + (
            f"; mine #{mine_next['pick_no']} ({mine_next['slot']}) in "
            f"{mine_next.get('picks_away')}"
            if mine_next
            else ""
        )
        + f" -> {paths.display(args.output)}",
        file=sys.stderr,
    )
    if args.report:
        report(document, rows, board, args.pool)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

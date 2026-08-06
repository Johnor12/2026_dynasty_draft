#!/usr/bin/env python3
"""Compare each drafter's choices with every normalized ranking provider.

For every pick, the selected player is ranked only against players who were still
available at that moment. A provider fits when the drafter repeatedly selects near
the top of that provider's remaining board.

Usage:
    uv run data_source_investigator/investigate.py
    uv run data_source_investigator/investigate.py --report
    uv run data_source_investigator/investigate.py --selftest
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import statistics
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import paths

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
MISSING_PICK_LOSS = 5.0  # Same loss as repeatedly choosing 32nd among available players.
SKIPPED_EXAMPLES = 5


def normalized_name(value: str) -> str:
    plain = unicodedata.normalize("NFKD", value).casefold()
    parts = re.findall(r"[a-z0-9]+", plain)
    while parts and parts[-1] in SUFFIXES:
        parts.pop()
    return "".join(parts)


def name_key(player: dict) -> tuple[str, str]:
    return normalized_name(str(player.get("name") or "")), str(player.get("position") or "")


def is_taken(player: dict, ids: set[str], names: set[tuple[str, str]]) -> bool:
    sleeper_id = player.get("sleeper_id")
    return (sleeper_id is not None and str(sleeper_id) in ids) or name_key(player) in names


def chosen_index(available: list[dict], pick: dict) -> int | None:
    sleeper_id = pick.get("sleeper_id")
    if sleeper_id is not None:
        matches = [
            index
            for index, player in enumerate(available)
            if player.get("sleeper_id") is not None
            and str(player["sleeper_id"]) == str(sleeper_id)
        ]
        if len(matches) == 1:
            return matches[0]
    key = name_key(pick)
    matches = [index for index, player in enumerate(available) if name_key(player) == key]
    return matches[0] if len(matches) == 1 else None


def evidence_for_pick(
    source_players: list[dict],
    pick: dict,
    prior_ids: set[str],
    prior_names: set[tuple[str, str]],
) -> dict:
    available = [
        player for player in source_players if not is_taken(player, prior_ids, prior_names)
    ]
    index = chosen_index(available, pick)
    base = {
        "pick_no": pick["pick_no"],
        "round": pick["round"],
        "player": pick["name"],
        "position": pick["position"],
    }
    if index is None:
        return {
            **base,
            "matched": False,
            "overall_rank": None,
            "availability_rank": None,
            "skipped_count": None,
            "higher_ranked_available": [],
        }

    chosen = available[index]
    skipped = [
        {
            "name": player["name"],
            "position": player["position"],
            "overall_rank": player["rank"],
        }
        for player in available[: min(index, SKIPPED_EXAMPLES)]
    ]
    return {
        **base,
        "matched": True,
        "overall_rank": chosen["rank"],
        "availability_rank": index + 1,
        "skipped_count": index,
        "higher_ranked_available": skipped,
    }


def score_source(source: dict, picks: list[dict], made_before: dict[int, list[dict]]) -> dict:
    evidence = []
    losses = []
    availability_ranks = []
    for pick in picks:
        previous = made_before[pick["pick_no"]]
        prior_ids = {
            str(row["sleeper_id"]) for row in previous if row.get("sleeper_id") is not None
        }
        prior_names = {name_key(row) for row in previous}
        item = evidence_for_pick(source["players"], pick, prior_ids, prior_names)
        evidence.append(item)
        if item["matched"]:
            availability_rank = item["availability_rank"]
            availability_ranks.append(availability_rank)
            losses.append(math.log2(availability_rank))
        else:
            losses.append(MISSING_PICK_LOSS)

    mean_loss = statistics.fmean(losses)
    fit_score = 100.0 * 2 ** (-mean_loss / 3.0)
    total = len(picks)
    matched = len(availability_ranks)
    return {
        "source_id": source["id"],
        "source_name": source["name"],
        "fit_score": round(fit_score, 1),
        "picks": total,
        "matched_picks": matched,
        "coverage": round(matched / total, 3),
        "mean_availability_rank": (
            round(statistics.fmean(availability_ranks), 2) if availability_ranks else None
        ),
        "median_availability_rank": (
            round(statistics.median(availability_ranks), 2) if availability_ranks else None
        ),
        "top_available_rate": round(sum(rank == 1 for rank in availability_ranks) / total, 3),
        "top_3_available_rate": round(sum(rank <= 3 for rank in availability_ranks) / total, 3),
        "mean_log2_loss": round(mean_loss, 3),
        "evidence": evidence,
    }


def confidence(scores: list[dict], pick_count: int) -> tuple[str, float | None]:
    if len(scores) < 2:
        return "insufficient", None
    gap = round(scores[0]["fit_score"] - scores[1]["fit_score"], 1)
    if pick_count < 3:
        return "insufficient", gap
    if gap < 5:
        return "weak", gap
    if pick_count >= 6 and gap >= 12:
        return "strong", gap
    return "moderate", gap


def public_score(score: dict) -> dict:
    return {key: value for key, value in score.items() if key != "evidence"}


def investigate(rankings: dict, draft: dict) -> dict:
    sources = rankings["sources"]
    made = sorted(
        (pick for pick in draft["picks"] if pick["status"] == "made"),
        key=lambda pick: pick["pick_no"],
    )
    made_before = {
        pick["pick_no"]: [previous for previous in made if previous["pick_no"] < pick["pick_no"]]
        for pick in made
    }
    by_owner: dict[tuple[int | None, str | None], list[dict]] = defaultdict(list)
    for pick in made:
        by_owner[(pick.get("roster_id"), pick.get("username"))].append(pick)

    owners = []
    for (roster_id, username), picks in sorted(
        by_owner.items(), key=lambda item: (item[0][1] or "", item[0][0] or 0)
    ):
        scores = [score_source(source, picks, made_before) for source in sources]
        scores.sort(key=lambda score: (-score["fit_score"], -score["coverage"], score["source_id"]))
        label, gap = confidence(scores, len(picks))
        owners.append(
            {
                "roster_id": roster_id,
                "username": username,
                "pick_count": len(picks),
                "picks": [
                    {
                        "pick_no": pick["pick_no"],
                        "round": pick["round"],
                        "name": pick["name"],
                        "position": pick["position"],
                    }
                    for pick in picks
                ],
                "inferred_source": public_score(scores[0]),
                "runner_up": public_score(scores[1]) if len(scores) > 1 else None,
                "confidence": label,
                "score_gap": gap,
                "provider_scores": scores,
            }
        )

    return {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "draft": {
            "draft_id": draft.get("draft_id"),
            "draft_fetched_at": draft.get("fetched_at"),
            "draft_started_at": draft.get("started_at"),
            "picks_analyzed": len(made),
        },
        "ranking_snapshot": {
            "generated_at": rankings.get("generated_at"),
            "source_count": len(sources),
            "sources": [
                {
                    "id": source["id"],
                    "name": source["name"],
                    "format": source.get("format"),
                    "fetched_at": source.get("fetched_at"),
                    "player_count": source["player_count"],
                }
                for source in sources
            ],
        },
        "method": {
            "unit": "rank of the selected player among that provider's players still available",
            "fit_score": (
                "100 * 2 ** (-mean(log2(availability_rank)) / 3); higher is closer. "
                "A missing player receives the same loss as availability rank 32."
            ),
            "interpretation": (
                "The inferred source is the closest supplied board, not proof the drafter used it. "
                "Confidence measures only separation from the runner-up."
            ),
        },
        "owner_count": len(owners),
        "owners": owners,
    }


def selftest() -> list[str]:
    problems = []
    source = {
        "players": [
            {"rank": 1, "name": "Alpha Jr.", "position": "WR", "sleeper_id": "1"},
            {"rank": 2, "name": "Bravo", "position": "RB", "sleeper_id": "2"},
            {"rank": 3, "name": "Charlie", "position": "QB", "sleeper_id": "3"},
        ]
    }
    pick = {"pick_no": 2, "round": 1, "name": "Bravo", "position": "RB", "sleeper_id": "2"}
    evidence = evidence_for_pick(source["players"], pick, {"1"}, {("alpha", "WR")})
    if evidence["availability_rank"] != 1:
        problems.append("a previously drafted player was not removed from the available board")
    suffix_pick = {
        "pick_no": 1,
        "round": 1,
        "name": "Alpha",
        "position": "WR",
        "sleeper_id": None,
    }
    evidence = evidence_for_pick(source["players"], suffix_pick, set(), set())
    if evidence["availability_rank"] != 1:
        problems.append("suffix-insensitive name fallback did not match")
    missing = {"pick_no": 1, "round": 1, "name": "Nobody", "position": "TE"}
    evidence = evidence_for_pick(source["players"], missing, set(), set())
    if evidence["matched"]:
        problems.append("a missing player was reported as matched")
    return problems


def write_json(path: Path, payload: dict, indent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=indent, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--rankings", type=Path, default=paths.RANKINGS)
    ap.add_argument("--draft", type=Path, default=paths.DRAFT)
    ap.add_argument("-o", "--output", type=Path, default=paths.REPORT)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--indent", type=int, default=2)
    args = ap.parse_args(argv)

    if args.selftest:
        problems = selftest()
        if problems:
            for problem in problems:
                print(f"selftest: {problem}", file=sys.stderr)
            return 1
        print("data-source investigator selftest passed", file=sys.stderr)
        return 0

    try:
        rankings = json.loads(args.rankings.read_text())
        draft = json.loads(args.draft.read_text())
        if len(rankings["sources"]) < 2:
            raise ValueError("need at least two ranking sources")
        if not any(pick["status"] == "made" for pick in draft["picks"]):
            raise ValueError("draft has no made picks to investigate")
        result = investigate(rankings, draft)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"cannot investigate sources: {exc}", file=sys.stderr)
        return 1

    write_json(args.output, result, args.indent)
    if args.report:
        for owner in result["owners"]:
            best = owner["inferred_source"]
            print(
                f"  {owner['username'] or 'roster ' + str(owner['roster_id'])}: "
                f"{best['source_name']} {best['fit_score']:.1f} "
                f"({owner['confidence']}, +{owner['score_gap'] or 0:.1f} over runner-up)",
                file=sys.stderr,
            )
    print(
        f"investigated {result['draft']['picks_analyzed']} picks for "
        f"{result['owner_count']} drafters -> {paths.display(args.output)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

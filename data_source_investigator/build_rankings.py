#!/usr/bin/env python3
"""Normalize public snapshots and manual CSV exports into one rankings file.

Built-in inputs come from ``fetch_rankings.py``. Any ``data/manual/*.csv`` file
is treated as another provider; its required columns are ``rank,name,position``
and its optional columns are ``team,sleeper_id,value``.

Usage:
    uv run data_source_investigator/build_rankings.py
    uv run data_source_investigator/build_rankings.py --report
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sys
import unicodedata
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable

import paths

POSITIONS = {"QB", "RB", "WR", "TE"}
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def words(value: str) -> list[str]:
    plain = unicodedata.normalize("NFKD", value).casefold()
    return re.findall(r"[a-z0-9]+", plain)


def normalized_name(value: str, *, drop_suffix: bool = False) -> str:
    parts = words(value)
    if drop_suffix:
        while parts and parts[-1] in SUFFIXES:
            parts.pop()
    return "".join(parts)


class PlayerResolver:
    """Resolve provider names onto the pool's canonical Sleeper ids."""

    def __init__(self, pool: dict):
        self.ids = {str(player["sleeper_id"]): player for player in pool["players"]}
        self.full: dict[tuple[str, str], list[dict]] = defaultdict(list)
        self.base: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for player in pool["players"]:
            position = player["position"]
            self.full[(normalized_name(player["name"]), position)].append(player)
            self.base[
                (normalized_name(player["name"], drop_suffix=True), position)
            ].append(player)

    def resolve(self, player: dict) -> str | None:
        supplied = player.get("sleeper_id")
        if supplied is not None and str(supplied) in self.ids:
            return str(supplied)

        position = player["position"]
        name = player["name"]
        for index, key in (
            (self.full, (normalized_name(name), position)),
            (self.base, (normalized_name(name, drop_suffix=True), position)),
        ):
            matches = index.get(key, [])
            if len(matches) == 1:
                return str(matches[0]["sleeper_id"])
        return None


def js_json(text: str, marker: str):
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"missing JavaScript marker {marker!r}")
    start += len(marker)
    while start < len(text) and text[start].isspace():
        start += 1
    value, _ = json.JSONDecoder().raw_decode(text, start)
    return value


def player(
    rank: int,
    name: str,
    position: str,
    team: str | None,
    value: int | float | None,
    sleeper_id: str | None = None,
) -> dict:
    return {
        "rank": int(rank),
        "name": name.strip(),
        "position": position.strip().upper(),
        "team": team.strip().upper() if team and team.strip() else None,
        "sleeper_id": str(sleeper_id) if sleeper_id not in (None, "") else None,
        "value": value,
    }


def parse_fantasycalc(path: Path) -> list[dict]:
    rows = json.loads(path.read_text())
    result = []
    for row in rows:
        raw = row["player"]
        if raw["position"] in POSITIONS:
            result.append(
                player(
                    row["overallRank"],
                    raw["name"],
                    raw["position"],
                    raw.get("maybeTeam"),
                    row.get("value"),
                    raw.get("sleeperId"),
                )
            )
    return result


def parse_keeptradecut(path: Path) -> list[dict]:
    rows = js_json(path.read_text(), "var playersArray =")
    ranked = []
    for raw in rows:
        if raw["position"] not in POSITIONS:
            continue
        # KTC calls its lightest TE-premium setting TE+. Its own description says
        # this includes a +0.5 PPR boost, which is the setting in this league.
        values = raw["superflexValues"]["tep"]
        ranked.append(
            player(
                values["rank"],
                raw["playerName"],
                raw["position"],
                raw.get("team"),
                values.get("value"),
            )
        )
    # KTC's TE+ overlay currently emits a few duplicate rank numbers even though
    # the values are distinct. Value is what drives its displayed board.
    ranked.sort(key=lambda row: (-row["value"], row["rank"], row["name"]))
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


class RankingsTable(HTMLParser):
    """Capture the five-column SEO table published by Dynasty Nerds."""

    def __init__(self):
        super().__init__()
        self.active = False
        self.in_row = False
        self.in_cell = False
        self.rows: list[list[str]] = []
        self.row: list[str] = []
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "div" and dict(attrs).get("id") == "dr-seo-table":
            self.active = True
        elif self.active and tag == "tr":
            self.in_row = True
            self.row = []
        elif self.in_row and tag in {"th", "td"}:
            self.in_cell = True
            self.text = []

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self.in_cell and tag in {"th", "td"}:
            self.row.append("".join(self.text).strip())
            self.in_cell = False
        elif self.in_row and tag == "tr":
            self.rows.append(self.row)
            self.in_row = False


def parse_dynastynerds(path: Path) -> list[dict]:
    parser = RankingsTable()
    parser.feed(path.read_text())
    result = []
    for row in parser.rows:
        if len(row) != 5 or not row[0].isdigit():
            continue
        result.append(player(int(row[0]), row[1], row[3], row[2], int(row[4].replace(",", ""))))
    return result


def parse_fantasypros(path: Path) -> list[dict]:
    data = js_json(path.read_text(), "var ecrData =")
    result = []
    for raw in data["players"]:
        position = raw["player_position_id"]
        if position in POSITIONS:
            result.append(
                player(
                    raw["rank_ecr"],
                    raw["player_name"],
                    position,
                    raw.get("player_team_id"),
                    float(raw["rank_ave"]),
                )
            )
    return result


PARSERS: dict[str, tuple[str, str, Callable[[Path], list[dict]]]] = {
    "fantasycalc": ("FantasyCalc", "fantasycalc.json", parse_fantasycalc),
    "keeptradecut": ("KeepTradeCut", "keeptradecut.html", parse_keeptradecut),
    "dynastynerds": ("Dynasty Nerds", "dynastynerds.html", parse_dynastynerds),
    "fantasypros": ("FantasyPros ECR", "fantasypros.html", parse_fantasypros),
}


def parse_manual(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"rank", "name", "position"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing CSV columns: {', '.join(sorted(missing))}")
        rows = []
        for raw in reader:
            value: int | float | None = None
            if raw.get("value"):
                value = float(raw["value"])
                if value.is_integer():
                    value = int(value)
            rows.append(
                player(
                    int(raw["rank"]),
                    raw["name"],
                    raw["position"],
                    raw.get("team"),
                    value,
                    raw.get("sleeper_id"),
                )
            )
        return rows


def draftsharks_adp(pool: dict) -> list[dict]:
    available = [row for row in pool["players"] if row.get("adp") is not None]
    available.sort(key=lambda row: (row["adp"], row["rank"]))
    return [
        player(
            rank,
            raw["name"],
            raw["position"],
            raw.get("team"),
            raw["adp"],
            raw.get("sleeper_id"),
        )
        for rank, raw in enumerate(available, start=1)
    ]


def validate(source_id: str, rows: list[dict]) -> list[str]:
    problems = []
    if len(rows) < 50:
        problems.append(f"{source_id}: only {len(rows)} players (expected at least 50)")
    ranks = [row["rank"] for row in rows]
    if any(rank < 1 for rank in ranks):
        problems.append(f"{source_id}: ranks must be positive")
    if len(ranks) != len(set(ranks)):
        problems.append(f"{source_id}: duplicate ranks")
    bad_positions = sorted({row["position"] for row in rows} - POSITIONS)
    if bad_positions:
        problems.append(f"{source_id}: unsupported positions {bad_positions}")
    identities = [
        (normalized_name(row["name"], drop_suffix=True), row["position"]) for row in rows
    ]
    if len(identities) != len(set(identities)):
        problems.append(f"{source_id}: duplicate player identities")
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
    ap.add_argument("--raw", type=Path, default=paths.RAW)
    ap.add_argument("--manual", type=Path, default=paths.MANUAL)
    ap.add_argument("--pool", type=Path, default=paths.POOL)
    ap.add_argument("-o", "--output", type=Path, default=paths.RANKINGS)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--indent", type=int, default=2)
    args = ap.parse_args(argv)

    try:
        pool = json.loads(args.pool.read_text())
        resolver = PlayerResolver(pool)
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"cannot load pool {args.pool}: {exc}", file=sys.stderr)
        return 1

    fetch_meta = {}
    meta_path = args.raw / "fetch_meta.json"
    if meta_path.exists():
        try:
            fetch_meta = json.loads(meta_path.read_text()).get("sources", {})
        except (OSError, json.JSONDecodeError) as exc:
            print(f"cannot load fetch metadata {meta_path}: {exc}", file=sys.stderr)
            return 1

    sources: list[dict] = []
    failures: list[str] = []
    for source_id, (name, filename, parser) in PARSERS.items():
        source_path = args.raw / filename
        if not source_path.exists():
            failures.append(f"{source_id}: missing {source_path}")
            continue
        try:
            rows = parser(source_path)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"{source_id}: {exc}")
            continue
        rows.sort(key=lambda row: row["rank"])
        problems = validate(source_id, rows)
        if problems:
            failures.extend(problems)
            continue
        for row in rows:
            row["sleeper_id"] = resolver.resolve(row)
        metadata = fetch_meta.get(source_id, {})
        sources.append(
            {
                "id": source_id,
                "name": name,
                "url": metadata.get("url"),
                "format": metadata.get("format"),
                "fetched_at": metadata.get("fetched_at"),
                "player_count": len(rows),
                "matched_to_sleeper": sum(row["sleeper_id"] is not None for row in rows),
                "players": rows,
            }
        )

    manual_paths = sorted(args.manual.glob("*.csv")) if args.manual.exists() else []
    for source_path in manual_paths:
        source_id = source_path.stem
        if source_id in PARSERS or source_id == "draftsharks_adp":
            failures.append(f"manual source id {source_id!r} conflicts with a built-in source")
            continue
        try:
            rows = parse_manual(source_path)
            rows.sort(key=lambda row: row["rank"])
            problems = validate(source_id, rows)
            if problems:
                failures.extend(problems)
                continue
            for row in rows:
                row["sleeper_id"] = resolver.resolve(row)
            sources.append(
                {
                    "id": source_id,
                    "name": source_id.replace("_", " ").title(),
                    "url": None,
                    "format": "user-supplied CSV; verify scoring format",
                    "fetched_at": None,
                    "player_count": len(rows),
                    "matched_to_sleeper": sum(row["sleeper_id"] is not None for row in rows),
                    "players": rows,
                }
            )
        except (OSError, KeyError, TypeError, ValueError) as exc:
            failures.append(f"{source_id}: {exc}")

    ds_rows = draftsharks_adp(pool)
    for row in ds_rows:
        row["sleeper_id"] = resolver.resolve(row)
    sources.append(
        {
            "id": "draftsharks_adp",
            "name": "DraftSharks superflex ADP",
            "url": None,
            "format": "dynasty superflex ADP from pool.json; TEP is not reflected in ADP",
            "fetched_at": None,
            "player_count": len(ds_rows),
            "matched_to_sleeper": sum(row["sleeper_id"] is not None for row in ds_rows),
            "players": ds_rows,
        }
    )

    if failures:
        print("ranking normalization failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    if len(sources) < 2:
        print("need at least two ranking sources to compare", file=sys.stderr)
        return 1

    payload = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "league_format": {
            "teams": 10,
            "type": "dynasty superflex",
            "ppr": 0.5,
            "te_reception_bonus": 0.5,
        },
        "source_count": len(sources),
        "sources": sources,
    }
    write_json(args.output, payload, args.indent)

    if args.report:
        for source in sources:
            print(
                f"  {source['id']}: {source['player_count']} players, "
                f"{source['matched_to_sleeper']} matched to Sleeper",
                file=sys.stderr,
            )
    print(f"normalized {len(sources)} sources -> {paths.display(args.output)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

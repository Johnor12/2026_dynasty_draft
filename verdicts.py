#!/usr/bin/env python3
"""How every package the sweep kept reads on KeepTradeCut's calculator, as he will see it.

sweep.py ranks packages by what they add to my title odds and dumps every one it kept to
sweep_rows.json. This sorts those by the verdict the calculator page gives him, which is
the likelihood he accepts: HE WINS (outside its fairness band, KTC_VARIANCE, in his favor)
needs only his interest in the pieces; FAIR needs a thesis; I WIN (by at most KTC_SLACK,
or the sweep would not have priced it) needs him to want the pieces more than the
calculator does. The league does concede on the calculator: margins.py puts the conceding
side of every completed trade this season between -100 and -3,500. Whether the package
fits his stance, the sweep's second filter, is shown beside each verdict rather than
folded into it. Within a list the best package per opponent and target is kept.

Usage:
    uv run verdicts.py
"""

from __future__ import annotations

import json

from trades import REPO_ROOT, ktc_close

ROWS = REPO_ROOT / "sweep_rows.json"
TOP = 40
TOP_PER_VERDICT = 25
TOP_PER_OPPONENT = 3
VERDICTS = ("he wins", "fair", "I win")


def verdict(r: dict) -> str:
    if ktc_close(r["t_me"], r["t_him"]):
        return "fair"
    return "he wins" if r["t_him"] > r["t_me"] else "I win"


def best_per(rows: list[dict], key) -> list[dict]:
    best: dict = {}
    for r in rows:
        k = key(r)
        if k not in best or r["titles"] > best[k]["titles"]:
            best[k] = r
    return sorted(best.values(), key=lambda r: -r["titles"])


def line(r: dict) -> str:
    gap = abs(r["t_me"] - r["t_him"])
    reads = "fair" if r["verdict"] == "fair" else f"{r['verdict']} {gap:.0f}"
    return (
        f"{r['titles']:+6.1%} (2026 {r['now']:+5.1%}, 27/28 {r['later']:+5.1%}/yr) him {r['his']:+5.1%} | "
        f"calc {r['calc']:+6.0f} {reads:<13} {'fits' if r['fits'] else 'vs stance':<9} | "
        f"{r['opp']}: give {', '.join(r['give'])} | get {', '.join(r['get'])}"
    )


def main() -> int:
    rows = json.loads(ROWS.read_text())
    for r in rows:
        r["get"], r["give"] = tuple(r["get"]), tuple(r["give"])
        r["verdict"] = verdict(r)
    by_opp_target = lambda r: (r["opp"], r["get"])  # noqa: E731
    counts = {v: sum(1 for r in rows if r["verdict"] == v) for v in VERDICTS}
    print(
        f"{len(rows)} kept packages from the last sweep; calculator verdicts {counts}\n"
        "columns: my cumulative title odds (2026 + 2 x 27/28), his; calc is my margin on the "
        "calculator, then what its page tells him and whether the package fits his stance"
    )
    print(f"\n== top {TOP} by my title odds ==")
    for r in best_per(rows, by_opp_target)[:TOP]:
        print(line(r))
    for v in VERDICTS:
        sub = [r for r in rows if r["verdict"] == v]
        print(f"\n== calculator says {v.upper()}: {counts[v]} packages ==")
        for r in best_per(sub, by_opp_target)[:TOP_PER_VERDICT]:
            print(line(r))
        print("  -- also fitting his stance --")
        for r in best_per([r for r in sub if r["fits"]], by_opp_target)[:TOP_PER_VERDICT]:
            print("  " + line(r))
    print(f"\n== best {TOP_PER_OPPONENT} targets per opponent and verdict ==")
    for opp in sorted({r["opp"] for r in rows}):
        print(f"-- {opp} ({next(r['stance'] for r in rows if r['opp'] == opp)})")
        for v in VERDICTS:
            for r in best_per([r for r in rows if r["opp"] == opp and r["verdict"] == v], lambda r: r["get"])[:TOP_PER_OPPONENT]:
                print("  " + line(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

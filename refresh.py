#!/usr/bin/env python3
"""Pull the live board and re-rank against it — the loop to run between picks.

Two steps, both of which already stand alone; this only fixes the order and stops on
the first failure:

    1. draft_pipeline/fetch_draft.py   Sleeper's draft API -> draft.json
    2. rank_vor.py                     pool.json + draft.json -> rankings.json

The pool pipeline is deliberately not a step. It is local, offline and re-run a handful
of times all offseason when the projections change; this runs every few minutes during a
draft and must not re-parse 8 MB of html to do it.

Each step is a separate ``uv run``, not an import, because the two pipelines share no
code — ``draft_pipeline/`` is reached the same way the README says to reach it. Both run
with the repo root as their working directory, so their own default paths apply and this
works from anywhere (``rank_vor.py`` resolves ``pool.json`` and ``draft.json`` from the
shell's cwd, not from the script).

Usage:
    uv run refresh.py            # draft.json, then rankings.json
    uv run refresh.py --report   # + both steps' validation summaries on stderr
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--report", action="store_true", help="pass --report to both steps")
    args = ap.parse_args(argv)

    report = ["--report"] if args.report else []
    steps = [
        ("draft", ["draft_pipeline/fetch_draft.py", *report]),
        ("rank", ["rank_vor.py", *report]),
    ]

    for number, (name, command) in enumerate(steps, start=1):
        print(f"\n=== [{number}/{len(steps)}] {name} ===", file=sys.stderr)
        started = time.monotonic()
        code = subprocess.run(["uv", "run", *command], cwd=REPO_ROOT).returncode
        if code != 0:
            print(f"refresh failed at step '{name}' (exit {code})", file=sys.stderr)
            return code
        print(f"--- {name} ok in {time.monotonic() - started:.1f}s", file=sys.stderr)

    print("\nrefresh complete -> rankings.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

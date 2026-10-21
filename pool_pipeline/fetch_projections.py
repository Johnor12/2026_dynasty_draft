#!/usr/bin/env python3
"""Fetch DraftSharks' dynasty rankings rows into data/projections.html.

The rankings page fills its table from ``/dynasty-rankings/load-rows``: a server-rendered
HTML fragment of one ``<tbody data-player-row>`` per player carrying every scoring scheme
and horizon in data attributes, which is exactly what parse_projections.py reads. The
endpoint needs no login, returns the whole list (about 880 rows, IDP and kickers included)
when asked for more than it has, and ``pprSuperflexSlug`` picks the scheme the bare
``data-value`` carries, so .5 PPR superflex matches the draft-time save.

In season DraftSharks' 1-year column is season-to-date plus rest of season, so 3yr - 1yr
is still the next two seasons; it publishes no ADP then.

Usage:
    uv run pool_pipeline/fetch_projections.py
"""

from __future__ import annotations

import re
import sys
import urllib.request

import paths

URL = (
    "https://www.draftsharks.com/dynasty-rankings/load-rows"
    "?offset=0&limit=2000&fantasyPosition=&pprSuperflexSlug=half-ppr-superflex"
    "&playerGroup=all&sort=-dsValue"
)
# Cloudflare fronts the site and answers a browser user agent.
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0 Safari/537.36",
    "Accept": "text/html,*/*",
}
MIN_ROWS = 800  # the list has held 877-900 rows; fewer means the endpoint changed
TIMEOUT_SECONDS = 120


def main() -> int:
    request = urllib.request.Request(URL, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        html = response.read().decode("utf-8")
    rows = len(re.findall(r"<tbody\s+data-player-row", html))
    if rows < MIN_ROWS:
        print(f"error: {rows} player rows in the response, expected at least {MIN_ROWS}", file=sys.stderr)
        return 1
    paths.PROJECTIONS_HTML.write_text(html, encoding="utf-8")
    print(
        f"wrote {rows} player rows ({len(html) / 1e6:.1f} MB) to {paths.display(paths.PROJECTIONS_HTML)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

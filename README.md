# 2026 dynasty fantasy draft

A toolkit for a 10-team superflex dynasty startup draft, and now for in-season
waiver moves. Independent processes publish stable JSON artifacts at the repository
root; the ranker consumes those artifacts and the static dashboard renders the result.

## League assumptions

- 0.5 PPR, plus another 0.5 points per TE reception
- Starters: 1 QB, 2 RB, 3 WR, 1 TE, 2 W/R/T flex, 1 W/R/T/Q superflex
- 15 bench, 3 IR, and 4 rookie-only taxi spots
- 10 teams and 29 drafted players per team (290 picks)
- Snake draft with a third-round reversal
- Before trades, my slot is 1.02: 1.02, 2.09, 3.09, 4.02, 5.09, 6.02, …,
  28.02, 29.09

These are project assumptions, not runtime configuration. Ranker constants live in
`ranker/league.py`.

## Setup

[uv](https://docs.astral.sh/uv/) pins Python 3.12.

```bash
uv sync
uv run <script>
```

Commands work from the repository root. Pipeline defaults are anchored to their own
directories, so their documented direct invocations also work from inside the pipeline.

## Data flow

```text
pool_pipeline/ ───────────────> pool.json ──────────┐
                                                   │
draft_pipeline/ ──────────────> draft.json ─────────┼─> rank.py ────> rankings.json
                                                   │
data_source_investigator/ ────> data_source_matches.json
                         └────> data/rankings.json ─┘
```

In season:

```text
pool.json ─────────────────────────────────────────┐
data_source_investigator/ ────> data/rankings.json ─┼─> waivers.py ──> +EV moves on stdout
league_pipeline/ ─────────────> league.json ────────┤
data_source_investigator/data/raw/ ─────────────────┼─> trades.py ───> +EV trades on stdout
league_pipeline/ ─────────────> pick_history.json ──┘
```

The published files have distinct owners:

- `pool.json`: 350 projection-backed QB/RB/WR/TE players, keyed to Sleeper
- `draft.json`: all 290 made and pending picks from Sleeper
- `data_source_matches.json`: the provider board closest to each opponent's picks
- `rankings.json`: undrafted-player rankings, recommendations, simulations, and validation
- `league.json`: current Sleeper rosters (active, taxi, IR) and every player's
  rest-of-season projected points in this league's scoring
- `pick_history.json`: for six past draft classes, the player at each rookie slot and his
  first two seasons of points in this league's scoring

`sleeper_id` is the cross-process player key. `roster_id` and `draft_slot` connect
opponent source matches to the live board.

## Components

- [Pool pipeline](pool_pipeline/README.md): provider HTML to the league-specific pool
- [Draft pipeline](draft_pipeline/README.md): Sleeper API to the complete live board
- [Data-source investigator](data_source_investigator/README.md): normalize provider
  boards and infer opponent strategies
- [Ranker](ranker/README.md): wire-level solver, opponent simulation, planning,
  and output contracts
- `league_pipeline/fetch_league.py`: Sleeper league rosters and weekly projections to
  `league.json`
- `league_pipeline/fetch_pick_history.py`: what rookie slots have returned, to
  `pick_history.json`
- `waivers.py`: in-season refresh and waiver-move pricing
- `trades.py`: trade pricing against one opponent, proposed or swept
- `index.html`: dependency-free dashboard for `rankings.json`
- `data_source_investigator/index.html`: source-fit and pick-evidence dashboard
- `serve.py`: serves both dashboards at http://127.0.0.1:8123

Each component keeps its own paths, entry points, and implementation context. Offline
checks remain beside the draft, investigator, and ranker code they exercise. The pipelines
meet through their published JSON contracts rather than shared orchestration.

The ranker values a roster as expected optimal lineup points in separate year-one and
years-2–3 horizons. Position-wide availability determines when depth is called on, and
one unique final waiver body per position supplies the fallback. Personal and
opponent strategies are intentionally separate: my slot uses the projection-based roster
objective, while each opponent follows its inferred external board with roster-balance
adjustments and fitted choice noise. Opponent picks never use my projections or board.

## Common workflows

Rebuild the projection pool after saving updated provider HTML:

```bash
uv run pool_pipeline/pipeline.py --report
```

Refresh ranking snapshots and opponent associations:

```bash
uv run data_source_investigator/pipeline.py --report
```

Refresh the live board and recommendations between picks:

```bash
uv run refresh.py --report
```

`refresh.py` deliberately does only three live steps: fetch the draft, re-run source
inference against the existing provider snapshot, then rank. It does not rebuild the
offline pool or fetch provider boards.

In season, refresh provider rankings and the league, then list +EV waiver moves:

```bash
uv run waivers.py
```

It refetches and rebuilds the provider boards, fetches `league.json`, and then prices
every free agent against my roster with the ranker's expected-lineup value. The yr1
horizon becomes rest-of-season points: Sleeper's weekly projections from the current
week through week 17, scored with the league's `scoring_settings`. Years 2–3 remain
DraftSharks' from `pool.json`, and are 0 for players DraftSharks does not project. The
best unrostered player at each position is the wire fallback, so a move's EV includes
the fallback it spends or creates. Each free agent is shown with his best active-roster
drop. Moves are not additive. The refreshed provider ranks are shown only as context
(`mkt`) and never enter EV.

Price a proposed trade, or sweep every 1-for-1, 2-for-1, 1-for-2, and 2-for-2 package
with one opponent:

```bash
uv run trades.py nedjh2 --tanking --give "Jeremiyah Love" --get "Joe Burrow"
uv run trades.py nedjh2 --tanking
```

It reads what `waivers.py` refreshed plus `pick_history.json`, and does not refetch. Both
rosters are priced with the same expected-lineup value and wire as waiver moves, and each
roster is refilled to the league shape afterwards: incoming players keep IR or taxi when a
spot is open, a surplus active body is cut cheapest-first, and an open spot takes the best
free agent.

My 2027 first and second are tradeable, priced as lotteries rather than projections.
Valuing a mid first as one averaged rookie made it bench depth worth +2 points, because
the average rookie never starts while the hit does. Instead a pick is the expectation, over
the players history put within two slots of mine, of the lineup value each would add to the
roster holding the pick, carrying his own position and his real first two seasons of
scoring as the pick's 2027-28 points. A pick that moves loses that value on the sender's
roster and gains it on the receiver's post-trade roster; one that stays put is taken as
worth the same before and after. `MY_2027_SLOT` in `trades.py` fixes the slot. The
history comes from `uv run league_pipeline/fetch_pick_history.py`, which proxies
rookie-draft order with NFL draft order among QB/RB/WR/TE (nflverse) and scores Sleeper's
season stats with the league's settings; it moves once a year, so it is not part of the
waivers refresh.

Every package also shows my market delta on KeepTradeCut superflex TE+, on FantasyCalc's
10-team values, and `mkt`, their blend with FantasyCalc rescaled onto KTC's scale. Pick
values come from the raw snapshots, which is why the KeepTradeCut fetch includes rookie
picks; `build_rankings.py` drops them from the board. The opponent is scored on both
horizons, or on 2027-28 only with `--tanking`. His EV is printed but a sweep filters on
what he will actually check: it lists packages that are +EV for me and cost him at most
`MARKET_SLACK` of the blended value he sends. A sweep runs on up to eight worker
processes.

Refreshing the boards overwrites the gitignored draft-time raw snapshots. It also
means a bare `rank.py` refuses to run until `investigate.py` re-matches against the
new snapshot, which `refresh.py` already does.

Run offline checks:

```bash
uv run rank.py --selftest
uv run draft_pipeline/fetch_draft.py --selftest
uv run data_source_investigator/investigate.py --selftest
uv run evaluate_opponents.py
```

Before and after changing an opponent model or pick policy, run
`uv run evaluate_opponents.py` and compare its replay accuracy.

## Dashboard and automation

Run `uv run serve.py` and open the local URL; direct `file://` access cannot fetch the
JSON files. The main dashboard also polls Sleeper for a compact live-status strip, shows
when its `draft.json` snapshot is stale, and pins the status of any active refresh or
deploy workflow run in that strip.

`.github/workflows/refresh.yml` runs the live refresh on manual dispatch and commits the
generated board, rankings, and source matches. `.github/workflows/deploy-pages.yml`
publishes both dashboards plus `rankings.json` and `data_source_matches.json` to GitHub
Pages, with the investigator at `data_source_investigator/`. They share one concurrency
group so refresh and deploy do not overlap.

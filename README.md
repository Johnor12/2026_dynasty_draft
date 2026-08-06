the projections.html are tables from a projections/rankings provider (draftsharks).

projections.html contains projections and rankings from a provider (draftsharks). it's an html table with rows like this for each player (brock bowers is the example): data-value="2191" data-scoring-value="1609" data-scoring-value-half-ppr="2191" data-scoring-value-ppr="2772" data-scoring-value-te-premium="3354" data-scoring-value-superflex="1609" data-scoring-value-half-ppr-superflex="2191" data-scoring-value-ppr-superflex="2772" data-scoring-value-te-premium-superflex="3354". its "3D value" assumes a superflex league and .5 ppr (unclear exactly what roster settings).

My league is .5 ppr with a .5 ppr tight end premium. Our rosters are 1 QB, 2 RB, 3 WR, 1 TE, 2 WRT, 1 WRTQ (superflex), with 15 man benches, 3 IR slots, and 4 rookie year taxi spots. The league has 10 teams (290 drafted players). We're doing a snake draft with a 3rd round reversal. My draft slot is 1.02, so my picks are: 1.02, 2.9, 3.9, 4.2, 5.9, 6.2 ... 28.2, 29.9.

## setup

[uv](https://docs.astral.sh/uv/) manages the interpreter; there is nothing to install
beyond it.

```
uv sync          # create .venv on the pinned Python (3.12, see .python-version)
uv run <script>  # run any script below — from the repo root or from inside a pipeline
```

Every dependency is the standard library: `html.parser` for the projections parse,
`urllib.request` for both Sleeper fetches, arithmetic for the ranker. `pyproject.toml`
declares no packages and `uv.lock` therefore locks none — uv is here for a pinned
interpreter and a reproducible env, not for resolving wheels. Nothing needs activating:
`uv run` finds the project by walking up from the current directory, so both invocation
styles the scripts document work unchanged.

## layout

Two independent pipelines, each publishing one file at the repo root. Everything a
pipeline reads and every intermediate it writes stays inside its own folder.

```
pool.json                    the draft pool — 350 players, 13 fields
draft.json                   the live board — 290 picks, made and pending
rank_vor.py                  pool.json + draft.json -> rankings.json (run separately)
refresh.py                   fetch the board, then re-rank — the between-picks loop
index.html                   static dashboard for rankings.json (serve the repo root, no build)
ranker/                      the ranker's internals, one module per concern:
  league.py                  league constants, strategy knobs, draft order
  pool.py                    pool.json -> Player objects
  board.py                   draft.json -> the simulation's starting state
  value.py                   roster valuation and replacement levels
  sim.py                     the draft simulation and the fixed point
  output.py                  rankings.json rows, payload, stderr reports
  validate.py                every-run invariants
  selftest.py                offline checks for states the live files can't reach

pool_pipeline/               provider html -> pool.json   (local, offline, 3 stages)
  pipeline.py                orchestrator: parse -> pool -> sleeper
  parse_projections.py       1. html -> projections.json   (faithful provider export)
  build_pool.py              2. projections.json -> pool.json  (this league's pool)
  match_sleeper.py           3. pool.json -> pool.json + sleeper_id
  fetch_sleeper.py           manual: download Sleeper's player dump (not a stage)
  paths.py
  data/
    projections.html         raw provider save, added by hand
    projections.json         intermediate, 900 players x 8 schemes
    sleeper_players.json     cached Sleeper dump, ~14 MB
    sleeper_players.meta.json  when it was fetched

draft_pipeline/              Sleeper draft API -> draft.json  (network, on demand)
  fetch_draft.py             the whole pipeline — one stage, so no orchestrator
  paths.py
```

**They are separate on purpose.** The pool pipeline is local, offline and deterministic,
and is re-run when the projections change — a handful of times all offseason. The draft
pipeline has no inputs on disk, caches nothing, and is re-run on demand during a live
draft. Sharing an orchestrator would mean either a pool rebuild that hits the live draft
or a draft refresh that re-parses 8 MB of html, so they share no code and no working
files — `draft_pipeline/` keeps its own small `paths.py` rather than importing one. They
meet only at `sleeper_id`: the key `match_sleeper.py` writes into every pool player is
the key every pick in `draft.json` carries. `rank_vor.py` is the one thing that reads both
and is where that join is actually performed.

## pool pipeline

```
uv run pool_pipeline/pipeline.py              # html -> projections.json -> pool.json + ids
uv run pool_pipeline/pipeline.py --report     # + every stage's validation summary on stderr
uv run pool_pipeline/pipeline.py --only pool  # single stage
uv run pool_pipeline/fetch_sleeper.py         # refresh the Sleeper dump (manual, ~14 MB)
uv run rank_vor.py                            # pool.json + draft.json -> rankings.json
```

Three stages, ordered: `parse_projections.py` (html -> `projections.json`, the full
provider export: 900 players, 8 schemes, 4 horizons, ~2.9 MB), `build_pool.py`
(-> `pool.json`, this league's 350-player draft pool with one value column, ~95 KB), then
`match_sleeper.py` (adds `sleeper_id` to that file in place). All are standalone CLIs; the
orchestrator only fixes the order and stops on the first failure. Stage 1 stays faithful to
what the provider published and is never narrowed; stage 2 is the narrow, draft-ready view,
so it can be re-run at a different rank limit without re-parsing 8 MB of html and the
dropped columns stay recoverable. Stage 3 comes last because stage 2 rewrites `pool.json`
from scratch, dropping the ids stage 3 adds — so a rebuild always re-joins them.

Default paths are anchored to the scripts, not the shell, so every command above works
from the repo root or from inside `pool_pipeline/`.

`rank_vor.py` (pool.json + draft.json -> rankings.json) is deliberately not a pipeline
stage: it takes a simulation seed and strategy knobs and is re-run far more often than the
data is built.

## parsing

`parse_projections.py` turns `data/projections.html` into `data/projections.json` — 900 players, ~2.5 MB, about 1.5s. Python stdlib only (`html.parser`), no dependencies.

```
uv run pool_pipeline/parse_projections.py                       # -> data/projections.json
uv run pool_pipeline/parse_projections.py --report              # + validation summary on stderr
uv run pool_pipeline/parse_projections.py in.html -o out.json
```

No browser or JS execution is needed: the page is server-rendered and **every scoring scheme is already in the DOM** as `data-scoring-value-*` attributes on each cell. Alpine.js only toggles which one is visible. So all 8 schemes (`standard`, `half_ppr`, `ppr`, `te_premium`, each also `_superflex`) come straight out of the static markup.

Each player gets identity fields (id, name, position, team, age, bye, rookie flag, tiers, profile path), the analyst comment, and per-scheme `three_d_value`, `adp`, and `projections` for the 1/3/5/10-year horizons. `half_ppr_superflex_3d_value` is lifted to the top level of each record.

## non-obvious quirks in the parsed output

**`data-value` is the .5 ppr superflex value at full precision.** The visible cell rounds it (Bijan Robinson displays `75`, the attribute says `75.1`). Kept as `<scheme>_precise` inside each scheme dict and as the top-level `half_ppr_superflex_3d_value`. Checking `data-value` against `data-scoring-value-half-ppr-superflex` across all 900 rows × 6 numeric columns gave 0 disagreements, which is also what confirms .5 ppr sf is the rendered default.

**The ranks baked into the html are stale — don't use them.** This file was saved from an already-hydrated page, and the printed rank numbers no longer match the values sitting next to them: 62 overall ranks are duplicated, 62 are missing, and document order jumps backwards once (doc position 417→418, rank 424→382). Document order is otherwise clean, descending by .5 ppr sf 3D value. The parser keeps the printed numbers verbatim as `rank_displayed` / `positional_rank_displayed` and adds derived `rank_by_3d_value` / `positional_rank_by_3d_value` (unique, gap-free, ties broken by document order). **Use the derived ones for a draft board.** `document_index` preserves the original row order.

**`te_premium` is not this league's scoring.** Draftsharks' `te_premium` is full ppr *plus* a 0.5/rec premium, i.e. 1.5 pts/rec for TEs, and is byte-identical to `ppr` for every non-TE. (Verified: the standard → half_ppr → ppr → te_premium steps are a uniform 0.5/rec on all 89 TEs, and te_premium == ppr for all 305 WR/RB.) This league is 0.5 base + 0.5 premium = 1.0/rec for TEs and 0.5/rec for WR/RB. So for raw points there is no single matching column — it's `half_ppr` for non-TEs and `ppr` for TEs. For 3D value there's no exact column either: `half_ppr_superflex` gives TEs no premium at all, while `te_premium_superflex` has the correct *relative* TE edge (+0.5/rec over WR/RB) but builds it on an inflated 1.0 reception base that also lifts every WR. That gap is what `build_pool.py` closes — for points, which is all it keeps.

## the pool

`build_pool.py` turns the 2.9 MB export into `pool.json` — 350 players, 13 fields each,
~100 KB — by throwing away everything a 10-team superflex dynasty draft can't use:

```
uv run pool_pipeline/build_pool.py                        # projections.json -> pool.json
uv run pool_pipeline/build_pool.py --report               # + validation summary on stderr
uv run pool_pipeline/build_pool.py --limit 450 -o big.json
```

Three cuts, in order: **positions** (QB/RB/WR/TE only — K and IDP have no roster slot,
446 players gone), **usability** (10 more carry a 0 or missing 3-year projection, which
isn't a rankable quantity), then the **rank limit** (top 350 of the remaining 444 — 290
picks plus a 60-player buffer; the cut lands at 170 points). Then 9 schemes × 4 horizons
collapse to two point columns (3-year and 1-year) and 8 ADP columns to one.

- **`points_3yr` is this league's scoring, copied not computed.** TEs from `ppr` (1.0/rec),
  everyone else from `half_ppr` (0.5/rec) — the published 1QB columns that already price
  those rates exactly. The algebraic route `half_ppr + (te_premium - ppr)` is an identity
  on raw points, but the source columns are pre-rounded integers, so evaluating it drifts
  ±1 on 32 of the 350 cells for nothing. 1QB family because points can't depend on roster
  format and that family is the consistent one (see the drift quirk below).
- **`points_1yr` is the same copy at the 1-year horizon.** It splits every player into
  the ranker's two value horizons: year 1, and years 2-3 (`points_3yr − points_1yr`).
  Lineups are fielded per season, so the starting lineup is solved per horizon against
  that horizon's own replacement levels — a 69-point injury year cannot ride into a
  starting slot on the strength of its years-2-3 rebound. Bench pricing splits the same
  way: year-1 bench value is insurance only (`INSURANCE_BASE` in `ranker/league.py`,
  the position's expected share of starter games missed — a player who cannot play this
  season cannot cover it), years 2-3 add growth (`DEPTH_BASE`) to that insurance, so a
  backloaded rookie still outranks a flat veteran with the same 3-year sum on the bench.
  A genuine 0 (a projected redshirt year) is kept as 0.
- **3D value is not carried at all.** It's a provider-scaled ordinal (best player pinned
  at 100, ~half the league negative) that bakes in someone else's roster assumptions and
  isn't in points, so it can't be differenced against a replacement level — which is all
  `rank_vor.py` does with the pool. Points are the only value input.
- **`adp` is the superflex ADP as an integer overall pick.** All four superflex scoring
  styles are identical here (ADP responds *only* to 1QB vs superflex), so there is no
  TE-premium signal to transfer and none is invented. The source's round.pick encoding is
  decoded — `2.03` is round 2 pick 3 of a *12-team* draft, not a decimal, and it sorts
  wrong if you treat it as one. Its deep tail is noise: 48 pool players sit past pick 540,
  so a four-digit pick means "effectively undrafted", not a real slot.
- **Rank is by `points_3yr` descending**, ties broken by the provider's dynasty rank, so
  the emitted rank is verifiable from the emitted column and the file references nothing
  it doesn't contain. Cutting on the provider's dynasty rank instead swaps 18 players at
  the boundary — all deep bench bodies below replacement, and no rookie the provider ranks
  inside 350; `--report` prints that diff.
- **Two source sentinels are cleaned up**: the 0-point projections above, and `bye_week:
  18` for unsigned players (real byes run weeks 5–14), which becomes `null`.

Dropped fields — stale printed ranks, `percent_low`/`percent_high`, `hidden_row`, analyst
comments, profile paths, the other horizons and schemes — are all still in
`projections.json`, which `build_pool.py` only reads. `--report` re-verifies every claim
above against the file it just wrote.

## sleeper ids

The league is hosted on Sleeper, and a roster, trade or pick over their API is expressed
in *their* player ids. The provider's `player_id` is a Draftsharks number that means
nothing there, so `match_sleeper.py` joins the pool to Sleeper's player dump and writes
`sleeper_id` back into `pool.json` next to `player_id`. Only the id is taken — name, team,
age and position are already in the pool, and a second disagreeing copy would just raise
the question of which to trust.

```
uv run pool_pipeline/fetch_sleeper.py            # manual download -> data/sleeper_players.json
uv run pool_pipeline/fetch_sleeper.py --force    # ignore the 24h re-download guard
uv run pool_pipeline/match_sleeper.py --report   # join + print every risky match and every miss
```

**The download is a separate, manual step and never runs as part of the pipeline.** It's
~14 MB, Sleeper's docs ask for at most one call per day, and an NFL roster doesn't change
because local projections were rebuilt. The dump is cached in `pool_pipeline/data/`
(gitignored) alongside a small committed `.meta.json` recording when it was pulled;
`match_sleeper.py` stamps that timestamp into `pool.json` and warns past 14 days. With no
dump present the stage warns and is skipped — `pool.json` is still complete apart from
`sleeper_id`. `--only sleeper` makes it an error instead.

**There is no shared key, so the join is by name**, in three tiers, each requiring exactly
one survivor — an ambiguous player is left `null`, never guessed. Position has to agree in
every tier (against `position` or `fantasy_positions`), which is what separates the two
Kenneth Walkers and the three Kyle Williamses; team codes are translated (`JAC`->`JAX`,
`LVR`->`LV`) and the provider's `UNS`/`RK` sentinels mean unsigned, so those players carry
no team constraint. Current run: **350/350 matched**.

- **full name — 322.** Sleeper's own `search_full_name` normalization (lowercase,
  alphanumerics only). Team and age aren't used to match here, only cross-checked:
  age agrees on 322/322, team on 321/322 — the exception is Brandon Aiyuk, whom
  Draftsharks lists as unsigned and Sleeper still has on SF.
- **name without the suffix — 21.** The suffix is editorial: Sleeper lists Michael Penix
  Jr. as "Michael Penix" and Kenneth Walker III as "Kenneth Walker".
- **last name + team — 7.** First names are editorial too, and this tier can't lean on
  them at all, so it demands a team match and an age within 2 years instead: Cam(eron)
  Ward, Cam(eron) Skattebo, Chig(oziem) Okonkwo, Kenny/Kenneth Gainwell, Matt(hew) Hibner,
  Mitch(ell) Tinsley, and Tank Dell, whose given name is Nathaniel. `--report` prints all
  seven — it's the tier where a wrong join would be plausible enough to slip through.

Re-running is idempotent (the field is dropped and re-derived), duplicate ids across two
pool players are a hard error, and `pool.json` grows a `sleeper` header block with the
fetch timestamp, the per-tier counts and any unmatched players.

**ADP is `null` for all 410 IDP players** (`DB`/`DL`/`LB`) — the html carries a literal `"N/A"` there, normalized to `null`. No IDP has an ADP; no offensive player is missing one.

**Only 443 of 900 players have an analysis comment.** That's real, not a parse gap — it matches the count of non-empty comment attributes in the raw html. The rest are `""`.

**Name is not a unique key.** There are two distinct Justin Jeffersons: the MIN WR (`player_id` 10586) and a CLE rookie LB (`36130`). All 900 `player_id`s are unique — always key on that.

**Teamless players use sentinel values.** 20 unsigned free agents (Diggs, Aiyuk, Hill, …) carry team `UNS`, and one carries `RK`. All 21 have `bye_week: 18`, which is a placeholder, not a real bye week.

**Zero is used where you might expect null.** 12 of 36 kickers have `0` for the 3/5/10-year horizons despite a real 1-year projection, and 45 non-kickers have a `0` 10-year projection. These are genuine zeros in the source, so filter on them explicitly rather than assuming a positive projection exists.

**Point projections drift slightly between the 1QB and superflex variants** even though point totals shouldn't depend on roster settings — 2,664 of 10,800 comparable pairs differ, median 4 points (1.4%), max 81. `te_premium` and `te_premium_superflex` are identical. This looks like provider-side noise; don't read signal into the gap.

**3D value is scaled per scheme and goes negative.** The best player in each scheme is pinned at 100, and roughly 500 of 900 players sit below zero. Values aren't meaningful as absolutes or comparable across schemes — only as a within-scheme ordering.

**Two fields are undocumented in the source page.** `percent_low` / `percent_high` (from `data-percent-low`/`-high`) have no legend and no JS reference anywhere in the file. `hidden_row` flags 204 rows that carried `class="hidden-row"`; they're all QB/RB/WR/TE, so it's likely a research-depth filter, but the page never says. Both are captured as-is rather than interpreted.

## draft pipeline

`draft_pipeline/fetch_draft.py` writes `draft.json` — the whole board, all 290 picks, every
time it runs. It is one stage, so the pipeline is just the script; there is no orchestrator
to wrap a single step. The draft is
[`1388293618208374784`](https://sleeper.com/draft/nfl/1388293618208374784), the trailing
number of the league's draft URL.

```
uv run draft_pipeline/fetch_draft.py               # -> draft.json
uv run draft_pipeline/fetch_draft.py --report      # + the board's validation summary
uv run draft_pipeline/fetch_draft.py --selftest    # check the board geometry offline
uv run draft_pipeline/fetch_draft.py --me someone  # whose picks get is_mine
```

**It is on demand and deliberately uncached** — the mirror image of `fetch_sleeper.py`,
which refuses to re-download inside 24h. The draft changes with every pick, the responses
total a few hundred KB, and a stale board is worse than none, so every run re-asks. Four
endpoints, because the picks alone don't say who owns what: `/draft/<id>` (teams, rounds,
snake type, reversal round, draft order), `/draft/<id>/picks`, `/draft/<id>/traded_picks`,
and `/league/<id>/users` for display names. Sleeper's only published limit is 1000 calls a
minute. A failure of the first three is fatal — half a board is worse than none, and a
missing `traded_picks` would silently misattribute picks — while the user list only supplies
names, so it degrades to a warning.

**`picks` is one array of all 290 entries, indexed by `pick_no`**, each with the same shape
and a `status` of `made` or `pending`. Made picks are Sleeper's own record; pending ones
carry `null` for the player and exist so the file answers *who picks next* and *when is my
next pick*. Header fields cover the rest: `on_the_clock`, `my_next_pick` (with `picks_away`),
`picks_made`/`picks_pending`, `slots` (the ten teams by draft slot), and the raw
`traded_picks`.

**`sleeper_id` is the join key back to `pool.json`.** Nothing here is joined to the pool —
that's the consumer's business, and this file stays a record of what Sleeper said. Sleeper's
own `name`/`position`/`team` come along so the file reads by eye, but they're informational;
the pool has the projection provider's copy. `--report` checks the join and names any drafted
player outside the pool (expect kickers and IDP late).

**Pending picks are derived, and the derivation is checked against reality every run.**
Sleeper reports a pick's slot, roster and user only once it's been made. A plain snake
alternates — odd rounds slot 1..10, even 10..1 — and a *reversal round* repeats the previous
round's order instead of flipping back, which inverts the parity from that round on. At
reversal round 3 that gives forward, reverse, reverse, forward, reverse, forward, …, which
puts slot 2 at 1.02, 2.09, 3.09, 4.02, 5.09, 6.02 … 28.02, 29.09 — the sequence above.
Traded picks are then applied on top, keyed by round and the pick's *original* roster.

For every made pick, the derived slot and owning roster are compared against the ones
Sleeper reported; disagreements are warned about on stderr and recorded in the output's
`board_derivation` block. That check strengthens with each pick and would catch a wrong
reversal rule or an unapplied trade before it misattributed the pending half of the board.

It can only exercise rounds that have actually been drafted, though — on pick 4 of 290
that's one round of 29, and none of the trade logic, since a board with no traded picks
can't test it. So **`--selftest` covers the rest offline** (28 checks, no network): the slot
order of each supported format, this league's slot-2 sequence against the one stated at the
top of this file, a traded pick landing with its acquirer, and the negative control that an
*un*applied trade is caught by the live check above.

`--report` prints the round-by-round slot order, that live cross-check, my full pick list,
the last dozen selections with a position breakdown, the pool join, and integrity checks
(`pick_no` gap-free, every slot appearing exactly `rounds` times, no player drafted twice,
every pick owned).

## rankings from the live board

`rank_vor.py` reads both files. `draft.json` is the simulation's **starting state**, not a
filter applied afterwards: made picks sit on their teams' rosters, the pending picks are the
only ones simulated, and they are played in the order that file gives — so a traded pick is
exercised by the roster that acquired it. `rankings.json` then covers the **undrafted
players only**, ranked over each other.

```
uv run rank_vor.py                      # pool.json + draft.json -> rankings.json
uv run rank_vor.py --report             # + the board it started from, per team
uv run rank_vor.py --no-draft           # ignore draft.json: rank the whole pool
uv run rank_vor.py --draft other.json   # a different board
uv run rank_vor.py --selftest           # lineup solver + board loader, offline
```

During a draft the two live steps are always run together, so `refresh.py` does exactly
that and nothing else — `fetch_draft.py`, then `rank_vor.py`, stopping on the first
failure. It shells out to each with `uv run` rather than importing them, keeping the
pipelines as separate as they are everywhere else, and runs them from the repo root so
their own default paths apply.

```
uv run refresh.py                       # draft.json, then rankings.json  (~18 min on 2 cores)
uv run refresh.py --report              # + both steps' validation summaries on stderr
```

The ranker's Monte Carlo, rollout and candidate-survival stages fan out over a process
pool (stdlib `multiprocessing`), so wall time scales with cores — the 2-core Codespace
is the slow case. To trade fidelity for speed between picks, the levers are `--sims`
(which also sizes the candidate-survival redraws) and `ROLLOUT_SIMS`/`SIMS` in
`ranker/league.py`.

The pool pipeline is not a step: it is offline and rebuilt a handful of times all
offseason, and this runs every few minutes during a draft.

`index.html` is a read-only dashboard view of `rankings.json`, ordered around the decisions
at the table: comparison tables for my next three picks, my projected final team, best
available, then the other nine projected teams. Player comparisons show the raw Year 1
projection, the Years 2–3 annual pace, their difference as the implied trend, and the
model's roster-aware scores. The first pick also surfaces the candidate-specific
`p_available_if_i_pass` redraws and full-draft rollout; best available uses the broader
Kaplan–Meier availability estimate and deliberately omits the redundant deterministic
pick / likely-through columns.

The projected-team entries in `example_draft.rosters[].picks` are structured records,
not display strings. Each includes pick status, player identity, age/team/rookie metadata,
`points_1yr`, `points_3yr`, `future_points_per_year`, and `growth_per_year`; this is
necessary because the headline `rankings` rows contain undrafted players only, while a
complete roster also contains live picks already removed from that list. The dashboard
uses those records to solve each team's optimal Year 1 and Years 2–3 starting lineup and
compare annual starter points. These are deterministic final-roster projections, not an
average over possible final teams.

The dashboard remains one static file with no build step or dependencies; it fetches
`rankings.json` from the same directory on page load, so the loop is `refresh.py`, then
reload the browser. It also polls the Sleeper draft endpoints directly (the same public
API the draft pipeline reads, every 30s) for a compact live strip — picks made, current
draft spot, my next held pick, and the last selection — and compares the live pick count
and status against the snapshot the board was built from, highlighting the refresh button
when the board has fallen behind. My-pick calculations use `draft.my_remaining_picks`, not
the original slot sequence in `league.my_picks`, because this draft contains traded picks.
In the Codespace the devcontainer already serves the repo root on port 8000 at startup
(`.devcontainer/devcontainer.json`); elsewhere, serve it with `python3 -m http.server 8123`
and open the forwarded port (a direct `file://` open is blocked by the
browser's fetch rules, and the page says so).

It is also hosted: `.github/workflows/deploy-pages.yml` publishes `index.html` and
`rankings.json` to GitHub Pages on every push to `main` that touches either file.
`.github/workflows/refresh.yml` runs the same `refresh.py` loop in Actions on a manual
dispatch and commits the resulting `draft.json` and `rankings.json`. Its push uses the
default `GITHUB_TOKEN`, which does not trigger other workflows, so when it pushed a
commit it dispatches the Pages deploy itself. Both workflows share one concurrency group
(`rankings-pipeline`, with `queue: max`), so a refresh and a deploy never run at the same
time — later runs queue behind instead of cancelling. The dashboard's "↻ Refresh board" button
dispatches that workflow from the page via the GitHub API; it asks once for a
fine-grained PAT (Actions read/write on this repo) and keeps it in localStorage.

The method itself is unchanged, and that is checked rather than asserted: a `draft.json`
with nothing drafted yet reproduces `--no-draft` exactly, byte for byte. Replacement levels
are still measured league-wide over whole final rosters and against the whole pool — per
horizon, since year-1 and years-2-3 lineups are priced separately — so the marginal starter
defining a level can be a player already drafted. What moves is the simulated *shape* of
the league: after a run on a position, the fixed point settles on different starter counts
than the preseason board's.

**A pick can land on a player the pool does not carry** — a kicker, an IDP, anyone past the
350-player cut. There is no projection to price him with, so he is held as an `off_pool`
roster entry: he fills a spot, so his team owes one fewer pick, and he answers a mandatory
position, so that team is not made to draft another QB. He never starts and is never worth
anything. `--report` names every one of them.

**Made picks are facts, never re-valued.** If a team reached, the board takes it as given
and prices what is left.

An absent `draft.json` is the preseason case, not an error: the script says so and ranks the
whole pool. A board that disagrees with the league constants in `ranker/league.py` (12
teams, a player drafted twice, a header contradicting its own picks, a pool with no
`sleeper_id`s to join on) is reported in `validation.problems` and exits non-zero rather
than being quietly absorbed. `--selftest` covers what the live file cannot: a traded pick, a
selection outside the pool, resuming a partial board, and six malformed boards.

the projections.html are tables from a projections/rankings provider (draftsharks).

projections.html contains projections and rankings from a provider (draftsharks). it's an html table with rows like this for each player (brock bowers is the example): data-value="2191" data-scoring-value="1609" data-scoring-value-half-ppr="2191" data-scoring-value-ppr="2772" data-scoring-value-te-premium="3354" data-scoring-value-superflex="1609" data-scoring-value-half-ppr-superflex="2191" data-scoring-value-ppr-superflex="2772" data-scoring-value-te-premium-superflex="3354". its "3D value" assumes a superflex league and .5 ppr (unclear exactly what roster settings).

My league is .5 ppr with a .5 ppr tight end premium. Our rosters are 1 QB, 2 RB, 3 WR, 1 TE, 2 WRT, 1 WRTQ (superflex), with 15 man benches, 3 IR slots, and 4 rookie year taxi spots. The league has 10 teams (290 drafted players). We're doing a snake draft with a 3rd round reversal. My draft slot is 1.02, so my picks are: 1.02, 2.9, 3.9, 4.2, 5.9, 6.2 ... 28.2, 29.9.

## layout

The build lives in `pipeline/` and publishes exactly one file: `pool.json` at the root.
Everything it reads and every intermediate it writes stays inside the folder.

```
pool.json                    the draft pool — the only output, 350 players, 12 fields
rank_vor.py                  pool.json -> rankings.json (run separately)
pipeline/
  pipeline.py                orchestrator: parse -> pool -> sleeper
  parse_projections.py       1. html -> projections.json   (faithful provider export)
  build_pool.py              2. projections.json -> pool.json  (this league's pool)
  match_sleeper.py           3. pool.json -> pool.json + sleeper_id
  fetch_sleeper.py           manual: download Sleeper's player dump (not a stage)
  data/
    projections.html         raw provider save, added by hand
    projections.json         intermediate, 900 players x 8 schemes
    sleeper_players.json     cached Sleeper dump, ~14 MB, gitignored
    sleeper_players.meta.json  when it was fetched (committed)
```

## pipeline

```
python3 pipeline/pipeline.py                  # html -> projections.json -> pool.json + ids
python3 pipeline/pipeline.py --report         # + every stage's validation summary on stderr
python3 pipeline/pipeline.py --only pool      # single stage
python3 pipeline/fetch_sleeper.py             # refresh the Sleeper dump (manual, ~14 MB)
python3 rank_vor.py                           # pool.json -> rankings.json (run separately)
```

Three data stages, ordered: `parse_projections.py` (html -> `projections.json`, the full
provider export: 900 players, 8 schemes, 4 horizons, ~2.9 MB), `build_pool.py`
(-> `pool.json`, this league's 350-player draft pool with one value column, ~95 KB), then
`match_sleeper.py` (adds `sleeper_id` to that file in place). All are standalone CLIs; the
orchestrator only fixes the order and stops on the first failure. Stage 1 stays faithful to
what the provider published and is never narrowed; stage 2 is the narrow, draft-ready view,
so it can be re-run at a different rank limit without re-parsing 8 MB of html and the
dropped columns stay recoverable. Stage 3 comes last because stage 2 rewrites `pool.json`
from scratch, dropping the ids stage 3 adds — so a rebuild always re-joins them.

Default paths are anchored to the scripts, not the shell, so every command above works
from the repo root or from inside `pipeline/`.

`rank_vor.py` (pool.json -> rankings.json) is deliberately not a pipeline stage: it takes
a simulation seed and strategy knobs and is re-run far more often than the data is built.

## parsing

`parse_projections.py` turns `data/projections.html` into `data/projections.json` — 900 players, ~2.5 MB, about 1.5s. Python stdlib only (`html.parser`), no dependencies.

```
python3 pipeline/parse_projections.py                      # -> data/projections.json
python3 pipeline/parse_projections.py --report             # + validation summary on stderr
python3 pipeline/parse_projections.py in.html -o out.json
```

No browser or JS execution is needed: the page is server-rendered and **every scoring scheme is already in the DOM** as `data-scoring-value-*` attributes on each cell. Alpine.js only toggles which one is visible. So all 8 schemes (`standard`, `half_ppr`, `ppr`, `te_premium`, each also `_superflex`) come straight out of the static markup.

Each player gets identity fields (id, name, position, team, age, bye, rookie flag, tiers, profile path), the analyst comment, and per-scheme `three_d_value`, `adp`, and `projections` for the 1/3/5/10-year horizons. `half_ppr_superflex_3d_value` is lifted to the top level of each record.

## non-obvious quirks in the parsed output

**`data-value` is the .5 ppr superflex value at full precision.** The visible cell rounds it (Bijan Robinson displays `75`, the attribute says `75.1`). Kept as `<scheme>_precise` inside each scheme dict and as the top-level `half_ppr_superflex_3d_value`. Checking `data-value` against `data-scoring-value-half-ppr-superflex` across all 900 rows × 6 numeric columns gave 0 disagreements, which is also what confirms .5 ppr sf is the rendered default.

**The ranks baked into the html are stale — don't use them.** This file was saved from an already-hydrated page, and the printed rank numbers no longer match the values sitting next to them: 62 overall ranks are duplicated, 62 are missing, and document order jumps backwards once (doc position 417→418, rank 424→382). Document order is otherwise clean, descending by .5 ppr sf 3D value. The parser keeps the printed numbers verbatim as `rank_displayed` / `positional_rank_displayed` and adds derived `rank_by_3d_value` / `positional_rank_by_3d_value` (unique, gap-free, ties broken by document order). **Use the derived ones for a draft board.** `document_index` preserves the original row order.

**`te_premium` is not this league's scoring.** Draftsharks' `te_premium` is full ppr *plus* a 0.5/rec premium, i.e. 1.5 pts/rec for TEs, and is byte-identical to `ppr` for every non-TE. (Verified: the standard → half_ppr → ppr → te_premium steps are a uniform 0.5/rec on all 89 TEs, and te_premium == ppr for all 305 WR/RB.) This league is 0.5 base + 0.5 premium = 1.0/rec for TEs and 0.5/rec for WR/RB. So for raw points there is no single matching column — it's `half_ppr` for non-TEs and `ppr` for TEs. For 3D value there's no exact column either: `half_ppr_superflex` gives TEs no premium at all, while `te_premium_superflex` has the correct *relative* TE edge (+0.5/rec over WR/RB) but builds it on an inflated 1.0 reception base that also lifts every WR. That gap is what `build_pool.py` closes — for points, which is all it keeps.

## the pool

`build_pool.py` turns the 2.9 MB export into `pool.json` — 350 players, 12 fields each,
~100 KB — by throwing away everything a 10-team superflex dynasty draft can't use:

```
python3 pipeline/build_pool.py                       # projections.json -> pool.json
python3 pipeline/build_pool.py --report              # + validation summary on stderr
python3 pipeline/build_pool.py --limit 450 -o big.json
```

Three cuts, in order: **positions** (QB/RB/WR/TE only — K and IDP have no roster slot,
446 players gone), **usability** (10 more carry a 0 or missing 3-year projection, which
isn't a rankable quantity), then the **rank limit** (top 350 of the remaining 444 — 290
picks plus a 60-player buffer; the cut lands at 170 points). Then 9 schemes × 4 horizons
collapse to one value column and 8 ADP columns to one.

- **`points_3yr` is this league's scoring, copied not computed.** TEs from `ppr` (1.0/rec),
  everyone else from `half_ppr` (0.5/rec) — the published 1QB columns that already price
  those rates exactly. The algebraic route `half_ppr + (te_premium - ppr)` is an identity
  on raw points, but the source columns are pre-rounded integers, so evaluating it drifts
  ±1 on 32 of the 350 cells for nothing. 1QB family because points can't depend on roster
  format and that family is the consistent one (see the drift quirk below).
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
python3 pipeline/fetch_sleeper.py           # manual download -> data/sleeper_players.json
python3 pipeline/fetch_sleeper.py --force   # ignore the 24h re-download guard
python3 pipeline/match_sleeper.py --report  # join + print every risky match and every miss
```

**The download is a separate, manual step and never runs as part of the pipeline.** It's
~14 MB, Sleeper's docs ask for at most one call per day, and an NFL roster doesn't change
because local projections were rebuilt. The dump is cached in `pipeline/data/`
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
the projections.html are tables from a projections/rankings provider (draftsharks).

projections.html contains projections and rankings from a provider (draftsharks). it's an html table with rows like this for each player (brock bowers is the example): data-value="2191" data-scoring-value="1609" data-scoring-value-half-ppr="2191" data-scoring-value-ppr="2772" data-scoring-value-te-premium="3354" data-scoring-value-superflex="1609" data-scoring-value-half-ppr-superflex="2191" data-scoring-value-ppr-superflex="2772" data-scoring-value-te-premium-superflex="3354". its "3D value" assumes a superflex league and .5 ppr (unclear exactly what roster settings).

My league is .5 ppr with a .5 ppr tight end premium. Our rosters are 1 QB, 2 RB, 3 WR, 1 TE, 2 WRT, 1 WRTQ (superflex), with 15 man benches, 3 IR slots, and 4 rookie year taxi spots. The league has 10 teams (290 drafted players).

## parsing

`parse_projections.py` turns `projections.html` into `projections.json` — 900 players, ~2.5 MB, about 1.5s. Python stdlib only (`html.parser`), no dependencies.

```
python3 parse_projections.py                      # -> projections.json
python3 parse_projections.py --report             # + validation summary on stderr
python3 parse_projections.py in.html -o out.json
```

No browser or JS execution is needed: the page is server-rendered and **every scoring scheme is already in the DOM** as `data-scoring-value-*` attributes on each cell. Alpine.js only toggles which one is visible. So all 8 schemes (`standard`, `half_ppr`, `ppr`, `te_premium`, each also `_superflex`) come straight out of the static markup.

Each player gets identity fields (id, name, position, team, age, bye, rookie flag, tiers, profile path), the analyst comment, and per-scheme `three_d_value`, `adp`, and `projections` for the 1/3/5/10-year horizons. `half_ppr_superflex_3d_value` is lifted to the top level of each record.

## non-obvious quirks in the parsed output

**`data-value` is the .5 ppr superflex value at full precision.** The visible cell rounds it (Bijan Robinson displays `75`, the attribute says `75.1`). Kept as `<scheme>_precise` inside each scheme dict and as the top-level `half_ppr_superflex_3d_value`. Checking `data-value` against `data-scoring-value-half-ppr-superflex` across all 900 rows × 6 numeric columns gave 0 disagreements, which is also what confirms .5 ppr sf is the rendered default.

**The ranks baked into the html are stale — don't use them.** This file was saved from an already-hydrated page, and the printed rank numbers no longer match the values sitting next to them: 62 overall ranks are duplicated, 62 are missing, and document order jumps backwards once (doc position 417→418, rank 424→382). Document order is otherwise clean, descending by .5 ppr sf 3D value. The parser keeps the printed numbers verbatim as `rank_displayed` / `positional_rank_displayed` and adds derived `rank_by_3d_value` / `positional_rank_by_3d_value` (unique, gap-free, ties broken by document order). **Use the derived ones for a draft board.** `document_index` preserves the original row order.

**`te_premium` is not this league's scoring.** Draftsharks' `te_premium` is full ppr *plus* a 0.5/rec premium, i.e. 1.5 pts/rec for TEs, and is byte-identical to `ppr` for every non-TE. (Verified: the standard → half_ppr → ppr → te_premium steps are a uniform 0.5/rec on all 89 TEs, and te_premium == ppr for all 305 WR/RB.) This league is 0.5 base + 0.5 premium = 1.0/rec for TEs and 0.5/rec for WR/RB. So for raw points there is no single matching column — it's `half_ppr` for non-TEs and `ppr` for TEs. For 3D value there's no exact column either: `half_ppr_superflex` gives TEs no premium at all, while `te_premium_superflex` has the correct *relative* TE edge (+0.5/rec over WR/RB) but builds it on an inflated 1.0 reception base that also lifts every WR. Worth bracketing TE decisions with both rather than trusting one.

**ADP is `null` for all 410 IDP players** (`DB`/`DL`/`LB`) — the html carries a literal `"N/A"` there, normalized to `null`. No IDP has an ADP; no offensive player is missing one.

**Only 443 of 900 players have an analysis comment.** That's real, not a parse gap — it matches the count of non-empty comment attributes in the raw html. The rest are `""`.

**Name is not a unique key.** There are two distinct Justin Jeffersons: the MIN WR (`player_id` 10586) and a CLE rookie LB (`36130`). All 900 `player_id`s are unique — always key on that.

**Teamless players use sentinel values.** 20 unsigned free agents (Diggs, Aiyuk, Hill, …) carry team `UNS`, and one carries `RK`. All 21 have `bye_week: 18`, which is a placeholder, not a real bye week.

**Zero is used where you might expect null.** 12 of 36 kickers have `0` for the 3/5/10-year horizons despite a real 1-year projection, and 45 non-kickers have a `0` 10-year projection. These are genuine zeros in the source, so filter on them explicitly rather than assuming a positive projection exists.

**Point projections drift slightly between the 1QB and superflex variants** even though point totals shouldn't depend on roster settings — 2,664 of 10,800 comparable pairs differ, median 4 points (1.4%), max 81. `te_premium` and `te_premium_superflex` are identical. This looks like provider-side noise; don't read signal into the gap.

**3D value is scaled per scheme and goes negative.** The best player in each scheme is pinned at 100, and roughly 500 of 900 players sit below zero. Values aren't meaningful as absolutes or comparable across schemes — only as a within-scheme ordering.

**Two fields are undocumented in the source page.** `percent_low` / `percent_high` (from `data-percent-low`/`-high`) have no legend and no JS reference anywhere in the file. `hidden_row` flags 204 rows that carried `class="hidden-row"`; they're all QB/RB/WR/TE, so it's likely a research-depth filter, but the page never says. Both are captured as-is rather than interpreted.
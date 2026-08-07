# Ranker

`rank_vor.py` consumes `pool.json`, `draft.json`, normalized provider boards, and
`data_source_matches.json`, then publishes `rankings.json`.

```bash
uv run rank_vor.py
uv run rank_vor.py --report
uv run rank_vor.py --no-draft
uv run rank_vor.py --draft other.json
uv run rank_vor.py --selftest
```

A live board is the simulation's starting state, not a post-processing filter. Made picks
remain on their rosters, only pending picks are played, and output ranking rows contain
undrafted players only.

## Modules

- `league.py`: league shape and hardcoded strategy constants
- `pool.py`: pool document to `Player` objects
- `board.py`: live `draft.json` to the immutable starting state
- `opponents.py`: inferred provider boards to complete opponent strategies
- `value.py`: horizon points, lineup value, depth value, and replacement measurement
- `simulation.py`: one deterministic draft state and pick policies
- `convergence.py`: replacement-level fixed point
- `planning.py`: Monte Carlo availability, candidate survival, lookahead, and rollouts
- `rankings.py`: ranking rows and serialized next-pick recommendations
- `output.py`: top-level `rankings.json` payload
- `report.py`: human-readable stderr diagnostics
- `validate.py`: every-run output and league invariants
- `selftest.py`: solver, opponent-separation, planning, and malformed-board checks

## Value model

The pool is split into year 1 and years 2–3 because a lineup is fielded each season.
Replacement is solved separately for both horizons. At each horizon it is the next player
after all simulated starters at that position; flexible starting slots determine their
position mix from actual roster value rather than a hardcoded positional count.

This creates a fixed point:

```text
replacement -> player/team value -> simulated draft -> starter counts -> replacement
```

The map is discrete and can alternate between neighboring league shapes, so convergence
detects a repeated state and averages levels over that cycle. Bench depth is separately
priced above the final waiver-wire level, with year-one insurance and years-2–3 growth.

## Opponents and planning

My slot alone uses projections and roster value. Each opponent uses the provider board
closest to its completed picks, with a soft boost for unfilled dedicated starters and a
compounding source-rank penalty for adding players beyond comfortable positional depth.
The depth targets sum to 25, so the last four spots remain source-driven rather than
forcing every opponent into one exact roster shape. These are preferences, not draft
limits: a large enough source-rank gap can still justify another player at a deep position.
Observed `mean_log2_loss` calibrates randomness around that preference. Missing provider
players are appended in DraftSharks ADP order; opponents never fall back to VOR.

My simulated pick policy applies the same depth penalty to marginal roster gain. It does
not change projected roster value; it breaks late-draft ties among small positive gains
so a thin position is not routinely crowded out by a ninth quarterback or tight end.

The bulk deterministic policy scores value now plus the expected best option at its next
pick. The live decision broadens the candidate set, redraws intervening opponents, measures
candidate survival when I keep passing, proposes target plans across four held picks, and
plays finalists to the end of the draft. Worker processes receive immutable inputs once,
and seeded task ids keep results deterministic across scheduling.

## Output contract

`rankings.json.rankings` contains VOR, horizon splits, projected and simulated pick
fields, opponent consensus deltas, and availability estimates for each undrafted player.
`my_next_picks` is the roster-aware recommendation and can intentionally disagree with
headline VOR. `example_draft` contains structured final-roster records for dashboard
lineup comparison. `validation.problems` is empty on success; any problem makes the CLI
exit nonzero.

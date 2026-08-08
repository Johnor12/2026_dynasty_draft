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
- `opponents.py`: inferred provider boards to source-implied point projections
- `value.py`: horizon points, expected lineup value, and replacement measurement
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

The final waiver depth affects every team's expected-lineup choices, and those choices
affect who remains undrafted. Opponents see the same positional depth translated into
their implied projections. This creates a fixed point:

```text
wire levels -> expected lineup value -> simulated draft -> starter counts + wire levels
```

The map is discrete and can alternate between neighboring league shapes, so convergence
detects a repeated state and averages levels over that cycle. For each legal positional
composition, the expected-lineup solver orders players by their points when active. A
deeper player contributes with the exact probability that too few higher teammates are
available. One always-available waiver body can fill one job at each position; it is not
an unlimited scalar. The best composition wins, making value monotone when a projection
improves or a player is added. Years 2–3 use their own projections and lineup, with no
second growth bonus.

## Opponents and planning

Every team uses the same expected-lineup objective and the same backup-point attribution.
Positional depth is priced by projected points and position-wide unavailability, with one
unique waiver fallback; there are no opponent starter boosts, depth targets, or positional
roster-size heuristics.

The difference is information and search depth. Each opponent uses the provider board
closest to its completed picks. Source rank `r` receives the `r`-th value on our
projection-backed VOR curve, then that player's positional replacement level is added to
express the value as a three-year point total. This borrows point units, not player
opinions: the external order decides which player receives each value. DraftSharks'
`points_1yr` pace versus its years 2–3 annual pace classifies the player as front-loaded,
balanced, or back-loaded (within 10% counts as balanced). The median year-1 share of that
class splits the implied total, so players retain the provider's timeline signal without
an LLM classification or a unique noisy split for every player.
Opponents maximize immediate roster EV under those implied projections (level 0). My slot
uses our projections, one-pick lookahead in the bulk simulation, and deeper planning for
the live decision. All final projections, VOR, replacement levels, and roster reporting
use our projections; opponent-implied points affect their simulated choices only.

Observed `mean_log2_loss` calibrates random mistakes around each opponent's level-0 EV
order. Draws remain among players not dominated at their position in both horizons: this
models an intentionally strong field instead of reproducing irrational reaches. Missing
provider players are appended in DraftSharks ADP order before ranks are translated, so
every mandatory pick remains possible.

The bulk deterministic policy scores value now plus the expected best option at its next
pick. The live shortlist starts from the current board before intervening opponents pick,
then removes candidates below 5% survival to my turn. Candidate branches are evaluated
conditional on reaching that turn. Four-pick planning applies the same 5% floor to each
later target's conditional survival before playing finalists to the end of the draft.
The first `take` is the EV recommendation if available; when the noiseless example has
already removed it, `deterministic_fallback` identifies the example draft's legal choice.
Worker processes receive immutable inputs once, and seeded task ids keep results
deterministic across scheduling.

## Output contract

`rankings.json.rankings` contains VOR, horizon splits, projected and simulated pick
fields, opponent consensus ranks and implied points, our projection edge against that
consensus, and availability estimates for each undrafted player.
`my_next_picks` is the roster-aware recommendation and can intentionally disagree with
headline VOR. `example_draft` contains structured final-roster records for dashboard
lineup comparison. `validation.problems` is empty on success; any problem makes the CLI
exit nonzero.

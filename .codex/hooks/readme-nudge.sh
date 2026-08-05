#!/usr/bin/env bash
# When files changed but README.md did not, ask Codex to check the docs once.

input=$(cat)
if command -v jq >/dev/null 2>&1 &&
   [ "$(jq -r '.stop_hook_active // false' <<<"$input")" = true ]; then
  exit 0
fi

changed=$(git status --porcelain 2>/dev/null)
[ -n "$changed" ] || exit 0
[[ "$changed" == *"README.md"* ]] && exit 0

cat <<'JSON'
{"decision":"block","reason":"Files changed this turn but README.md did not. Check whether README.md still describes the affected pipeline layout and stages, file contracts, how scripts are run, and league or scoring assumptions. Update the affected sections if they are stale. If the README is unaffected, say so briefly and finish."}
JSON

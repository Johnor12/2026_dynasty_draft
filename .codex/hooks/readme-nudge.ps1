# When files changed but README.md did not, ask Codex to check the docs once.
$inputJson = [Console]::In.ReadToEnd() | ConvertFrom-Json

if ($inputJson.stop_hook_active) {
    exit 0
}

$changed = git status --porcelain 2>$null
if (-not $changed) {
    exit 0
}

if ($changed -match 'README\.md$') {
    exit 0
}

@{
    decision = 'block'
    reason = 'Files changed this turn but README.md did not. Check whether README.md still describes the affected pipeline layout and stages, file contracts, how scripts are run, and league or scoring assumptions. Update the affected sections if they are stale. If the README is unaffected, say so briefly and finish.'
} | ConvertTo-Json -Compress

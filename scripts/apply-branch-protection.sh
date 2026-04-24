#!/usr/bin/env bash
# Apply the main-branch ruleset to the remote GitHub repository.
#
# Preconditions:
#   1. The repo is pushed to GitHub (a remote named `origin` exists and
#      points to github.com).
#   2. `gh` is authenticated with admin rights on the repo.
#   3. The test workflow has run at least once so the status-check
#      contexts in .github/rulesets/protect-main.json are recognised.
#
# Usage:
#   scripts/apply-branch-protection.sh                       # creates
#   scripts/apply-branch-protection.sh --update RULESET_ID   # updates
#
# List existing rulesets to find a RULESET_ID:
#   gh api repos/:owner/:repo/rulesets

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
RULESET_JSON="${REPO_ROOT}/.github/rulesets/protect-main.json"

if [ ! -f "${RULESET_JSON}" ]; then
    echo "error: ruleset file not found at ${RULESET_JSON}" >&2
    exit 1
fi

remote_url="$(git -C "${REPO_ROOT}" remote get-url origin 2>/dev/null || true)"
if [ -z "${remote_url}" ]; then
    echo "error: no 'origin' remote configured; push the repo to GitHub first" >&2
    exit 1
fi

# Parse owner/repo from either https or ssh URL.
if [[ "${remote_url}" =~ github\.com[:/]+([^/]+)/([^/.]+)(\.git)?$ ]]; then
    owner="${BASH_REMATCH[1]}"
    repo="${BASH_REMATCH[2]}"
else
    echo "error: could not parse github.com owner/repo from '${remote_url}'" >&2
    exit 1
fi

endpoint="repos/${owner}/${repo}/rulesets"

case "${1:-create}" in
    create)
        echo "Creating main-branch ruleset on ${owner}/${repo}..."
        gh api \
            --method POST \
            -H "Accept: application/vnd.github+json" \
            "${endpoint}" \
            --input "${RULESET_JSON}"
        ;;
    --update)
        if [ -z "${2:-}" ]; then
            echo "error: --update requires a ruleset id" >&2
            echo "list existing: gh api ${endpoint}" >&2
            exit 1
        fi
        ruleset_id="$2"
        echo "Updating ruleset ${ruleset_id} on ${owner}/${repo}..."
        gh api \
            --method PUT \
            -H "Accept: application/vnd.github+json" \
            "${endpoint}/${ruleset_id}" \
            --input "${RULESET_JSON}"
        ;;
    *)
        echo "usage: $0 [create|--update RULESET_ID]" >&2
        exit 64
        ;;
esac

echo "Done. Verify: gh api ${endpoint}"

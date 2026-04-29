#!/usr/bin/env bash
# Release coordination for the INLA.jl monorepo.
#
# JuliaRegistrator is comment-driven, not script-driven, so this file
# is documentation-as-code: it prints the submission order and the
# exact comment to post on the v-tagged commit for each package.
# Run it, follow the prompts, copy the comment text into a GitHub
# comment on the appropriate commit. Each step waits for AutoMerge to
# land the previous package in General.
#
# Pre-flight (run once before the first submission):
#   1. `git status` clean on `main`; `main` pushed to origin.
#   2. Every core package's `Project.toml` `version =` matches the
#      tag you intend to register. CI green on the tagged commit.
#   3. `gh auth status` clean (Registrator works via the GitHub App,
#      but `gh` is needed for the verification step).
#   4. `[compat]` bounds set for every direct dep (Registrator's
#      AutoMerge requires this); confirm with
#      `julia --project=packages/<Pkg>.jl -e 'using Pkg; Pkg.status()'`.
#
# Usage:
#   scripts/release.sh                # prints the four-step plan
#   scripts/release.sh verify <pkg>   # post-merge sanity check on
#                                     # General-registered <pkg>

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"

readonly RELEASE_ORDER=(
    "GMRFs"
    "LatentGaussianModels"
    "INLASPDE"
    "INLA"
)

print_plan() {
    cat <<'EOF'
INLA.jl General-registry submission plan
========================================

Submit packages serially, lowest in the dep graph first. Wait for
AutoMerge on each before submitting the next. AutoMerge is typically
~3 days unless tripped by a compat or naming issue.

  1. GMRFs                — leaf; no internal deps.
  2. LatentGaussianModels — depends on GMRFs.
  3. INLASPDE             — depends on GMRFs.
  4. INLA (umbrella)      — depends on GMRFs, LatentGaussianModels, INLASPDE.

For each package, post the following as a GitHub comment on the
release commit (typically the tip of `main` at tag time):

EOF
    for pkg in "${RELEASE_ORDER[@]}"; do
        echo "  @JuliaRegistrator register subdir=packages/${pkg}.jl"
    done
    cat <<'EOF'

Optional sub-packages (LGMFormula, LGMTuring, GMRFsPardiso,
INLASPDERasters) stay on the personal registry until each ships its
own tagged v0.1; do not register from this monorepo path.

If AutoMerge trips: read the bot's comment, fix the issue (most
commonly a too-narrow [compat] bound), commit, retag, and re-post the
register comment. Do NOT continue down the list until the failing
package has merged in General.

After all four merge, tag a top-level monorepo release:
    git tag v$(grep -E '^version' packages/INLA.jl/Project.toml | cut -d'"' -f2) \
        && git push origin --tags
EOF
}

verify_pkg() {
    local pkg="$1"
    echo "Verifying ${pkg} in General registry..."
    local depot
    depot="$(mktemp -d)"
    JULIA_DEPOT_PATH="${depot}" julia --startup-file=no -e \
        "using Pkg; Pkg.add(\"${pkg}\"); using ${pkg}; @info \"${pkg} loaded OK\""
    rm -rf "${depot}"
}

case "${1:-plan}" in
    plan)        print_plan ;;
    verify)
        if [ -z "${2:-}" ]; then
            echo "error: 'verify' requires a package name" >&2
            exit 64
        fi
        verify_pkg "$2"
        ;;
    *)
        echo "usage: $0 [plan|verify <package>]" >&2
        exit 64
        ;;
esac

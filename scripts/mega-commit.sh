#!/usr/bin/env bash

# Commit and push the main RISE-2026 repository and its sibling repositories

# using one commit message.

#

# Repositories:

# ~/RISE-2026

# ~/RISE-2026/SpecRLBench

# ~/RISE-2026/GenZ-LTL

# ~/RISE-2026/Safe-Policy-Optimization

#

# Usage:

# bash scripts/mega-commit.sh "commit message"

# bash scripts/mega-commit.sh --dry-run "commit message"

set -euo pipefail

MESSAGE=""
DRY_RUN=0

RISE_ROOT="$HOME/RISE-2026"

REPO_NAMES=(
"SpecRLBench"
"GenZ-LTL"
"Safe-Policy-Optimization"
)

REPO_BRANCHES=(
"feat/sar-envs"
"feat/sar-props"
"feat/specrlbench-additions"
)

usage() {
cat <<'EOF'
Usage:
bash scripts/mega-commit.sh "commit message"
bash scripts/mega-commit.sh --dry-run "commit message"
bash scripts/mega-commit.sh -n "commit message"
EOF
exit 1
}

while [[ $# -gt 0 ]]; do
case "$1" in
--dry-run|-n)
DRY_RUN=1
shift
;;
-h|--help)
usage
;;
-*)
echo "Unknown option: $1" >&2
usage
;;
*)
if [[ -z "$MESSAGE" ]]; then
MESSAGE="$1"
else
echo "Unexpected argument: $1" >&2
usage
fi
shift
;;
esac
done

[[ -n "$MESSAGE" ]] || usage

write_step() {
printf '\n=== %s ===\n' "$1"
}

is_git_repo() {
local path="$1"
git -C "$path" rev-parse --is-inside-work-tree >/dev/null 2>&1
}

current_branch() {
local path="$1"
git -C "$path" branch --show-current
}

verify_repo() {
local path="$1"
local name="$2"
local expected_branch="$3"
local branch


if [[ ! -d "$path" ]]; then
    echo "[$name] Directory does not exist: $path" >&2
    return 1
fi

if ! is_git_repo "$path"; then
    echo "[$name] Not a Git repository: $path" >&2
    return 1
fi

branch="$(current_branch "$path")"

if [[ -z "$branch" ]]; then
    echo "[$name] Detached HEAD; refusing to continue." >&2
    return 1
fi

if [[ -n "$expected_branch" && "$branch" != "$expected_branch" ]]; then
    echo "[$name] Branch mismatch:" >&2
    echo "  Current:  $branch" >&2
    echo "  Expected: $expected_branch" >&2
    return 1
fi


}

commit_if_dirty() {
local path="$1"
local name="$2"
local expected_branch="$3"


verify_repo "$path" "$name" "$expected_branch"

if [[ -z "$(git -C "$path" status --porcelain)" ]]; then
    echo "[$name] No changes to commit."
    return 0
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[$name] Would commit with message: $MESSAGE"
    git -C "$path" status --short | sed 's/^/    /'
    return 0
fi

git -C "$path" add -A
git -C "$path" commit -m "$MESSAGE"

echo "[$name] Committed."


}

push_if_ahead() {
local path="$1"
local name="$2"
local expected_branch="$3"
local branch
local upstream
local ahead


verify_repo "$path" "$name" "$expected_branch"

branch="$(current_branch "$path")"

if ! upstream="$(git -C "$path" rev-parse --abbrev-ref '@{u}' 2>/dev/null)"; then
    echo "[$name] No upstream configured; skipping push."
    return 0
fi

ahead="$(git -C "$path" rev-list --count "$upstream..HEAD")"

if [[ "$ahead" -eq 0 ]]; then
    echo "[$name] Nothing to push."
    return 0
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[$name] Would push $ahead commit(s): $branch -> $upstream"
    git -C "$path" log --oneline "$upstream..HEAD" | sed 's/^/    /'
    return 0
fi

git -C "$path" push

echo "[$name] Pushed $ahead commit(s)."


}

main() {
local i
local name
local branch
local path


if [[ ! -d "$RISE_ROOT" ]]; then
    echo "RISE root does not exist: $RISE_ROOT" >&2
    exit 1
fi

if ! is_git_repo "$RISE_ROOT"; then
    echo "Not a Git repository: $RISE_ROOT" >&2
    exit 1
fi

echo "Main repository: $RISE_ROOT"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Mode: dry run"
fi

write_step "Commit sibling repositories"

for i in "${!REPO_NAMES[@]}"; do
    name="${REPO_NAMES[$i]}"
    branch="${REPO_BRANCHES[$i]}"
    path="$RISE_ROOT/$name"

    commit_if_dirty "$path" "$name" "$branch"
done

write_step "Commit main repository"

if [[ -z "$(git -C "$RISE_ROOT" status --porcelain)" ]]; then
    echo "[RISE-2026] No changes to commit."
elif [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[RISE-2026] Would commit with message: $MESSAGE"
    git -C "$RISE_ROOT" status --short | sed 's/^/    /'
else
    git -C "$RISE_ROOT" add -A
    git -C "$RISE_ROOT" commit -m "$MESSAGE"
    echo "[RISE-2026] Committed."
fi

write_step "Push sibling repositories"

for i in "${!REPO_NAMES[@]}"; do
    name="${REPO_NAMES[$i]}"
    branch="${REPO_BRANCHES[$i]}"
    path="$RISE_ROOT/$name"

    push_if_ahead "$path" "$name" "$branch"
done

write_step "Push main repository"

if ! is_git_repo "$RISE_ROOT"; then
    echo "[RISE-2026] Not a Git repository." >&2
    exit 1
fi

branch="$(current_branch "$RISE_ROOT")"

if [[ -z "$branch" ]]; then
    echo "[RISE-2026] Detached HEAD; skipping push." >&2
else
    push_if_ahead "$RISE_ROOT" "RISE-2026" "$branch"
fi

printf '\nDone.\n'


}

main "$@"
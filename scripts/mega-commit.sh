#!/usr/bin/env bash

# Commit and push the RISE-2026 repository and its sibling repositories

# using one commit message.

#

# Repositories:

# ~/RISE-2026

# ~/RISE-2026/SpecRLBench

# ~/RISE-2026/GenZ-LTL

# ~/RISE-2026/Safe-Policy-Optimization

#

# Only commits repositories with uncommitted changes.

# Only pushes repositories that are ahead of their configured upstream.

# Fails if a repository is not on its expected branch.

# Uses Git only; no sudo is required.

#

# Usage:

# ./scripts/mega-commit.sh "072828 sync SpecRLBench and RISE"

# ./scripts/mega-commit.sh --dry-run "072828 sync repositories"

set -euo pipefail

MESSAGE=""
DRY_RUN=0

RISE_ROOT="$HOME/RISE-2026"

# Format:

# repository label | relative path | expected branch

REPOSITORIES=(
"SpecRLBench|SpecRLBench|feat/sar-envs"
"GenZ-LTL|GenZ-LTL|feat/sar-props"
"Safe-Policy-Optimization|Safe-Policy-Optimization|feat/specrlbench-additions"
)

usage() {
cat <<'EOF'
Usage:
./scripts/mega-commit.sh "commit message"
./scripts/mega-commit.sh --dry-run "commit message"
./scripts/mega-commit.sh -n "commit message"

On Windows, use scripts/mega-commit.ps1 instead.
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

is_git_repository() {
git -C "$1" rev-parse --is-inside-work-tree >/dev/null 2>&1
}

is_dirty() {
git status --porcelain | grep -q .
}

current_branch() {
git symbolic-ref --quiet --short HEAD 2>/dev/null || true
}

upstream_ref() {
git rev-parse --abbrev-ref '@{u}' 2>/dev/null || true
}

ahead_count() {
local upstream="$1"

```
git rev-list --count "$upstream..HEAD"
```

}

verify_repository() {
local path="$1"
local label="$2"
local expected_branch="$3"
local branch

```
if [[ ! -d "$path" ]]; then
    echo "[$label] Repository path does not exist: $path" >&2
    return 1
fi

if ! is_git_repository "$path"; then
    echo "[$label] Not a Git repository: $path" >&2
    return 1
fi

branch="$(git -C "$path" symbolic-ref --quiet --short HEAD 2>/dev/null || true)"

if [[ -z "$branch" ]]; then
    echo "[$label] Detached HEAD; refusing to continue." >&2
    return 1
fi

if [[ "$branch" != "$expected_branch" ]]; then
    echo "[$label] Branch mismatch:" >&2
    echo "  Current:  $branch" >&2
    echo "  Expected: $expected_branch" >&2
    return 1
fi
```

}

show_pending_commit() {
local path="$1"
local label="$2"
local status_lines

```
echo "[$label] Would commit with message: $MESSAGE"

status_lines="$(git -C "$path" status --short)"

if [[ -n "$status_lines" ]]; then
    while IFS= read -r line; do
        [[ -n "$line" ]] && printf '    %s\n' "$line"
    done <<< "$status_lines"
fi
```

}

show_pending_push() {
local path="$1"
local label="$2"
local branch="$3"
local upstream="$4"
local ahead="$5"

```
echo "[$label] Would push $ahead commit(s): $branch -> $upstream"

git -C "$path" log \
    --oneline \
    "$upstream..HEAD" |
    sed 's/^/    /'
```

}

commit_if_dirty() {
local path="$1"
local label="$2"
local expected_branch="$3"

```
verify_repository "$path" "$label" "$expected_branch"

if ! git -C "$path" status --porcelain | grep -q .; then
    echo "[$label] No changes to commit."
    return 0
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    show_pending_commit "$path" "$label"
    return 0
fi

git -C "$path" add -A
git -C "$path" commit -m "$MESSAGE"

echo "[$label] Committed."
```

}

push_if_ahead() {
local path="$1"
local label="$2"
local expected_branch="$3"
local branch upstream ahead

```
verify_repository "$path" "$label" "$expected_branch"

branch="$(git -C "$path" symbolic-ref --quiet --short HEAD)"
upstream="$(git -C "$path" rev-parse --abbrev-ref '@{u}' 2>/dev/null || true)"

if [[ -z "$upstream" ]]; then
    echo "[$label] No upstream configured; skipping push."
    return 0
fi

ahead="$(git -C "$path" rev-list --count "$upstream..HEAD")"

if [[ "$ahead" -eq 0 ]]; then
    echo "[$label] Nothing to push."
    return 0
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    show_pending_push \
        "$path" \
        "$label" \
        "$branch" \
        "$upstream" \
        "$ahead"
    return 0
fi

git -C "$path" push

echo "[$label] Pushed $ahead commit(s)."
```

}

main() {
local entry
local label relative_path expected_branch path

```
if [[ ! -d "$RISE_ROOT" ]]; then
    echo "RISE-2026 directory does not exist: $RISE_ROOT" >&2
    exit 1
fi

if ! is_git_repository "$RISE_ROOT"; then
    echo "Main repository is not a Git repository: $RISE_ROOT" >&2
    exit 1
fi

echo "Main repository: $RISE_ROOT"
echo "Sibling repositories:"

for entry in "${REPOSITORIES[@]}"; do
    IFS='|' read -r label relative_path expected_branch <<< "$entry"

    printf '  - %s\n' "$label"
    printf '    path: %s/%s\n' "$RISE_ROOT" "$relative_path"
    printf '    branch: %s\n' "$expected_branch"
done

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Mode: dry run (no commit or push)"
fi

write_step "Commit sibling repositories"

for entry in "${REPOSITORIES[@]}"; do
    IFS='|' read -r label relative_path expected_branch <<< "$entry"

    path="$RISE_ROOT/$relative_path"

    commit_if_dirty \
        "$path" \
        "$label" \
        "$expected_branch"
done

write_step "Commit main repository"

if git -C "$RISE_ROOT" status --porcelain | grep -q .; then
    if [[ "$DRY_RUN" -eq 1 ]]; then
        show_pending_commit "$RISE_ROOT" "RISE-2026"
    else
        git -C "$RISE_ROOT" add -A
        git -C "$RISE_ROOT" commit -m "$MESSAGE"

        echo "[RISE-2026] Committed."
    fi
else
    echo "[RISE-2026] No changes to commit."
fi

write_step "Push sibling repositories"

for entry in "${REPOSITORIES[@]}"; do
    IFS='|' read -r label relative_path expected_branch <<< "$entry"

    path="$RISE_ROOT/$relative_path"

    push_if_ahead \
        "$path" \
        "$label" \
        "$expected_branch"
done

write_step "Push main repository"

push_if_ahead \
    "$RISE_ROOT" \
    "RISE-2026" \
    "$(git -C "$RISE_ROOT" branch --show-current)"

printf '\nDone.\n'
```

}

main "$@"

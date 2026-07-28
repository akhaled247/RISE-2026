#!/usr/bin/env bash

# Commit and push the main repository and all configured Git submodules

# using one commit message.

#

# Only commits repositories with uncommitted changes.

# Only pushes repositories that are ahead of their configured upstream.

# Reads submodule paths and configured branches from .gitmodules.

# Uses Git only; no sudo is required.

#

# Usage:

# ./scripts/mega-commit.sh "072828 sync SpecRLBench and RISE"

# ./scripts/mega-commit.sh --dry-run "072828 sync submodules"

set -euo pipefail

MESSAGE=""
DRY_RUN=0

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

write_detail() {
printf '%s\n' "$1"
}

repo_root() {
git rev-parse --show-toplevel
}

get_submodules() {
local root="$1"
local gitmodules="$root/.gitmodules"
local key name

```
[[ -f "$gitmodules" ]] || return 0

while IFS= read -r key; do
    name="${key#submodule.}"
    name="${name%.path}"

    printf '%s\t%s\t%s\t%s\n' \
        "$name" \
        "$(git config --file "$gitmodules" --get "$key")" \
        "$(git config --file "$gitmodules" --get "submodule.$name.url" 2>/dev/null || true)" \
        "$(git config --file "$gitmodules" --get "submodule.$name.branch" 2>/dev/null || true)"
done < <(
    git config \
        --file "$gitmodules" \
        --name-only \
        --get-regexp '^submodule\..*\.path$'
)
```

}

is_git_worktree() {
git rev-parse --is-inside-work-tree >/dev/null 2>&1
}

is_dirty() {
[[ -n "$(git status --porcelain)" ]]
}

current_branch() {
git symbolic-ref --quiet --short HEAD 2>/dev/null || true
}

upstream_ref() {
git rev-parse --abbrev-ref '@{u}' 2>/dev/null || true
}

ahead_count() {
local upstream

```
upstream="$(upstream_ref)"
if [[ -z "$upstream" ]]; then
    echo 0
    return 0
fi

git rev-list --count "$upstream..HEAD" 2>/dev/null || echo 0
```

}

show_pending_commit() {
local label="$1"
local status_lines staged_lines unstaged_lines porcelain_lines

```
echo "[$label] Would commit with message: $MESSAGE"

status_lines="$(git status --short 2>/dev/null || true)"
if [[ -n "$status_lines" ]]; then
    write_detail "  Status:"
    while IFS= read -r line; do
        [[ -n "$line" ]] && write_detail "    $line"
    done <<< "$status_lines"
fi

staged_lines="$(git diff --cached --stat 2>/dev/null || true)"
if [[ -n "$staged_lines" ]]; then
    write_detail "  Staged diff:"
    while IFS= read -r line; do
        [[ -n "$line" ]] && write_detail "    $line"
    done <<< "$staged_lines"
fi

unstaged_lines="$(git diff --stat 2>/dev/null || true)"
if [[ -n "$unstaged_lines" ]]; then
    write_detail "  Unstaged diff:"
    while IFS= read -r line; do
        [[ -n "$line" ]] && write_detail "    $line"
    done <<< "$unstaged_lines"
fi

if [[ -z "$status_lines" &&
      -z "$staged_lines" &&
      -z "$unstaged_lines" ]]; then
    write_detail "  Status details:"
    porcelain_lines="$(git status --porcelain 2>/dev/null || true)"

    while IFS= read -r line; do
        [[ -n "$line" ]] && write_detail "    $line"
    done <<< "$porcelain_lines"
fi
```

}

show_pending_push() {
local label="$1"
local ahead="$2"
local branch="$3"
local upstream="$4"
local commits

```
echo "[$label] Would push $ahead commit(s): $branch -> $upstream"

commits="$(git log --oneline "$upstream..HEAD" 2>/dev/null || true)"
while IFS= read -r line; do
    [[ -n "$line" ]] && write_detail "    $line"
done <<< "$commits"
```

}

commit_if_dirty() {
local path="$1"
local label="$2"
local configured_branch="${3:-}"

```
(
    cd "$path"

    if ! is_git_worktree; then
        echo "[$label] Not an initialized Git worktree; skipping."
        exit 0
    fi

    local branch
    branch="$(current_branch)"

    if [[ -z "$branch" ]]; then
        echo "[$label] Detached HEAD; skipping commit."
        exit 0
    fi

    if [[ -n "$configured_branch" &&
          "$branch" != "$configured_branch" ]]; then
        echo "[$label] Current branch: $branch; configured branch: $configured_branch"
    fi

    if ! is_dirty; then
        echo "[$label] No changes to commit."
        exit 0
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        show_pending_commit "$label"
        exit 0
    fi

    git add -A
    git commit -m "$MESSAGE"

    echo "[$label] Committed on $branch."
)
```

}

push_if_ahead() {
local path="$1"
local label="$2"
local configured_branch="${3:-}"

```
(
    cd "$path"

    if ! is_git_worktree; then
        echo "[$label] Not an initialized Git worktree; skipping."
        exit 0
    fi

    local branch upstream ahead

    branch="$(current_branch)"
    if [[ -z "$branch" ]]; then
        echo "[$label] Detached HEAD; skipping push."
        exit 0
    fi

    if [[ -n "$configured_branch" &&
          "$branch" != "$configured_branch" ]]; then
        echo "[$label] Current branch: $branch; configured branch: $configured_branch"
    fi

    upstream="$(upstream_ref)"
    if [[ -z "$upstream" ]]; then
        echo "[$label] No upstream configured; skipping push."
        exit 0
    fi

    ahead="$(ahead_count)"
    if [[ "$ahead" -le 0 ]]; then
        echo "[$label] Nothing to push."
        exit 0
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        show_pending_push "$label" "$ahead" "$branch" "$upstream"
        exit 0
    fi

    git push

    echo "[$label] Pushed $ahead commit(s): $branch -> $upstream"
)
```

}

main() {
local root
local name path url branch
local -a submodules=()

```
root="$(repo_root)"
cd "$root"

while IFS=$'\t' read -r name path url branch; do
    [[ -n "$path" ]] || continue
    submodules+=("$name"$'\t'"$path"$'\t'"$url"$'\t'"$branch")
done < <(get_submodules "$root")

echo "Repo: $root"

if [[ "${#submodules[@]}" -eq 0 ]]; then
    echo "Submodules: (none)"
else
    echo "Submodules:"

    for name in "${submodules[@]}"; do
        IFS=$'\t' read -r name path url branch <<< "$name"

        printf '  - %s\n' "$name"
        printf '    path: %s\n' "$path"

        [[ -n "$url" ]] &&
            printf '    url: %s\n' "$url"

        [[ -n "$branch" ]] &&
            printf '    branch: %s\n' "$branch"
    done
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Mode: dry run (no commit or push)"
fi

write_step "Commit submodules"

for entry in "${submodules[@]}"; do
    IFS=$'\t' read -r name path url branch <<< "$entry"

    if [[ ! -d "$root/$path" ]]; then
        echo "[$name] Path missing: $path; skipping."
        continue
    fi

    commit_if_dirty "$root/$path" "$name" "$branch"
done

write_step "Commit main repository"
commit_if_dirty "$root" "main"

write_step "Push submodules"

for entry in "${submodules[@]}"; do
    IFS=$'\t' read -r name path url branch <<< "$entry"

    if [[ ! -d "$root/$path" ]]; then
        echo "[$name] Path missing: $path; skipping."
        continue
    fi

    push_if_ahead "$root/$path" "$name" "$branch"
done

write_step "Push main repository"
push_if_ahead "$root" "main"

printf '\nDone.\n'
```

}

main "$@"

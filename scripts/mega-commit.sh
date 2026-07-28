#!/usr/bin/env bash
# Commit and push the main RISE-2026 repository and its submodules with one message.
# Only commits dirty repos; only pushes repos that are ahead of upstream.
# If a submodule is on a detached HEAD, attaches it to the branch from .gitmodules.
# Uses git only (no sudo).
#
# Repository layout (submodules discovered from .gitmodules):
#   RISE-2026/
#     SpecRLBench/
#     GenZ-LTL/
#     Safe-Policy-Optimization/
#
# Usage:
#   ./scripts/mega-commit.sh "commit message"
#   ./scripts/mega-commit.sh --dry-run "commit message"
#   ./scripts/mega-commit.sh -n "commit message"

set -euo pipefail

MESSAGE=""
DRY_RUN=0

RISE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SUBMODULE_NAMES=()
SUBMODULE_PATHS=()
SUBMODULE_BRANCHES=()

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

is_git_repo() {
    git -C "$1" rev-parse --is-inside-work-tree >/dev/null 2>&1
}

current_branch() {
    git -C "$1" branch --show-current
}

load_submodules() {
    local gitmodules="$RISE_ROOT/.gitmodules"
    local name path branch key value

    SUBMODULE_NAMES=()
    SUBMODULE_PATHS=()
    SUBMODULE_BRANCHES=()

    [[ -f "$gitmodules" ]] || return 0

    declare -A paths=()
    declare -A branches=()

    while IFS= read -r line; do
        [[ -n "$line" ]] || continue
        if [[ "$line" =~ ^submodule\.(.+)\.(path|branch)[[:space:]]+(.+)$ ]]; then
            name="${BASH_REMATCH[1]}"
            key="${BASH_REMATCH[2]}"
            value="${BASH_REMATCH[3]}"
            if [[ "$key" == "path" ]]; then
                paths["$name"]="$value"
            else
                branches["$name"]="$value"
            fi
        fi
    done < <(git config --file "$gitmodules" --get-regexp '^submodule\..*\.(path|branch)$')

    for name in "${!paths[@]}"; do
        path="${paths[$name]}"
        branch="${branches[$name]:-}"
        SUBMODULE_NAMES+=("$name")
        SUBMODULE_PATHS+=("$path")
        SUBMODULE_BRANCHES+=("$branch")
    done
}

ensure_attached_head() {
    local path="$1"
    local name="$2"
    local expected_branch="$3"
    local branch

    branch="$(current_branch "$path")"
    if [[ -n "$branch" ]]; then
        printf '%s' "$branch"
        return 0
    fi

    if [[ -z "$expected_branch" ]]; then
        echo "[$name] Detached HEAD and no branch configured in .gitmodules." >&2
        return 1
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[$name] Would attach detached HEAD to branch: $expected_branch"
        printf '%s' "$expected_branch"
        return 0
    fi

    echo "[$name] Detached HEAD detected; attaching to $expected_branch"
    git -C "$path" fetch origin "$expected_branch" >/dev/null 2>&1 || true

    if git -C "$path" show-ref --verify --quiet "refs/heads/$expected_branch"; then
        git -C "$path" checkout "$expected_branch"
    else
        git -C "$path" checkout -B "$expected_branch" "origin/$expected_branch"
    fi

    if ! git -C "$path" rev-parse --abbrev-ref '@{u}' >/dev/null 2>&1; then
        git -C "$path" branch --set-upstream-to "origin/$expected_branch" "$expected_branch" >/dev/null 2>&1 || true
    fi

    branch="$(current_branch "$path")"
    if [[ -z "$branch" ]]; then
        echo "[$name] Failed to attach detached HEAD to $expected_branch." >&2
        return 1
    fi

    echo "[$name] Attached HEAD to $branch"
    printf '%s' "$branch"
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

    branch="$(ensure_attached_head "$path" "$name" "$expected_branch")"

    if [[ -n "$expected_branch" && "$branch" != "$expected_branch" ]]; then
        echo "[$name] Warning: on '$branch' (configured branch is '$expected_branch')."
    fi

    printf '%s' "$branch"
}

show_pending_commit() {
    local path="$1"
    local name="$2"
    local status_lines staged_lines unstaged_lines

    echo "[$name] Would commit with message: $MESSAGE"

    status_lines="$(git -C "$path" status --short 2>/dev/null || true)"
    if [[ -n "$status_lines" ]]; then
        write_detail "  Status:"
        while IFS= read -r line; do
            [[ -n "$line" ]] && write_detail "    $line"
        done <<< "$status_lines"
    fi

    staged_lines="$(git -C "$path" diff --cached --stat 2>/dev/null || true)"
    if [[ -n "$staged_lines" ]]; then
        write_detail "  Staged diff:"
        while IFS= read -r line; do
            [[ -n "$line" ]] && write_detail "    $line"
        done <<< "$staged_lines"
    fi

    unstaged_lines="$(git -C "$path" diff --stat 2>/dev/null || true)"
    if [[ -n "$unstaged_lines" ]]; then
        write_detail "  Unstaged diff:"
        while IFS= read -r line; do
            [[ -n "$line" ]] && write_detail "    $line"
        done <<< "$unstaged_lines"
    fi
}

commit_if_dirty() {
    local path="$1"
    local name="$2"
    local expected_branch="$3"

    verify_repo "$path" "$name" "$expected_branch" >/dev/null

    if [[ -z "$(git -C "$path" status --porcelain)" ]]; then
        echo "[$name] No changes to commit."
        return 0
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        show_pending_commit "$path" "$name"
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
    local branch upstream ahead

    branch="$(verify_repo "$path" "$name" "$expected_branch")"

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
    local i name path branch submodule_list

    if [[ ! -d "$RISE_ROOT" ]]; then
        echo "RISE root does not exist: $RISE_ROOT" >&2
        exit 1
    fi

    if ! is_git_repo "$RISE_ROOT"; then
        echo "Not a Git repository: $RISE_ROOT" >&2
        exit 1
    fi

    load_submodules

    echo "Main repository: $RISE_ROOT"
    if [[ "${#SUBMODULE_PATHS[@]}" -eq 0 ]]; then
        echo "Submodules: (none)"
    else
        submodule_list="$(printf '%s, ' "${SUBMODULE_PATHS[@]}")"
        echo "Submodules: ${submodule_list%, }"
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "Mode: dry run"
    fi

    write_step "Commit submodule repositories"
    for i in "${!SUBMODULE_PATHS[@]}"; do
        name="${SUBMODULE_NAMES[$i]}"
        path="$RISE_ROOT/${SUBMODULE_PATHS[$i]}"
        branch="${SUBMODULE_BRANCHES[$i]}"
        commit_if_dirty "$path" "$name" "$branch"
    done

    write_step "Commit main repository"
    if [[ -z "$(git -C "$RISE_ROOT" status --porcelain)" ]]; then
        echo "[RISE-2026] No changes to commit."
    elif [[ "$DRY_RUN" -eq 1 ]]; then
        show_pending_commit "$RISE_ROOT" "RISE-2026"
    else
        git -C "$RISE_ROOT" add -A
        git -C "$RISE_ROOT" commit -m "$MESSAGE"
        echo "[RISE-2026] Committed."
    fi

    write_step "Push submodule repositories"
    for i in "${!SUBMODULE_PATHS[@]}"; do
        name="${SUBMODULE_NAMES[$i]}"
        path="$RISE_ROOT/${SUBMODULE_PATHS[$i]}"
        branch="${SUBMODULE_BRANCHES[$i]}"
        push_if_ahead "$path" "$name" "$branch"
    done

    write_step "Push main repository"
    branch="$(current_branch "$RISE_ROOT")"
    if [[ -z "$branch" ]]; then
        echo "[RISE-2026] Detached HEAD; refusing to push main repository." >&2
        exit 1
    fi

    push_if_ahead "$RISE_ROOT" "RISE-2026" "$branch"

    printf '\nDone.\n'
}

main "$@"

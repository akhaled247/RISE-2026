#!/usr/bin/env bash
# Commit and push the main repo plus all git submodules with one message.
# Only commits dirty repos; only pushes repos that are ahead of upstream.
# Uses git only (no sudo).
#
# Usage:
#   ./scripts/mega-commit.sh "072828 sync SpecRLBench and RISE"
#   ./scripts/mega-commit.sh --dry-run "072828 sync submodules"

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
    if [[ ! -f "$gitmodules" ]]; then
        return 0
    fi
    git config --file "$gitmodules" --get-regexp '^submodule\..*\.path$' | awk '{ print $2 }'
}

is_dirty() {
    [[ -n "$(git status --porcelain)" ]]
}

ahead_count() {
    if ! git rev-parse --abbrev-ref '@{u}' >/dev/null 2>&1; then
        echo 0
        return 0
    fi
    git rev-list --count '@{u}..HEAD' 2>/dev/null || echo 0
}

show_pending_commit() {
    local label="$1"
    local status_lines staged_lines unstaged_lines porcelain_lines

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

    if [[ -z "$status_lines" && -z "$staged_lines" && -z "$unstaged_lines" ]]; then
        write_detail "  (dirty per porcelain, but no diff details - possibly line-ending or mode-only changes)"
        porcelain_lines="$(git status --porcelain 2>/dev/null || true)"
        while IFS= read -r line; do
            [[ -n "$line" ]] && write_detail "    $line"
        done <<< "$porcelain_lines"
    fi
}

show_pending_push() {
    local label="$1"
    local ahead="$2"
    local branch="$3"
    local commits

    echo "[$label] Would push $ahead commit(s) on $branch:"
    commits="$(git log --oneline '@{u}..HEAD' 2>/dev/null || true)"
    while IFS= read -r line; do
        [[ -n "$line" ]] && write_detail "    $line"
    done <<< "$commits"
}

commit_if_dirty() {
    local path="$1"
    local label="$2"

    (
        cd "$path"
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
        echo "[$label] Committed."
    )
}

push_if_ahead() {
    local path="$1"
    local label="$2"

    (
        cd "$path"
        local branch ahead

        branch="$(git rev-parse --abbrev-ref HEAD)"
        if ! git rev-parse --abbrev-ref '@{u}' >/dev/null 2>&1; then
            echo "[$label] No upstream configured; skipping push."
            exit 0
        fi

        ahead="$(ahead_count)"
        if [[ "$ahead" -le 0 ]]; then
            echo "[$label] Nothing to push."
            exit 0
        fi

        if [[ "$DRY_RUN" -eq 1 ]]; then
            show_pending_push "$label" "$ahead" "$branch"
            exit 0
        fi

        git push
        echo "[$label] Pushed $ahead commit(s)."
    )
}

main() {
    local root subs sub_list

    root="$(repo_root)"
    cd "$root"

    mapfile -t sub_list < <(get_submodules "$root")

    echo "Repo: $root"
    if [[ "${#sub_list[@]}" -eq 0 ]]; then
        echo "Submodules: (none)"
    else
        subs="$(printf '%s, ' "${sub_list[@]}")"
        echo "Submodules: ${subs%, }"
    fi
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "Mode: dry run (no commit/push)"
    fi

    write_step "Commit submodules"
    for sub in "${sub_list[@]}"; do
        [[ -n "$sub" ]] || continue
        if [[ ! -d "$root/$sub" ]]; then
            echo "[$sub] Path missing; skipping."
            continue
        fi
        commit_if_dirty "$root/$sub" "$sub"
    done

    write_step "Commit main repo"
    commit_if_dirty "$root" "main"

    write_step "Push submodules"
    for sub in "${sub_list[@]}"; do
        [[ -n "$sub" ]] || continue
        if [[ -d "$root/$sub" ]]; then
            push_if_ahead "$root/$sub" "$sub"
        fi
    done

    write_step "Push main repo"
    push_if_ahead "$root" "main"

    printf '\nDone.\n'
}

main "$@"

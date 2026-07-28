# Commit and push the main repo plus all git submodules with one message.
# Only commits dirty repos; only pushes repos that are ahead of upstream.
#
# Usage:
#   .\scripts\mega-commit.ps1 "072828 sync SpecRLBench and RISE"
#   .\scripts\mega-commit.ps1 "fix eval" -DryRun

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Message,

    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$Text) {
    Write-Host "`n=== $Text ===" -ForegroundColor Cyan
}

function Write-Detail([string]$Text) {
    Write-Host $Text -ForegroundColor DarkGray
}

function Get-RepoRoot {
    $root = git rev-parse --show-toplevel 2>$null
    if (-not $root) { throw "Not inside a git repository." }
    return $root
}

function Get-SubmodulePaths([string]$Root) {
    $gitmodules = Join-Path $Root ".gitmodules"
    if (-not (Test-Path $gitmodules)) { return @() }

    git config --file $gitmodules --get-regexp '^submodule\..*\.path$' |
        ForEach-Object {
            ($_ -split ' ', 2)[1]
        }
}

function Get-SubmoduleBranch {
    param(
        [string]$Root,
        [string]$SubPath
    )

    $gitmodules = Join-Path $Root ".gitmodules"
    if (-not (Test-Path $gitmodules)) { return $null }

    $entries = @(git config --file $gitmodules --get-regexp '^submodule\..*\.path$')
    foreach ($entry in $entries) {
        $name, $path = $entry -split ' ', 2
        if ($path -ne $SubPath) { continue }

        $section = $name -replace '\.path$', ''
        return git config --file $gitmodules --get "${section}.branch"
    }

    return $null
}

function Test-RepoDirty {
    return [bool](git status --porcelain)
}

function Test-HasUpstream {
    $null = git rev-parse --abbrev-ref "@{u}" 2>&1
    return $LASTEXITCODE -eq 0
}

function Get-AheadCount {
    if (-not (Test-HasUpstream)) { return 0 }

    $count = git rev-list --count "@{u}..HEAD" 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $count) { return 0 }
    return [int]$count
}

function Ensure-OnSubmoduleBranch {
    param(
        [string]$Root,
        [string]$SubPath,
        [string]$Label
    )

    Push-Location (Join-Path $Root $SubPath)
    try {
        $current = git rev-parse --abbrev-ref HEAD
        if ($current -ne "HEAD") { return }

        $branch = Get-SubmoduleBranch -Root $Root -SubPath $SubPath
        if (-not $branch) {
            Write-Warning "[$Label] Detached HEAD and no submodule.*.branch in .gitmodules."
            return
        }

        $at = git rev-parse HEAD
        git checkout $branch 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "[$Label] Could not checkout branch '$branch'."
        }

        $branchAt = git rev-parse HEAD
        if ($branchAt -ne $at) {
            git merge-base --is-ancestor $branchAt $at 2>$null
            if ($LASTEXITCODE -eq 0) {
                git merge --ff-only $at 2>&1 | Out-Null
                if ($LASTEXITCODE -ne 0) {
                    throw "[$Label] Could not fast-forward '$branch' to $at."
                }
            }
        }

        Write-Host "[$Label] On branch $branch (was detached at $($at.Substring(0, 7)))."
    }
    finally {
        Pop-Location
    }
}

function Show-PendingCommit {
    param([string]$Label)

    Write-Host "[$Label] Would commit with message: $Message"

    $status = @(git status --short 2>$null)
    if ($status.Count -gt 0) {
        Write-Detail "  Status:"
        $status | ForEach-Object { Write-Detail "    $_" }
    }

    $staged = @(git diff --cached --stat 2>$null)
    if ($staged.Count -gt 0) {
        Write-Detail "  Staged diff:"
        $staged | ForEach-Object { Write-Detail "    $_" }
    }

    $unstaged = @(git diff --stat 2>$null)
    if ($unstaged.Count -gt 0) {
        Write-Detail "  Unstaged diff:"
        $unstaged | ForEach-Object { Write-Detail "    $_" }
    }

    if ($status.Count -eq 0 -and $staged.Count -eq 0 -and $unstaged.Count -eq 0) {
        Write-Detail "  (dirty per porcelain, but no diff details - possibly line-ending or mode-only changes)"
        $porcelain = @(git status --porcelain 2>$null)
        $porcelain | ForEach-Object { Write-Detail "    $_" }
    }
}

function Show-PendingPush {
    param(
        [string]$Label,
        [int]$Ahead,
        [string]$Branch
    )

    Write-Host "[$Label] Would push $Ahead commit(s) on ${Branch}:"
    $commits = @(git log --oneline "@{u}..HEAD" 2>$null)
    if ($commits.Count -gt 0) {
        $commits | ForEach-Object { Write-Detail "    $_" }
    }
}

function Invoke-CommitIfDirty {
    param(
        [string]$Path,
        [string]$Label,
        [string]$Message,
        [switch]$DryRun
    )

    Push-Location $Path
    try {
        if (-not (Test-RepoDirty)) {
            Write-Host "[$Label] No changes to commit."
            return
        }

        if ($DryRun) {
            Show-PendingCommit -Label $Label
            return
        }

        git add -A
        git commit -m $Message
        Write-Host "[$Label] Committed."
    }
    finally {
        Pop-Location
    }
}

function Invoke-PushIfAhead {
    param(
        [string]$Path,
        [string]$Label,
        [switch]$DryRun
    )

    Push-Location $Path
    try {
        $branch = git rev-parse --abbrev-ref HEAD
        if ($branch -eq "HEAD") {
            Write-Host "[$Label] Detached HEAD; skipping push."
            return
        }

        if (-not (Test-HasUpstream)) {
            Write-Host "[$Label] No upstream configured; skipping push."
            return
        }

        $ahead = Get-AheadCount
        if ($ahead -le 0) {
            Write-Host "[$Label] Nothing to push."
            return
        }

        if ($DryRun) {
            Show-PendingPush -Label $Label -Ahead $ahead -Branch $branch
            return
        }

        git push
        Write-Host "[$Label] Pushed $ahead commit(s)."
    }
    finally {
        Pop-Location
    }
}

$root = Get-RepoRoot
Set-Location $root

$submodules = Get-SubmodulePaths $root
Write-Host "Repo: $root"
if ($submodules.Count -eq 0) {
    Write-Host "Submodules: (none)"
} else {
    Write-Host "Submodules: $($submodules -join ', ')"
}
if ($DryRun) {
    Write-Host "Mode: dry run (no commit/push)" -ForegroundColor Yellow
}

Write-Step "Commit submodules"
foreach ($sub in $submodules) {
    $subPath = Join-Path $root $sub
    if (-not (Test-Path $subPath)) {
        Write-Host "[$sub] Path missing; skipping."
        continue
    }
    if (-not $DryRun) {
        Ensure-OnSubmoduleBranch -Root $root -SubPath $sub -Label $sub
    }
    Invoke-CommitIfDirty -Path $subPath -Label $sub -Message $Message -DryRun:$DryRun
}

Write-Step "Commit main repo"
Invoke-CommitIfDirty -Path $root -Label "main" -Message $Message -DryRun:$DryRun

Write-Step "Push submodules"
foreach ($sub in $submodules) {
    $subPath = Join-Path $root $sub
    if (Test-Path $subPath) {
        Invoke-PushIfAhead -Path $subPath -Label $sub -DryRun:$DryRun
    }
}

Write-Step "Push main repo"
Invoke-PushIfAhead -Path $root -Label "main" -DryRun:$DryRun

Write-Host "`nDone."

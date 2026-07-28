<#
Commit and push the main RISE-2026 repository and its submodules with one message.
Only commits dirty repos; only pushes repos that are ahead of upstream.
If a submodule is on a detached HEAD, attaches it to the branch from .gitmodules.

Repository layout (submodules discovered from .gitmodules):
  RISE-2026/
    SpecRLBench/
    GenZ-LTL/
    Safe-Policy-Optimization/

Usage:
  .\scripts\mega-commit.ps1 "commit message"
  .\scripts\mega-commit.ps1 -DryRun "commit message"
  .\scripts\mega-commit.ps1 -n "commit message"
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Message,

    [Alias("n")]
    [switch]$DryRun,

    [switch]$Help
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RiseRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Show-Usage {
    Write-Host @"
Usage:
  .\scripts\mega-commit.ps1 "commit message"
  .\scripts\mega-commit.ps1 -DryRun "commit message"
  .\scripts\mega-commit.ps1 -n "commit message"

Examples:
  .\scripts\mega-commit.ps1 "072828 sync submodules"
  .\scripts\mega-commit.ps1 -DryRun "072828 sync submodules"
"@
}

function Write-Step {
    param([Parameter(Mandatory)][string]$Text)
    Write-Host ""
    Write-Host "=== $Text ==="
}

function Write-Detail {
    param([Parameter(Mandatory)][string]$Text)
    Write-Host $Text -ForegroundColor DarkGray
}

function ConvertTo-StringArray {
    param($Value)

    if ($null -eq $Value) {
        return @()
    }

    return @(
        $Value |
            Where-Object { $null -ne $_ } |
            ForEach-Object { $_.ToString() }
    )
}

function Invoke-GitCommand {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$GitArguments
    )

    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & git -C $Path @GitArguments 2>&1
        if ($null -eq $output) {
            $output = @()
        }
        elseif ($output -isnot [System.Array]) {
            $output = @($output)
        }

        return @{
            ExitCode = $LASTEXITCODE
            Output   = $output
        }
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
}

function Invoke-Git {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$GitArguments
    )

    $result = Invoke-GitCommand -Path $Path @GitArguments
    if ($result.ExitCode -ne 0) {
        $detail = ($result.Output | Out-String).Trim()
        if ($detail) {
            throw "Git command failed in '$Path': git $($GitArguments -join ' ')`n$detail"
        }
        throw "Git command failed in '$Path': git $($GitArguments -join ' ')"
    }
}

function Test-GitRepository {
    param([Parameter(Mandatory)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        return $false
    }

    $result = Invoke-GitCommand -Path $Path rev-parse --is-inside-work-tree
    return (
        $result.ExitCode -eq 0 -and
        (($result.Output | Select-Object -First 1).ToString().Trim() -eq "true")
    )
}

function Get-CurrentBranch {
    param([Parameter(Mandatory)][string]$Path)

    $result = Invoke-GitCommand -Path $Path branch --show-current
    if ($result.ExitCode -ne 0) {
        throw "Could not determine the current branch for '$Path'."
    }

    $branch = $result.Output | Select-Object -First 1
    if ($null -eq $branch) {
        return ""
    }

    return ($branch.ToString().Trim())
}

function Get-SubmoduleRepositories {
    param([Parameter(Mandatory)][string]$Root)

    $gitmodules = Join-Path $Root ".gitmodules"
    if (-not (Test-Path -LiteralPath $gitmodules)) {
        return @()
    }

    $entries = @{}
    $configResult = Invoke-GitCommand -Path $Root config --file $gitmodules --get-regexp '^submodule\..*\.(path|branch)$'
    if ($configResult.ExitCode -ne 0) {
        return @()
    }

    foreach ($line in $configResult.Output) {
        $line = $line.ToString()
        if ($line -match '^submodule\.(.+)\.(path|branch)\s+(.+)$') {
            $name = $Matches[1]
            $key = $Matches[2]
            $value = $Matches[3].Trim()

            if (-not $entries.ContainsKey($name)) {
                $entries[$name] = @{
                    Name   = $name
                    Path   = ""
                    Branch = ""
                }
            }

            $entries[$name][$key] = $value
        }
    }

    return @(
        $entries.Values |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_.Path) } |
            Sort-Object Path
    )
}

function Ensure-AttachedHead {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Name,
        [string]$ExpectedBranch
    )

    $branch = Get-CurrentBranch -Path $Path
    if (-not [string]::IsNullOrWhiteSpace($branch)) {
        return $branch
    }

    if ([string]::IsNullOrWhiteSpace($ExpectedBranch)) {
        throw "[$Name] Detached HEAD and no branch configured in .gitmodules."
    }

    if ($DryRun) {
        Write-Host "[$Name] Would attach detached HEAD to branch: $ExpectedBranch"
        return $ExpectedBranch
    }

    Write-Host "[$Name] Detached HEAD detected; attaching to $ExpectedBranch"

    $fetchResult = Invoke-GitCommand -Path $Path fetch origin $ExpectedBranch
    if ($fetchResult.ExitCode -ne 0) {
        $detail = ($fetchResult.Output | Out-String).Trim()
        throw "[$Name] Failed to fetch origin/$ExpectedBranch.`n$detail"
    }

    $localRef = "refs/heads/$ExpectedBranch"
    $localResult = Invoke-GitCommand -Path $Path show-ref --verify --quiet $localRef
    if ($localResult.ExitCode -eq 0) {
        Invoke-Git -Path $Path checkout $ExpectedBranch
    }
    else {
        Invoke-Git -Path $Path checkout -B $ExpectedBranch "origin/$ExpectedBranch"
    }

    $upstreamResult = Invoke-GitCommand -Path $Path rev-parse --abbrev-ref "@{u}"
    if ($upstreamResult.ExitCode -ne 0) {
        Invoke-GitCommand -Path $Path branch --set-upstream-to "origin/$ExpectedBranch" $ExpectedBranch | Out-Null
    }

    $branch = Get-CurrentBranch -Path $Path
    if ([string]::IsNullOrWhiteSpace($branch)) {
        throw "[$Name] Failed to attach detached HEAD to $ExpectedBranch."
    }

    Write-Host "[$Name] Attached HEAD to $branch"
    return $branch
}

function Test-Repository {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Name,
        [string]$ExpectedBranch
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "[$Name] Directory does not exist: $Path"
    }

    if (-not (Test-GitRepository -Path $Path)) {
        throw "[$Name] Not a Git repository: $Path"
    }

    $branch = Ensure-AttachedHead -Path $Path -Name $Name -ExpectedBranch $ExpectedBranch

    if (
        -not [string]::IsNullOrWhiteSpace($ExpectedBranch) -and
        $branch -ne $ExpectedBranch
    ) {
        Write-Host "[$Name] Warning: on '$branch' (configured branch is '$ExpectedBranch')."
    }

    return $branch
}

function Get-GitStatus {
    param([Parameter(Mandatory)][string]$Path)

    $result = Invoke-GitCommand -Path $Path status --porcelain
    if ($result.ExitCode -ne 0) {
        throw "Could not get Git status for '$Path'."
    }

    return ConvertTo-StringArray $result.Output
}

function Show-PendingCommit {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Name
    )

    Write-Host "[$Name] Would commit with message: $Message"

    $status = @(Get-GitStatus -Path $Path)
    if ($status) {
        Write-Detail "  Status:"
        foreach ($line in $status) {
            Write-Detail "    $line"
        }
    }

    $staged = @(ConvertTo-StringArray (Invoke-GitCommand -Path $Path diff --cached --stat).Output)
    if ($staged) {
        Write-Detail "  Staged diff:"
        foreach ($line in $staged) {
            Write-Detail "    $line"
        }
    }

    $unstaged = @(ConvertTo-StringArray (Invoke-GitCommand -Path $Path diff --stat).Output)
    if ($unstaged) {
        Write-Detail "  Unstaged diff:"
        foreach ($line in $unstaged) {
            Write-Detail "    $line"
        }
    }
}

function Commit-IfDirty {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Name,
        [string]$ExpectedBranch
    )

    Test-Repository -Path $Path -Name $Name -ExpectedBranch $ExpectedBranch | Out-Null

    $status = @(Get-GitStatus -Path $Path)
    if (-not $status) {
        Write-Host "[$Name] No changes to commit."
        return
    }

    if ($DryRun) {
        Show-PendingCommit -Path $Path -Name $Name
        return
    }

    Invoke-Git -Path $Path add -A
    Invoke-Git -Path $Path commit -m $Message
    Write-Host "[$Name] Committed."
}

function Push-IfAhead {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Name,
        [string]$ExpectedBranch
    )

    $branch = Test-Repository -Path $Path -Name $Name -ExpectedBranch $ExpectedBranch

    $upstreamResult = Invoke-GitCommand -Path $Path rev-parse --abbrev-ref "@{u}"
    if ($upstreamResult.ExitCode -ne 0) {
        Write-Host "[$Name] No upstream configured; skipping push."
        return
    }

    $upstream = (($upstreamResult.Output | Select-Object -First 1).ToString().Trim())
    $aheadResult = Invoke-GitCommand -Path $Path rev-list --count "$upstream..HEAD"
    if ($aheadResult.ExitCode -ne 0) {
        throw "Could not determine ahead count for '$Name'."
    }

    $ahead = [int](($aheadResult.Output | Select-Object -First 1).ToString().Trim())
    if ($ahead -eq 0) {
        Write-Host "[$Name] Nothing to push."
        return
    }

    if ($DryRun) {
        Write-Host "[$Name] Would push $ahead commit(s): $branch -> $upstream"
        $commits = ConvertTo-StringArray (Invoke-GitCommand -Path $Path log --oneline "$upstream..HEAD").Output
        foreach ($commit in $commits) {
            Write-Detail "    $commit"
        }
        return
    }

    Invoke-Git -Path $Path push
    Write-Host "[$Name] Pushed $ahead commit(s)."
}

function Invoke-Main {
    if ($Help) {
        Show-Usage
        return
    }

    if ([string]::IsNullOrWhiteSpace($Message)) {
        Show-Usage
        exit 1
    }

    if (-not (Test-Path -LiteralPath $RiseRoot -PathType Container)) {
        throw "RISE root does not exist: $RiseRoot"
    }

    if (-not (Test-GitRepository -Path $RiseRoot)) {
        throw "Not a Git repository: $RiseRoot"
    }

    $repositories = @(Get-SubmoduleRepositories -Root $RiseRoot)

    Write-Host "Main repository: $RiseRoot"
    if ($repositories.Count -eq 0) {
        Write-Host "Submodules: (none)"
    }
    else {
        Write-Host "Submodules: $($repositories.Path -join ', ')"
    }

    if ($DryRun) {
        Write-Host "Mode: dry run"
    }

    Write-Step "Commit submodule repositories"
    foreach ($repository in $repositories) {
        $path = Join-Path $RiseRoot $repository.Path
        Commit-IfDirty `
            -Path $path `
            -Name $repository.Name `
            -ExpectedBranch $repository.Branch
    }

    Write-Step "Commit main repository"
    $mainStatus = @(Get-GitStatus -Path $RiseRoot)
    if (-not $mainStatus) {
        Write-Host "[RISE-2026] No changes to commit."
    }
    elseif ($DryRun) {
        Show-PendingCommit -Path $RiseRoot -Name "RISE-2026"
    }
    else {
        Invoke-Git -Path $RiseRoot add -A
        Invoke-Git -Path $RiseRoot commit -m $Message
        Write-Host "[RISE-2026] Committed."
    }

    Write-Step "Push submodule repositories"
    foreach ($repository in $repositories) {
        $path = Join-Path $RiseRoot $repository.Path
        Push-IfAhead `
            -Path $path `
            -Name $repository.Name `
            -ExpectedBranch $repository.Branch
    }

    Write-Step "Push main repository"
    $mainBranch = Get-CurrentBranch -Path $RiseRoot
    if ([string]::IsNullOrWhiteSpace($mainBranch)) {
        throw "[RISE-2026] Detached HEAD; refusing to push main repository."
    }

    Push-IfAhead `
        -Path $RiseRoot `
        -Name "RISE-2026" `
        -ExpectedBranch $mainBranch

    Write-Host ""
    Write-Host "Done."
}

try {
    Invoke-Main
}
catch {
    Write-Error $_.Exception.Message
    exit 1
}

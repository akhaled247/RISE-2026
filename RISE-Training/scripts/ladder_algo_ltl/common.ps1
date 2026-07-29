# Shared settings for ladder_algo_ltl training scripts (PowerShell).
# Six envs: PointLTL{0,1,3}MASAR1[-WC]-v0

$script:LadderAlgoLtlRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

if (-not $env:DEVICE) { $env:DEVICE = "cuda" }
if (-not $env:DEVICE_ID) { $env:DEVICE_ID = "1" }
if (-not $env:SEED) { $env:SEED = "0" }
if (-not $env:LOG_ROOT) { $env:LOG_ROOT = "./_training_logs/safepo" }

$script:LadderJobs = @(
    @{ Level = "0"; Variant = "plain" },
    @{ Level = "0"; Variant = "wc" },
    @{ Level = "1"; Variant = "plain" },
    @{ Level = "1"; Variant = "wc" },
    @{ Level = "3"; Variant = "plain" },
    @{ Level = "3"; Variant = "wc" }
)

function Get-LadderTask {
    param([string]$Level, [string]$Variant)
    if ($Variant -eq "wc") {
        return "PointLTL${Level}MASAR1WC-v0"
    }
    return "PointLTL${Level}MASAR1-v0"
}

function Get-LadderExperiment {
    param([string]$Algo, [string]$Level, [string]$Variant)
    $seed = $env:SEED
    if ($Variant -eq "wc") {
        return "${Algo}_ltl_l${Level}wc_s${seed}"
    }
    return "${Algo}_ltl_l${Level}_s${seed}"
}

function Get-FilteredLadderJobs {
    if ($env:LEVEL -and $env:VARIANT) {
        return @(@{ Level = $env:LEVEL; Variant = $env:VARIANT })
    }
    return $script:LadderJobs
}

function Invoke-LadderTrain {
    param(
        [string]$AlgoScript,
        [string[]]$ExtraArgs
    )
    Set-Location $script:LadderAlgoLtlRoot
    $jobs = Get-FilteredLadderJobs
    foreach ($job in $jobs) {
        & python @("train/$AlgoScript") @ExtraArgs
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}

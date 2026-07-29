# SafePO TRPO-Lag — PointLTL{0,1,3}MASAR1[-WC]-v0 (6 envs, fixed hyperparams).
# Run in its own terminal:  .\scripts\ladder_algo_ltl\train_trpo_lag.ps1
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\common.ps1"

$jobs = Get-FilteredLadderJobs
Set-Location $script:LadderAlgoLtlRoot

foreach ($job in $jobs) {
    $task = Get-LadderTask -Level $job.Level -Variant $job.Variant
    $experiment = Get-LadderExperiment -Algo "trpo_lag" -Level $job.Level -Variant $job.Variant
    Write-Host "======== TRAIN trpo_lag $task  exp=$experiment  gpu=$($env:DEVICE_ID) ========"

    python train/trpo_lag_train_env.py `
        --task $task --seed $env:SEED `
        --experiment $experiment `
        --log-dir $env:LOG_ROOT `
        --total-steps 5000000 --num-envs 8 --steps-per-epoch 65536 `
        --actor-lr 5e-5 --critic-lr 1e-3 `
        --batch-size 256 --learning-iters 1 `
        --target-kl 0.05 --gamma 0.995 --lam 0.98 --lam-c 0.98 `
        --max-grad-norm 40 --hidden-sizes 64 64 `
        --cost-limit 0.25 `
        --lagrangian-multiplier-init 0.25 `
        --lagrangian-multiplier-lr 0.01 `
        --save-model-freq 10 `
        --device $env:DEVICE --device-id $env:DEVICE_ID `
        --write-terminal False --use-tensorboard True `
        --parallel True

    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "TRPO-Lag ladder done (seed=$($env:SEED), log_root=$($env:LOG_ROOT))."

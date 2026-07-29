# SafePO PPO — PointLTL{0,1,3}MASAR1[-WC]-v0 (6 envs, fixed hyperparams).
# Run in its own terminal:  .\scripts\ladder_algo_ltl\train_ppo.ps1
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\common.ps1"

$jobs = Get-FilteredLadderJobs
Set-Location $script:LadderAlgoLtlRoot

foreach ($job in $jobs) {
    $task = Get-LadderTask -Level $job.Level -Variant $job.Variant
    $experiment = Get-LadderExperiment -Algo "ppo" -Level $job.Level -Variant $job.Variant
    Write-Host "======== TRAIN ppo $task  exp=$experiment  gpu=$($env:DEVICE_ID) ========"

    python train/ppo_train_env.py `
        --task $task --seed $env:SEED `
        --experiment $experiment `
        --log-dir $env:LOG_ROOT `
        --total-steps 5000000 --num-envs 8 --steps-per-epoch 65536 `
        --actor-lr 5e-5 --critic-lr 1e-3 `
        --batch-size 256 --learning-iters 10 `
        --target-kl 0.05 --gamma 0.995 --lam 0.98 --lam-c 0.98 `
        --clip-ratio 0.2 --max-grad-norm 40 --hidden-sizes 64 64 `
        --cost-limit 0.0 `
        --lagrangian-multiplier-init 1.0 `
        --lagrangian-multiplier-lr 0.01 `
        --save-model-freq 10 `
        --device $env:DEVICE --device-id $env:DEVICE_ID `
        --write-terminal False --use-tensorboard True `
        --parallel True --lr_end_factor 1.0 --ent-coef 0.0

    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "PPO ladder done (seed=$($env:SEED), log_root=$($env:LOG_ROOT))."

# RISE-Training

SafePO training, evaluation, and CMDP adapters for [SpecRLBench](../SpecRLBench). SpecRLBench remains benchmark-only (envs + SAR fork).

## Install (from RISE-2026 root, venv active)

```bash
cd SpecRLBench
pip install -e .
pip install -e specbench/envs/zones/safety-gymnasium

cd ../Safe-Policy-Optimization
pip install -e . --no-deps

cd ../RISE-Training
pip install -r requirements.txt
pip install -e .
```

Optional RND: `pip install -e ../RISE-RND`

## Smoke train

```bash
cd RISE-Training
python train/ppo_train_env.py --task PointLTL1MASAR1WC-v0 --seed 0 \
  --total-steps 40000 --num-envs 8 --steps-per-epoch 16384 --device cpu
```

## Eval

```bash
python eval_safepo_sa_env.py --run-dir ./_training_logs/safepo/... --eval-episodes 50
python eval_safepo_ma_env.py --run-dir ./_training_logs/safepo/... --eval-episodes 50
```

Logs default to `RISE-Training/_training_logs/safepo/` (override with `RISE_LOG_ROOT`).

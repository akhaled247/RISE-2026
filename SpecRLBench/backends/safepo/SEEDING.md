# SafePO / SpecRLBench seeding

Three different seeds matter. Do not conflate them.

## Train seed (`--seed`)

Passed into SafePO `args.seed` and SpecRL `make_specrlbench_sa_env(..., seed=)`.
SafePO also seeds `random` / `numpy` / `torch` at the start of `main()`.

Log folder name includes `seed-NNN` (zero-padded).

## Eval seed (`eval_safepo_env.py --seed`)

`backends.safepo.evaluate.eval_single_run` seeds `random`, `numpy`, and `torch`
when `seed` is not `None`, then builds a **1-env** eval CMDP with that seed.

Each eval episode calls `env.reset(seed=ep_seed)` and increments `ep_seed`, so
episode layouts differ when the WC/SAR wrapper honors `reset(seed=)`.

## Layout / WC seed quirk

SpecRLBench SAR wrappers only apply the MuJoCo / keepout layout seed when
`reset(seed=...)` is passed. Calling vec `seed(n)` then `reset()` **without**
a seed argument does **not** reseed building/casualty placement.

For reproducible eval episodes: always pass `seed=` into `reset` (eval path
already does this). Train-time vec envs follow SafePO’s factory seeding.

## Obs normalizer

Eval loads the run’s `*.pkl` Normalizer into `ObsNormalizeWrapper` with
`training=False` (frozen RMS). Seed does not affect loaded RMS stats.

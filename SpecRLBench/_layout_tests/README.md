# SAR layout mode comparison

Set `SAR_LAYOUT_MODE` to one of `current`, `A`, `B`, or `C`, then run the matrix below.
Save output per mode for side-by-side review.

| Mode | Strategy |
|------|----------|
| `current` | Pin buildings + refresh placements + perimeter apply (baseline) |
| `A` | Teleport buildings/perimeter after layout; resample overlapping ring walls |
| `B` | Exclude perimeter/building LTL walls from layout; pin buildings for keepout |
| `C` | Same as current + slow-reset warning (>2s) |

## Test matrix

Run from `SpecRLBench/` on Linux (`specbench` conda).

```bash
MODE=current  # repeat for A, B, C
export SAR_LAYOUT_MODE=$MODE

# 1 — basic reset + timing (seeds 0-7)
python -u scripts/diagnose_sar_layout.py --env PointLTL5MASAR1Debug-v0 --seeds 0-7 \
  | tee _layout_tests/${MODE}_basic.txt

# 2 — perimeter check
python -u scripts/diagnose_sar_layout.py --env PointLTL5MASAR1Debug-v0 --seeds 0-7 \
  --check-perimeter | tee _layout_tests/${MODE}_perimeter.txt

# 3 — building variance across seeds
python -u scripts/diagnose_sar_layout.py --env PointLTL5MASAR1Debug-v0 --seeds 0-7 \
  --check-building-variance | tee _layout_tests/${MODE}_variance.txt

# 4 — ring wall vs building clearance
python -u scripts/diagnose_sar_layout.py --env PointLTL5MASAR1Debug-v0 --seeds 0-7 \
  --check-wall-clearance | tee _layout_tests/${MODE}_clearance.txt

# 5 — visual (human render)
SAR_LAYOUT_MODE=$MODE python debug_env.py

# 6 — 8-vec SubprocVecEnv smoke
python -u scripts/diagnose_sar_layout.py --env PointLTL5MASAR1Debug-v0 --vec --n-envs 8 --seeds 0 \
  | tee _layout_tests/${MODE}_vec8.txt
```

## Pass criteria

1. All seeds reset in <5s, no traceback
2. `--check-perimeter`: each `ltl_wall{i}` within 0.15m of geom corner
3. `--check-building-variance`: >=3 unique building XY across 8 seeds
4. `--check-wall-clearance`: ring walls >= building_keepout + wall_keepout + margin
5. Visual: square perimeter frame + building on border
6. Vec8: workers reset without hang

## After picking a mode

Tell which mode won; remove `SAR_LAYOUT_MODE` flag and delete unused code paths.

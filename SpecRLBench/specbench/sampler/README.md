# Specification Samplers

Samplers for generating LTL (Linear Temporal Logic) specifications as sequences of goals.

## Quick Start

### ReachOnlySampler Example

```python
from specbench.sampler import ReachOnlySampler

sampler = ReachOnlySampler(seed=42)
aps = ['a', 'b', 'c', 'd']

# Single AP per goal
f1 = sampler.sample(aps, num_branches=1, goal_sequence_length=3)
print(f1.to_str(format='spot'))  # F(a & F(c & Fb))

# Multiple APs per goal (disjunction)
f2 = sampler.sample(aps, num_branches=2, goal_sequence_length=2)
print(f2.to_str(format='spot'))  # F((a | b) & F(c | d))
```

### ReachAvoidSampler Example

```python
from specbench.sampler import ReachAvoidSampler

sampler = ReachAvoidSampler(seed=42)
aps = ['a', 'b', 'c', 'd', 'e', 'f']

# Reach 1 AP while avoiding 1 AP
f1 = sampler.sample(aps, num_branches=1, num_avoid=1, goal_sequence_length=2)
print(f1.to_str(format='spot'))  # !b U (a & !c U d)

# Reach 2 APs while avoiding 1 AP
f2 = sampler.sample(aps, num_branches=2, num_avoid=1, goal_sequence_length=2)
print(f2.to_str(format='spot'))  # !d U ((a | c) & !e U (b | f))
```

## ReachOnlySampler

Generates reach-only specifications where each goal is a set of atomic propositions to eventually reach. Formulas follow the pattern:

```
F (goal1 & F (goal2 & F goal3))
```

where `F` is the eventually operator.

## ReachAvoidSampler

Generates reach-avoid specifications where each goal has a reach set (APs to reach) and an avoid set (APs to avoid). Formulas follow the pattern:

```
(!avoid1) U (reach1 & (!avoid2 U reach2))
```

This encodes: "reach goal1 while avoiding avoid1, then reach goal2 while avoiding avoid2".

## Goal Constraints

**ReachOnlySampler:**
- Consecutive goals are guaranteed to be different (goal_i ≠ goal_{i+1})

**ReachAvoidSampler:**
- Reach goals are unique across the sequence (reach_i ≠ reach_j for i ≠ j)
- Reach and avoid sets must be disjoint within each goal
- A reach goal cannot equal any previous avoid goal (prevents impossible constraints)

## Persistence and Playback

Both samplers support save/load for deterministic replay:

```python
# Save generated formulas
sampler.save('formulas.json')

# Load and replay (in playback mode, sample() returns saved formulas)
sampler2 = ReachAvoidSampler(seed=999)
sampler2.load('formulas.json')
replay1 = sampler2.sample(aps, num_branches=2, num_avoid=1, goal_sequence_length=2)
```

In playback mode, `sample()` arguments must match the recorded call exactly. This ensures deterministic reproducibility across code updates.

"""
ReachOnlySampler for generating reach-only specifications.

Generates sequences of goals represented as LTL formulas of the form:
F (goal1 & F (goal2 & F goal3))

Each goal can be a single atomic proposition or a disjunction of multiple atomic propositions.
"""

import random
from typing import List, FrozenSet, Dict, Any

from .base_sampler import BaseSampler
from .parser import LTLFormula, F


SAMPLE_GOAL_MAX_ATTEMPTS = 10_000

class ReachOnlySampler(BaseSampler):
    """Generates reach-only specifications as sequences of goals in LTL form."""

    def _sample(
        self,
        atomic_propositions: List[str],
        num_branches: int,
        goal_sequence_length: int,
    ) -> LTLFormula:
        """
        Generate a specification as a sequence of goals.

        Args:
            atomic_propositions: List of available atomic propositions
            num_branches: Number of disjunctions per goal. If 1, each goal is a
                         single atomic proposition. Otherwise, it's a disjunction
                         of randomly sampled atomic propositions.
            goal_sequence_length: Number of goals in the sequence

        Returns:
            LTLFormula representing the specification

        Raises:
            ValueError: If invalid parameters are provided
        """
        if goal_sequence_length <= 0:
            raise ValueError("goal_sequence_length must be positive")
        if num_branches <= 0:
            raise ValueError("num_branches must be positive")
        if len(atomic_propositions) == 0:
            raise ValueError("atomic_propositions cannot be empty")

        # Generate goals ensuring subsequent goals are different
        goals: List[FrozenSet[str]] = []
        prev_goal_aps: FrozenSet[str] = frozenset()

        for _ in range(goal_sequence_length):
            # Sample goal_aps ensuring it's different from previous goal
            for attempt in range(SAMPLE_GOAL_MAX_ATTEMPTS):
                num_aps_to_sample = min(num_branches, len(atomic_propositions))
                goal_aps = self.rng.sample(atomic_propositions, num_aps_to_sample)
                goal_aps_set = frozenset(goal_aps)

                # Ensure different from previous goal
                if goal_aps_set != prev_goal_aps:
                    break
            else:
                # Loop exhausted without finding a different goal
                raise RuntimeError(
                    f"Failed to generate a different goal after {SAMPLE_GOAL_MAX_ATTEMPTS} attempts. "
                    f"This may happen when num_branches equals or exceeds the number of "
                    f"atomic propositions, making it impossible to generate distinct goals."
                )

            goals.append(goal_aps_set)
            prev_goal_aps = goal_aps_set

        # Build LTL formula
        formula = self._build_formula(goals)
        
        return formula

    def _build_formula(self, goals: List[FrozenSet[str]]) -> LTLFormula:
        """
        Build LTL formula from goals.

        Constructs: F (goal1 & F (goal2 & F goal3))

        Args:
            goals: List of goals, where each goal is a frozenset of atomic propositions

        Returns:
            LTLFormula representing the specification
        """
        if not goals:
            raise ValueError("Goals list cannot be empty")

        # Build the deepest goal (innermost)
        last_goal_aps = goals[-1]
        formula = self._build_goal(last_goal_aps)

        # Wrap in EVENTUALLY
        formula = F.EVENTUALLY(formula)

        # Work backwards through remaining goals
        for i in range(len(goals) - 2, -1, -1):
            goal_aps = goals[i]
            goal_formula = self._build_goal(goal_aps)

            # Create: goal_i & F(...)
            formula = F.AND(goal_formula, formula)

            # Wrap in EVENTUALLY: F(goal_i & F(...))
            formula = F.EVENTUALLY(formula)

        return formula

    def _build_goal(self, aps: FrozenSet[str]) -> LTLFormula:
        """
        Build a goal formula from a set of atomic propositions.

        If multiple APs, creates a disjunction: ap1 | ap2 | ...
        If single AP, just returns the AP.

        Args:
            aps: frozenset of atomic propositions

        Returns:
            LTLFormula representing the goal
        """
        if not aps:
            raise ValueError("Cannot build goal from empty atomic propositions")

        ap_list = list(aps)
        if len(ap_list) == 1:
            return F.AP(ap_list[0])

        # Create disjunction: ap1 | ap2 | ... | apn
        formula = F.AP(ap_list[0])
        for ap in ap_list[1:]:
            formula = F.OR(formula, F.AP(ap))

        return formula

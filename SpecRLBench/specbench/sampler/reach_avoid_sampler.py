"""
ReachAvoidSampler for generating reach-avoid specifications.

Generates sequences of (reach, avoid) goal pairs represented as LTL formulas.
The formula structure uses UNTIL (U) operators to represent the constraints.

For a sequence of goals: [(reach1, avoid1), (reach2, avoid2)]
The formula is: (!avoid1) U (reach1 & (!avoid2 U reach2))

Constraints:
- A reach goal cannot be equal to the previous reach goal
- An atomic proposition cannot be both a reach and avoid goal for the same step
- A reach goal cannot be equal to any of the previous avoid goals
"""

import random
from typing import List, FrozenSet, Dict, Any, Tuple

from .base_sampler import BaseSampler
from .parser import LTLFormula, F, Token, LTLFormulaBinaryOp, LTLFormulaLeaf


SAMPLE_GOAL_MAX_ATTEMPTS = 10_000


class ReachAvoidSampler(BaseSampler):
    """Generates reach-avoid specifications as sequences of (reach, avoid) goal pairs."""

    def _sample(
        self,
        atomic_propositions: List[str],
        num_branches: int,
        num_avoid: int,
        goal_sequence_length: int,
    ) -> LTLFormula:
        """
        Generate a reach-avoid specification as a sequence of goals.

        Args:
            atomic_propositions: List of available atomic propositions
            num_branches: Number of atomic propositions per reach goal
            num_avoid: Number of atomic propositions per avoid goal
            goal_sequence_length: Number of (reach, avoid) goal pairs in the sequence

        Returns:
            LTLFormula representing the reach-avoid specification

        Raises:
            ValueError: If invalid parameters are provided
            RuntimeError: If unable to generate valid goals due to constraints
        """
        if goal_sequence_length <= 0:
            raise ValueError("goal_sequence_length must be positive")
        if num_branches <= 0:
            raise ValueError("num_branches must be positive")
        if num_avoid <= 0:
            raise ValueError("num_avoid must be positive")
        if len(atomic_propositions) == 0:
            raise ValueError("atomic_propositions cannot be empty")

        # Generate goal pairs ensuring constraints
        goals: List[Tuple[FrozenSet[str], FrozenSet[str]]] = []
        prev_reach_goal: FrozenSet[str] = frozenset()
        all_prev_avoid_goals: FrozenSet[str] = frozenset()

        for _ in range(goal_sequence_length):
            # Sample reach goal ensuring:
            # 1. Different from previous reach goal
            # 2. Not in any previous avoid goals
            reach_goal = self._sample_reach_goal(
                atomic_propositions,
                num_branches,
                prev_reach_goal,
                all_prev_avoid_goals,
            )

            # Sample avoid goal ensuring:
            # 1. Disjoint from reach goal
            avoid_goal = self._sample_avoid_goal(
                atomic_propositions,
                num_avoid,
                reach_goal,
            )

            goals.append((reach_goal, avoid_goal))
            prev_reach_goal = reach_goal
            all_prev_avoid_goals = all_prev_avoid_goals | avoid_goal

        # Build LTL formula
        formula = self._build_formula(goals)

        return formula

    def _sample_reach_goal(
        self,
        atomic_propositions: List[str],
        num_branches: int,
        prev_reach_goal: FrozenSet[str],
        all_prev_avoid_goals: FrozenSet[str],
    ) -> FrozenSet[str]:
        """
        Sample a reach goal with constraints.

        Constraints:
        - Different from previous reach goal
        - Not equal to any of the previous avoid goals

        Args:
            atomic_propositions: List of available atomic propositions
            num_branches: Number of APs to sample (up to)
            prev_reach_goal: The previous reach goal to avoid
            all_prev_avoid_goals: All avoid goals from previous steps

        Returns:
            FrozenSet of atomic propositions representing the reach goal

        Raises:
            RuntimeError: If unable to find a valid reach goal after max attempts
        """
        for attempt in range(SAMPLE_GOAL_MAX_ATTEMPTS):
            num_aps_to_sample = min(num_branches, len(atomic_propositions))
            sampled_aps = self.rng.sample(atomic_propositions, num_aps_to_sample)
            reach_goal = frozenset(sampled_aps)

            # Check constraints
            if reach_goal != prev_reach_goal and reach_goal.isdisjoint(all_prev_avoid_goals):
                return reach_goal
        else:
            raise RuntimeError(
                f"Failed to generate a valid reach goal after {SAMPLE_GOAL_MAX_ATTEMPTS} attempts. "
                f"This may happen when num_branches is too large relative to the number of atomic propositions, "
                f"or when there are many previous avoid goals that constrain the space."
            )

    def _sample_avoid_goal(
        self,
        atomic_propositions: List[str],
        num_avoid: int,
        reach_goal: FrozenSet[str],
    ) -> FrozenSet[str]:
        """
        Sample an avoid goal with constraints.

        Constraint:
        - Disjoint from the reach goal (no AP can be both reach and avoid)

        Args:
            atomic_propositions: List of available atomic propositions
            num_avoid: Number of APs to sample (up to)
            reach_goal: The reach goal to avoid overlap with

        Returns:
            FrozenSet of atomic propositions representing the avoid goal

        Raises:
            RuntimeError: If unable to find a valid avoid goal after max attempts
        """
        for attempt in range(SAMPLE_GOAL_MAX_ATTEMPTS):
            # Get APs not in reach goal
            available_for_avoid = [ap for ap in atomic_propositions if ap not in reach_goal]

            if not available_for_avoid:
                # All APs are in reach goal, return empty avoid goal
                return frozenset()

            num_aps_to_sample = min(num_avoid, len(available_for_avoid))
            sampled_aps = self.rng.sample(available_for_avoid, num_aps_to_sample)
            avoid_goal = frozenset(sampled_aps)

            # Check that it's disjoint from reach goal
            if avoid_goal.isdisjoint(reach_goal):
                return avoid_goal
        else:
            raise RuntimeError(
                f"Failed to generate a valid avoid goal after {SAMPLE_GOAL_MAX_ATTEMPTS} attempts. "
                f"This should rarely happen unless all APs are in the reach goal."
            )

    def _build_formula(
        self,
        goals: List[Tuple[FrozenSet[str], FrozenSet[str]]],
    ) -> LTLFormula:
        """
        Build LTL formula from reach-avoid goal pairs.

        Constructs: (!avoid_1) U (reach_1 & (!avoid_2 U reach_2))

        Args:
            goals: List of (reach, avoid) pairs

        Returns:
            LTLFormula representing the specification
        """
        if not goals:
            raise ValueError("Goals list cannot be empty")

        # Build the deepest goal (innermost)
        last_reach, last_avoid = goals[-1]
        reach_formula = self._build_goal(last_reach)
        formula = reach_formula

        # Work backwards through remaining goals
        for i in range(len(goals) - 2, -1, -1):
            reach_i, avoid_i = goals[i]
            reach_formula_i = self._build_goal(reach_i)

            # Get the avoid formula for the next step
            avoid_next = goals[i + 1][1]
            avoid_formula_next = self._build_negated_goal(avoid_next)

            # Create: (!avoid_{i+1} U reach_i)
            formula = self._build_until(avoid_formula_next, reach_formula_i)

            # Wrap in AND with reach_i: reach_i & (...)
            # Wait, I need to reconsider the structure...

        # Actually, let me reconsider the formula structure
        # For [(reach1, avoid1), (reach2, avoid2)]:
        # (!avoid1) U (reach1 & (!avoid2 U reach2))
        #
        # So the pattern from right to left is:
        # 1. Start with reach2
        # 2. Create (!avoid2 U reach2)
        # 3. Create (reach1 & (...))
        # 4. Create (!avoid1 U (...))

        # Build the deepest goal
        last_reach, last_avoid = goals[-1]
        formula = self._build_goal(last_reach)

        # Work backwards
        for i in range(len(goals) - 2, -1, -1):
            reach_i, avoid_i = goals[i]
            avoid_next = goals[i + 1][1]

            # Create negated avoid for next step
            avoid_formula_next = self._build_negated_goal(avoid_next)

            # Create: (!avoid_{i+1} U formula)
            formula = self._build_until(avoid_formula_next, formula)

            # Create: reach_i & formula
            reach_formula_i = self._build_goal(reach_i)
            formula = F.AND(reach_formula_i, formula)

        # Finally, wrap the entire thing with (!avoid1 U ...)
        avoid_1 = goals[0][1]
        avoid_formula_1 = self._build_negated_goal(avoid_1)
        formula = self._build_until(avoid_formula_1, formula)

        return formula

    def _build_goal(self, aps: FrozenSet[str]) -> LTLFormula:
        """
        Build a goal formula from a set of atomic propositions.

        If multiple APs, creates a disjunction: ap1 | ap2 | ...
        If single AP, just returns the AP.
        If empty, returns TRUE (empty goal means "already satisfied").

        Args:
            aps: frozenset of atomic propositions

        Returns:
            LTLFormula representing the goal
        """
        if not aps:
            # Empty goal - return TRUE as it's always satisfied
            return LTLFormulaLeaf(Token.TRUE)

        ap_list = list(aps)
        if len(ap_list) == 1:
            return F.AP(ap_list[0])

        # Create disjunction: ap1 | ap2 | ... | apn
        formula = F.AP(ap_list[0])
        for ap in ap_list[1:]:
            formula = F.OR(formula, F.AP(ap))

        return formula

    def _build_negated_goal(self, aps: FrozenSet[str]) -> LTLFormula:
        """
        Build a negated goal formula from a set of atomic propositions.

        If multiple APs, creates: !ap1 & !ap2 & ...
        If single AP, creates: !ap
        If empty, returns TRUE.

        Args:
            aps: frozenset of atomic propositions

        Returns:
            LTLFormula representing the negated goal
        """
        if not aps:
            # Empty avoid goal - return TRUE (nothing to avoid)
            return LTLFormulaLeaf(Token.TRUE)

        ap_list = list(aps)
        if len(ap_list) == 1:
            return F.NOT(F.AP(ap_list[0]))

        # Create conjunction of negations: !ap1 & !ap2 & ... & !apn
        formula = F.NOT(F.AP(ap_list[0]))
        for ap in ap_list[1:]:
            formula = F.AND(formula, F.NOT(F.AP(ap)))

        return formula

    def _build_until(self, left: LTLFormula, right: LTLFormula) -> LTLFormula:
        """
        Build an UNTIL formula.

        Args:
            left: Left operand of UNTIL
            right: Right operand of UNTIL

        Returns:
            LTLFormula representing (left U right)
        """
        return LTLFormulaBinaryOp(Token.UNTIL, left, right)

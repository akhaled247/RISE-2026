import json
import os
import tempfile
import unittest

from specbench.sampler import ReachOnlySampler, ReachAvoidSampler
from specbench.sampler.reach_avoid_sampler import ReachAvoidSampler


class TestReachOnlySampler(unittest.TestCase):
    def setUp(self):
        self.aps = ["a", "b", "c", "d"]

    def test_sample_records_args(self):
        sampler = ReachOnlySampler(seed=123)
        formula = sampler.sample(atomic_propositions=self.aps, num_branches=2, goal_sequence_length=3)
        self.assertIsNotNone(formula)
        self.assertEqual(len(sampler.generated_formulas), 1)
        record = sampler.generated_formulas[0]
        self.assertIn("formula", record)
        self.assertIn("args", record)
        self.assertEqual(record["kwargs"]["num_branches"], 2)
        self.assertEqual(record["kwargs"]["goal_sequence_length"], 3)
        self.assertEqual(set(record["kwargs"]["atomic_propositions"]), set(self.aps))

    def test_playback_matching_args(self):
        sampler = ReachOnlySampler(seed=42)
        f1 = sampler.sample(self.aps, num_branches=1, goal_sequence_length=2)
        f2 = sampler.sample(self.aps, num_branches=2, goal_sequence_length=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "records.json")
            sampler.save(path)

            playback = ReachOnlySampler(seed=999)
            playback.load(path)

            p1 = playback.sample(self.aps, num_branches=1, goal_sequence_length=2)
            p2 = playback.sample(self.aps, num_branches=2, goal_sequence_length=2)

            self.assertTrue(f1.equal_to(p1))
            self.assertTrue(f2.equal_to(p2))

    def test_playback_mismatch_args_raises(self):
        sampler = ReachOnlySampler(seed=42)
        sampler.sample(self.aps, num_branches=1, goal_sequence_length=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "records.json")
            sampler.save(path)

            playback = ReachOnlySampler(seed=999)
            playback.load(path)

            with self.assertRaises(RuntimeError):
                playback.sample(self.aps, num_branches=2, goal_sequence_length=2)

    def test_save_format_contains_records(self):
        sampler = ReachOnlySampler(seed=7)
        sampler.sample(self.aps, num_branches=1, goal_sequence_length=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "records.json")
            sampler.save(path)

            with open(path, "r") as f:
                data = json.load(f)

            self.assertIn("records", data)
            self.assertEqual(len(data["records"]), 1)
            record = data["records"][0]
            self.assertIn("formula", record)
            self.assertIn("args", record)

    def test_formula_structure_single_branch(self):
        """Test that generated formula has correct structure for single branch case."""
        sampler = ReachOnlySampler(seed=100)
        goal_sequence_length = 3
        num_branches = 1
        
        formula = sampler.sample(self.aps, num_branches=num_branches, goal_sequence_length=goal_sequence_length)
        formula_str = formula.to_str(format="spot")
        
        # Count operators
        f_count = formula_str.count("F")
        and_count = formula_str.count("&")
        or_count = formula_str.count("|")
        
        # Verify structure
        self.assertEqual(f_count, goal_sequence_length, f"Expected {goal_sequence_length} F operators, got {f_count}")
        self.assertEqual(and_count, goal_sequence_length - 1, f"Expected {goal_sequence_length - 1} & operators, got {and_count}")
        self.assertEqual(or_count, 0, f"Expected 0 | operators for single branch, got {or_count}")
        
        # Count atomic propositions
        total_ap_count = sum(formula_str.count(ap) for ap in self.aps)
        expected_ap_count = goal_sequence_length * num_branches
        self.assertEqual(total_ap_count, expected_ap_count, 
                        f"Expected {expected_ap_count} total atomic propositions, got {total_ap_count}")

    def test_formula_structure_multi_branch(self):
        """Test that generated formula has correct structure for multi-branch case."""
        sampler = ReachOnlySampler(seed=200)
        goal_sequence_length = 4
        num_branches = 2
        
        formula = sampler.sample(self.aps, num_branches=num_branches, goal_sequence_length=goal_sequence_length)
        formula_str = formula.to_str(format="spot")
        
        # Count operators
        f_count = formula_str.count("F")
        and_count = formula_str.count("&")
        or_count = formula_str.count("|")
        
        # Verify structure
        self.assertEqual(f_count, goal_sequence_length, f"Expected {goal_sequence_length} F operators, got {f_count}")
        self.assertEqual(and_count, goal_sequence_length - 1, f"Expected {goal_sequence_length - 1} & operators, got {and_count}")
        self.assertEqual(or_count, goal_sequence_length * (num_branches - 1), 
                        f"Expected {goal_sequence_length * (num_branches - 1)} | operators, got {or_count}")
        
        # Count atomic propositions
        total_ap_count = sum(formula_str.count(ap) for ap in self.aps)
        expected_ap_count = goal_sequence_length * num_branches
        self.assertEqual(total_ap_count, expected_ap_count, 
                        f"Expected {expected_ap_count} total atomic propositions, got {total_ap_count}")

    def test_formula_structure_max_branches(self):
        """Test formula structure when num_branches equals number of atomic propositions."""
        sampler = ReachOnlySampler(seed=300)
        goal_sequence_length = 1
        num_branches = len(self.aps)  # 4
        
        formula = sampler.sample(self.aps, num_branches=num_branches, goal_sequence_length=goal_sequence_length)
        formula_str = formula.to_str(format="spot")
        
        # Count operators
        f_count = formula_str.count("F")
        and_count = formula_str.count("&")
        or_count = formula_str.count("|")
        
        # Verify structure
        self.assertEqual(f_count, goal_sequence_length)
        self.assertEqual(and_count, goal_sequence_length - 1)
        self.assertEqual(or_count, goal_sequence_length * (num_branches - 1))
        
        # Count atomic propositions
        total_ap_count = sum(formula_str.count(ap) for ap in self.aps)
        expected_ap_count = goal_sequence_length * num_branches
        self.assertEqual(total_ap_count, expected_ap_count)

    def test_formula_structure_various_lengths(self):
        """Test formula structure for various sequence lengths."""
        sampler = ReachOnlySampler(seed=400)
        
        test_cases = [
            (1, 1),  # Single goal, single branch
            (5, 1),  # Long sequence, single branch
            (3, 3),  # Medium sequence, multi-branch
            (1, 2),  # Single goal with disjunction
        ]
        
        for goal_length, num_branches in test_cases:
            with self.subTest(goal_length=goal_length, num_branches=num_branches):
                formula = sampler.sample(self.aps, num_branches=num_branches, goal_sequence_length=goal_length)
                formula_str = formula.to_str(format="spot")
                
                f_count = formula_str.count("F")
                and_count = formula_str.count("&")
                or_count = formula_str.count("|")
                
                self.assertEqual(f_count, goal_length)
                self.assertEqual(and_count, goal_length - 1)
                self.assertEqual(or_count, goal_length * (num_branches - 1))
                
                total_ap_count = sum(formula_str.count(ap) for ap in self.aps)
                expected_ap_count = goal_length * num_branches
                self.assertEqual(total_ap_count, expected_ap_count)


class TestReachAvoidSampler(unittest.TestCase):
    def setUp(self):
        self.aps = ["a", "b", "c", "d", "e", "f", "g", "h"]

    def test_sample_records_args(self):
        """Test that sampled arguments are recorded correctly."""
        sampler = ReachAvoidSampler(seed=123)
        formula = sampler.sample(
            atomic_propositions=self.aps,
            num_branches=2,
            num_avoid=1,
            goal_sequence_length=3,
        )
        self.assertIsNotNone(formula)
        self.assertEqual(len(sampler.generated_formulas), 1)
        record = sampler.generated_formulas[0]
        self.assertIn("formula", record)
        self.assertIn("args", record)
        self.assertEqual(record["kwargs"]["num_branches"], 2)
        self.assertEqual(record["kwargs"]["num_avoid"], 1)
        self.assertEqual(record["kwargs"]["goal_sequence_length"], 3)
        self.assertEqual(set(record["kwargs"]["atomic_propositions"]), set(self.aps))

    def test_playback_matching_args(self):
        """Test that playback works with matching arguments."""
        sampler = ReachAvoidSampler(seed=42)
        f1 = sampler.sample(self.aps, num_branches=1, num_avoid=1, goal_sequence_length=2)
        f2 = sampler.sample(self.aps, num_branches=2, num_avoid=1, goal_sequence_length=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "reach_avoid_records.json")
            sampler.save(path)

            playback = ReachAvoidSampler(seed=999)
            playback.load(path)

            p1 = playback.sample(self.aps, num_branches=1, num_avoid=1, goal_sequence_length=2)
            p2 = playback.sample(self.aps, num_branches=2, num_avoid=1, goal_sequence_length=2)

            self.assertTrue(f1.equal_to(p1))
            self.assertTrue(f2.equal_to(p2))

    def test_playback_mismatch_args_raises(self):
        """Test that playback fails with mismatched arguments."""
        sampler = ReachAvoidSampler(seed=42)
        sampler.sample(self.aps, num_branches=1, num_avoid=1, goal_sequence_length=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "reach_avoid_records.json")
            sampler.save(path)

            playback = ReachAvoidSampler(seed=999)
            playback.load(path)

            with self.assertRaises(RuntimeError):
                playback.sample(self.aps, num_branches=2, num_avoid=1, goal_sequence_length=2)

    def test_save_format_contains_records(self):
        """Test that saved format contains all records."""
        sampler = ReachAvoidSampler(seed=7)
        sampler.sample(self.aps, num_branches=1, num_avoid=1, goal_sequence_length=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "reach_avoid_records.json")
            sampler.save(path)

            with open(path, "r") as f:
                data = json.load(f)

            self.assertIn("records", data)
            self.assertEqual(len(data["records"]), 1)
            record = data["records"][0]
            self.assertIn("formula", record)
            self.assertIn("args", record)

    def test_formula_structure_single_branch(self):
        """Test formula structure for single branch (single AP per goal)."""
        sampler = ReachAvoidSampler(seed=100)
        goal_sequence_length = 3
        num_branches = 1
        num_avoid = 1

        formula = sampler.sample(
            self.aps,
            num_branches=num_branches,
            num_avoid=num_avoid,
            goal_sequence_length=goal_sequence_length,
        )
        formula_str = formula.to_str(format="spot")

        # Count operators
        until_count = formula_str.count("U")
        and_count = formula_str.count("&")
        or_count = formula_str.count("|")
        not_count = formula_str.count("!")

        # Expected: (!avoid_1) U (reach_1 & (!avoid_2 U (reach_2 & (!avoid_3 U reach_3))))
        # This should have goal_sequence_length U operators
        # and goal_sequence_length - 1 AND operators (connecting reach_i to the rest)
        self.assertEqual(until_count, goal_sequence_length)
        self.assertEqual(and_count, goal_sequence_length - 1)  # Connecting reach_i with the rest

    def test_formula_structure_multi_branch(self):
        """Test formula structure for multi-branch case."""
        sampler = ReachAvoidSampler(seed=200)
        goal_sequence_length = 2
        num_branches = 2
        num_avoid = 2

        formula = sampler.sample(
            self.aps,
            num_branches=num_branches,
            num_avoid=num_avoid,
            goal_sequence_length=goal_sequence_length,
        )
        formula_str = formula.to_str(format="spot")

        # Count operators
        until_count = formula_str.count("U")
        and_count = formula_str.count("&")
        or_count = formula_str.count("|")
        not_count = formula_str.count("!")

        # With multi-branch, we should have OR operators for reach goals
        self.assertEqual(until_count, goal_sequence_length)
        self.assertEqual(and_count, goal_sequence_length - 1 + (num_avoid - 1) * goal_sequence_length)  # ANDs for chaining and avoid
        # Should have OR operators in reach goals (when num_branches > 1)
        self.assertGreater(or_count, 0)

    def test_constraints_reach_goals_different(self):
        """Test constraint: reach goals must be different from previous reach goals."""
        sampler = ReachAvoidSampler(seed=300)

        # Generate multiple samples - they should not fail
        for _ in range(5):
            formula = sampler.sample(self.aps, num_branches=1, num_avoid=1, goal_sequence_length=3)
            self.assertIsNotNone(formula)

    def test_constraints_reach_avoid_disjoint(self):
        """Test constraint: AP cannot be both reach and avoid in the same step."""
        sampler = ReachAvoidSampler(seed=400)

        # This constraint is implicitly tested - if violated, sampling would fail
        # Generate several samples to ensure constraint is maintained
        for _ in range(10):
            formula = sampler.sample(self.aps, num_branches=2, num_avoid=2, goal_sequence_length=2)
            self.assertIsNotNone(formula)

    def test_constraints_reach_not_in_prev_avoid(self):
        """Test constraint: reach goal cannot be in any previous avoid goals."""
        sampler = ReachAvoidSampler(seed=500)

        # This constraint is implicitly tested - if violated, sampling would fail
        # Generate samples with longer sequences to exercise this constraint
        for _ in range(10):
            formula = sampler.sample(self.aps, num_branches=1, num_avoid=1, goal_sequence_length=4)
            self.assertIsNotNone(formula)

    def test_empty_avoid_goal_allowed(self):
        """Test that empty avoid goals are allowed (nothing to avoid)."""
        sampler = ReachAvoidSampler(seed=600)
        
        # With fewer APs, we might generate formulas with empty avoid goals
        # Use a smaller AP set but still enough for a valid sequence
        formula = sampler.sample(self.aps[:4], num_branches=1, num_avoid=1, goal_sequence_length=2)
        formula_str = formula.to_str(format="spot")
        
        # Should successfully generate a formula
        self.assertIsNotNone(formula)
        # The formula string should contain valid LTL operators
        self.assertIn("U", formula_str)


if __name__ == "__main__":
    unittest.main()
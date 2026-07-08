"""Base sampler class with playback and save/load functionality."""

import json
import random
from typing import List, Dict, Any

from .parser import ltl_formula, LTLFormula


class BaseSampler:
    """Base class for samplers with playback and save/load functionality."""

    def __init__(self, seed: int):
        """
        Initialize the sampler with a random seed.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = random.Random(seed)
        self.generated_formulas: List[Dict[str, Any]] = []  # Contains dicts with 'formula' and 'args'
        self.playback_index = 0
        self.in_playback_mode = False

    def sample(self, *args, **kwargs) -> LTLFormula:
        """
        Generate a specification with playback mode support.

        This method handles playback mode checking and delegates to _sample()
        for actual formula generation. Subclasses should implement _sample().

        Returns:
            LTLFormula representing the specification

        Raises:
            RuntimeError: If in playback mode and no more formulas are available,
                         or if recorded arguments don't match the function call
        """
        if self.in_playback_mode:
            if self.playback_index >= len(self.generated_formulas):
                raise RuntimeError("No more formulas in playback mode")
            
            record = self.generated_formulas[self.playback_index]
            
            # Verify arguments match
            if (recorded_args := tuple(record.get('args', ()))) != args:
                raise RuntimeError(
                    f"Playback arguments mismatch. "
                    f"Expected: {recorded_args}, "
                    f"Got: {args}"
                )
            if (recorded_kwargs := record.get('kwargs', {})) != kwargs:
                raise RuntimeError(
                    f"Playback keyword arguments mismatch. "
                    f"Expected: {recorded_kwargs}, "
                    f"Got: {kwargs}"
                )
            
            formula = record['formula']
            self.playback_index += 1
            return formula

        # Generate new formula
        formula = self._sample(*args, **kwargs)
        
        # Record formula and arguments
        self.generated_formulas.append({
            'formula': formula,
            'args': args,
            'kwargs': kwargs,
        })
        
        return formula

    def _sample(self, *args, **kwargs) -> LTLFormula:
        """
        Generate a specification. Must be implemented by subclasses.

        Returns:
            LTLFormula representing the specification
        """
        raise NotImplementedError("Subclasses must implement _sample()")

    def save(self, filepath: str) -> None:
        """
        Save generated formulas and their arguments to a JSON file.

        Args:
            filepath: Path to save the JSON file
        """
        records = []
        for record in self.generated_formulas:
            formula_str = record['formula'].to_str(format="spot")
            records.append({
                'formula': formula_str,
                'args': record.get('args', ()),
                'kwargs': record.get('kwargs', {}),
            })

        with open(filepath, "w") as f:
            json.dump({"seed": self.seed, "records": records}, f, indent=2)

    def load(self, filepath: str) -> None:
        """
        Load formulas and their arguments from a JSON file and enter playback mode.

        After loading, calling `sample()` will return previously generated formulas
        in the saved order rather than generating new ones. Arguments must match
        the recorded arguments or a RuntimeError is raised.

        Args:
            filepath: Path to the JSON file to load
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        records = data.get('records', [])

        self.generated_formulas = []
        for record in records:
            formula_str = record['formula']
            formula = ltl_formula(formula_str, format="spot")
            self.generated_formulas.append({
                'formula': formula,
                'args': record.get('args', ()),
                'kwargs': record.get('kwargs', {}),
            })

        self.in_playback_mode = True
        self.playback_index = 0

    def reset_playback(self) -> None:
        """Reset playback index to the beginning."""
        self.playback_index = 0

    def exit_playback_mode(self) -> None:
        """Exit playback mode and allow generating new formulas."""
        self.in_playback_mode = False
        self.playback_index = 0

    def get_generated_formulas(self) -> List[LTLFormula]:
        """Get all generated formulas (without their associated arguments)."""
        return [record['formula'] for record in self.generated_formulas]

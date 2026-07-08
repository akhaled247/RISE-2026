from typing import Any, Dict

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.utils import distance


class LTLReach(Task):
    def __init__(
        self,
        sim,
        get_ee_position,
        reward_type="sparse",
        distance_threshold=0.05,
        goal_range=0.6,
        partial_observability=False,
        observe_vision=False,
        obs_use_ee_only=True,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        
        # LTL-related
        self.colors = ['magenta', 'yellow', 'blue', 'green']
        self.region_radius = 0.06
        self.region_num = 2
        self.keepout = 0.08
        self.region_names = []
        # partial observability
        self.max_dist = None if not partial_observability else 0.25
        self.exp_gain = 2.0
        # vision observation
        self.observe_vision = observe_vision

        # ee only observation
        self.obs_use_ee_only = obs_use_ee_only

        self.goal_range_low = np.array([-0.5, -0.3, self.region_radius])
        self.goal_range_high = np.array([0.2, 0.3, 0.5 + self.region_radius])

        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        for color in self.colors:
            for i in range(self.region_num):
                body_name = f"{color}_{i}"
                self.region_names.append(body_name)
                self.sim.create_sphere(
                    body_name=body_name,
                    radius=self.region_radius,
                    mass=0.0,
                    ghost=False,
                    position=np.zeros(3),
                    rgba_color=self.COLORS[color],
                    collision_group=4,
                    collision_mask=0,
                )

    def get_obs(self) -> np.ndarray:
        return np.array([])  # no task-specific observation

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        return ee_position

    def reset(self) -> None:
        """Reset the task by building and placing the layout."""
        for _ in range(10000):  # Try up to 10,000 times to find a valid layout    
            layout = self._sample_layout()
            if len(layout) == len(self.region_names):
                for region_name, position in layout.items():
                    self.sim.set_base_pose(region_name, position, np.array([0.0, 0.0, 0.0, 1.0]))
                return
        raise ValueError("Failed to sample a valid layout within constraints after 10,000 attempts.")

    def _sample_layout(self) -> dict:
        """Sample positions for regions using a layout sampling approach.

        Returns:
            dict: A dictionary mapping region names to their positions.
        """
        layout = {}
        positions = []

        for region_name in self.region_names:
            for _ in range(100):  # Try up to 10000 times to find a valid position for each region
                position = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
                if self._is_valid_position(position, positions):
                    layout[region_name] = position
                    positions.append(position)
                    break
        return layout

    def _is_valid_position(self, position: np.ndarray, positions: list) -> bool:
        """Check if a position is valid based on existing positions.

        Args:
            position (np.ndarray): The position to validate.
            positions (list): List of existing positions.

        Returns:
            bool: True if the position is valid, False otherwise.
        """
        if not positions:
            return True
        distances = np.linalg.norm(np.array(positions) - position, axis=1)
        return np.all(distances >= self.keepout + 2 * self.region_radius)

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float32)
        else:
            return -d.astype(np.float32)
